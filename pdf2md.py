
# pdf_to_markdown_and_summary_concurrent.py

import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import os
import io
import re
import traceback
import csv
import concurrent.futures
import math
import time
import threading
import hashlib
from collections import Counter
import camelot

#########################
# Global Rate Limiter Variables
#########################
# These globals help enforce the request-per-minute limit.
global_request_lock = threading.Lock()
global_last_request_time = 0  # timestamp of last request (in seconds since epoch)
global_delay = 4  # seconds to wait between requests (60 / rpm)

#########################
# Utility Functions
#########################

def create_safe_filename(text):
    """Create a safe filename from text by removing invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "", text.replace(' ', '_'))[:50]

def images_are_identical(image_path1, image_bytes):
    """Compare a file on disk with image bytes using MD5 hash."""
    if not os.path.exists(image_path1):
        return False
    
    with open(image_path1, "rb") as f1:
        hash1 = hashlib.md5(f1.read()).hexdigest()
    
    hash2 = hashlib.md5(image_bytes).hexdigest()
    return hash1 == hash2

def find_identical_image(image_bytes, images_dir):
    """Find an identical image in the directory by comparing MD5 hashes."""
    if not os.path.exists(images_dir):
        return None
        
    for filename in os.listdir(images_dir):
        if os.path.isfile(os.path.join(images_dir, filename)):
            if images_are_identical(os.path.join(images_dir, filename), image_bytes):
                return filename
    return None

#########################
# Markdown Extraction
#########################

def find_identical_image(image_bytes: bytes, images_dir: str) -> str:
    """
    Stub for image deduplication. Always returns None.
    You can replace this with a real hash-based lookup if desired.
    """
    return None

def get_font_size_stats(pdf_document):
    """
    Scan the entire PDF (excluding first 3 pages) to collect and count all font sizes (rounded).
    """
    font_sizes = []
    doc_for_stats = pdf_document[3:] if len(pdf_document) > 3 else pdf_document
    for page in doc_for_stats:
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    size = span.get("size", 0)
                    if size > 0:
                        font_sizes.append(round(size))
    return Counter(font_sizes)

def get_heading_thresholds(font_counter):
    """
    Determine thresholds for ## and ### headings.
    """
    if not font_counter:
        return 0, 0
    sizes_desc = sorted(font_counter.keys(), reverse=True)
    body_size = font_counter.most_common(1)[0][0]
    section_size = sizes_desc[0] if sizes_desc[0] > body_size else body_size + 1
    subsection_size = (
        sizes_desc[1] if len(sizes_desc) > 1 and sizes_desc[1] > body_size else body_size + 0.5
    )
    return section_size, subsection_size

def extract_tables_with_camelot(pdf_path, page_num, table_images_dir, io_lock):
    """
    Use Camelot to extract tables from the given page.
    Returns a markdown string with tables as both markdown and images.
    """
    md = ""
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream')
        if tables:
            md += "### Tables\n\n"
        for idx, table in enumerate(tables):
            df = table.df
            md += f"#### Table {idx+1}\n\n"
            md += df.to_markdown(index=False) + "\n\n"

            # save table as image
            img_filename = f"page_{page_num+1}_table_{idx+1}.png"
            img_path = os.path.join(table_images_dir, img_filename)
            with io_lock:
                table.to_image(img_path)
            md += f"![Table {idx+1}](table_images/{img_filename})\n\n"

    except Exception as e:
        md += f"**Error extracting tables on page {page_num+1}:** {e}\n\n"
    return md

def generate_md(pdf_path, api_key, output_file=None, max_workers=4):
    """
    Extracts text, tables, images (with OCR), and hyperlinks from a PDF
    and writes out a structured Markdown file per page.
    """
    try:
        genai.configure(api_key=api_key)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        if output_file is None:
            output_file = pdf_name
            output_dir = os.path.join(os.path.dirname(pdf_path), f"{pdf_name}_markdown")
        else:
            output_dir = os.path.dirname(os.path.abspath(output_file)) or os.path.dirname(pdf_path)
            output_dir = os.path.join(output_dir, f"{os.path.basename(output_file)}_markdown")

        images_dir = os.path.join(output_dir, "images")
        table_images_dir = os.path.join(output_dir, "table_images")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(table_images_dir, exist_ok=True)

        # Open PDF
        pdf_document = fitz.open(pdf_path)
        font_counter = get_font_size_stats(pdf_document)
        section_size, subsection_size = get_heading_thresholds(font_counter)
        page_count = len(pdf_document)
        print(f"PDF has {page_count} pages. section_size={section_size}, subsection_size={subsection_size}")

        # Initialize Gemini model
        try:
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
        except Exception:
            model = genai.GenerativeModel("gemini-pro-vision")

        io_lock = threading.Lock()
        api_lock = threading.Lock()
        global_request_lock = threading.Lock()
        global_last_request_time = time.time()
        global_delay = 1.0  # seconds between OCR calls

        def process_page(page_num):
            try:
                page = pdf_document[page_num]
                md = f"#### Page {page_num + 1}\n\n"

                # — Text & Heading Extraction —
                try:
                    for block in page.get_text("dict")["blocks"]:
                        if "lines" not in block:
                            continue
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if not text:
                                    continue
                                size = round(span.get("size", 0))
                                font = span.get("font", "").lower()
                                is_bold = "bold" in font
                                if size >= section_size:
                                    md += f"## {text}\n\n"
                                elif size >= subsection_size or is_bold:
                                    md += f"### {text}\n\n"
                                else:
                                    md += f"{text} "
                            md += "\n"
                except Exception as e:
                    md += f"**Text extraction error:** {e}\n\n"

                # — Table Extraction via Camelot —
                md += extract_tables_with_camelot(pdf_path, page_num, table_images_dir, io_lock)

                # — Image Extraction & OCR —
                try:
                    images = page.get_images(full=True)
                    if images:
                        md += "### Images\n\n"

                    def process_image(args):
                        idx, img_info = args
                        try:
                            xref = img_info[0]
                            base = pdf_document.extract_image(xref)
                            img_bytes = base["image"]
                            width, height = base.get("width", 0), base.get("height", 0)
                            if width * height < 5000:
                                return None

                            pil = Image.open(io.BytesIO(img_bytes))
                            # dedupe
                            with io_lock:
                                dup = find_identical_image(img_bytes, images_dir)
                            if dup:
                                filename = dup
                            else:
                                ext = base.get("ext", "png")
                                filename = f"page_{page_num+1}_img_{idx+1}.{ext}"
                                with io_lock:
                                    with open(os.path.join(images_dir, filename), "wb") as f:
                                        f.write(img_bytes)

                            block = f"![Image {idx+1}](images/{filename})\n\n"
                            # OCR if new
                            if not dup:
                                with api_lock:
                                    with global_request_lock:
                                        now = time.time()
                                        wait = global_delay - (now - global_last_request_time)
                                        if wait > 0:
                                            time.sleep(wait)
                                        global_last_request_time = time.time()
                                    resp = model.generate_content(["Extract all text from this image", pil])
                                text = resp.text.strip()
                                if text:
                                    block += f"**OCR Image {idx+1}:**\n\n{text}\n\n"
                            return block
                        except Exception as ex:
                            return f"**Error processing image {idx+1}:** {ex}\n\n"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(images), max_workers)) as ex:
                        futures = ex.map(process_image, enumerate(images))
                        for result in futures:
                            if result:
                                md += result

                except Exception as e:
                    md += f"**Error extracting images:** {e}\n\n"

                # — Hyperlink Extraction —
                try:
                    links = page.get_links()
                    if links:
                        md += "### Hyperlinks\n\n"
                        for i, link in enumerate(links, 1):
                            ltype = link.get("type", link.get("kind", None))
                            if ltype == fitz.LINK_URI and link.get("uri"):
                                md += f"{i}. [{link['uri']}]({link['uri']})\n"
                            elif ltype == fitz.LINK_GOTO:
                                dest = link.get("page", 0) + 1
                                md += f"{i}. [Go to page {dest}](#page-{dest})\n"
                            else:
                                md += f"{i}. {link}\n"
                        md += "\n"
                except Exception as e:
                    md += f"**Error extracting hyperlinks:** {e}\n\n"

                md += "---\n\n"
                print(f"Processed page {page_num+1}")
                return md

            except Exception as e:
                print(f"Error on page {page_num+1}: {e}")
                return f"**Error processing page {page_num+1}:** {e}\n---\n"

        # Run pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            page_markdowns = list(executor.map(process_page, range(page_count)))

        # Write out
        full_md = f"# {pdf_name}\n\n" + "".join(page_markdowns)
        md_path = f"{output_file}.md"
        with open(md_path, "w", encoding="utf-8") as out_f:
            out_f.write(full_md)
        pdf_document.close()
        print(f"Markdown saved to {md_path}")

    except Exception as e:
        print(f"Failed to generate markdown: {e}\n{traceback.format_exc()}")

#########################
# Summarization (Grouped Based on x with Rate Limiting)
#########################

def generate_group_summary(model, group_text):
    """
    Use the Gemini API model to generate a 2-to-3 line summary for the given group of pages.
    Enforces rate limiting before making the API call.

    Args:
        model: Initialized Gemini model.
        group_text (str): Combined text of the group of pages.

    Returns:
        str: Summary text.
    """
    try:
        # Rate limiting: wait if necessary so that requests are spaced at least 'global_delay' seconds apart.
        global global_request_lock, global_last_request_time, global_delay
        with global_request_lock:
            now = time.time()
            wait_time = global_delay - (now - global_last_request_time)
            if wait_time > 0:
                time.sleep(wait_time)
            global_last_request_time = time.time()

        prompt = (
            "Please summarize the following content in 10 to 15 concise lines, "
            "capturing the main topics and their names and descriptions:\n\n" + group_text
        )
        response = model.generate_content([prompt])
        summary = response.text.strip()
        return summary
    except Exception as e:
        print(f"Error generating summary for a group: {e}")
        return "Summary generation failed."

def split_markdown_into_pages(markdown_file):
    """
    Split the markdown file content by pages based on the marker "## Page".
    Returns a list of tuples (page_id, page_content).
    """
    pages = []
    try:
        with open(markdown_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Assume each page starts with "## Page X"
        split_pages = re.split(r"(## Page \d+)", content)
        combined = []
        for part in split_pages:
            if part.strip():
                combined.append(part.strip())
        for i in range(0, len(combined), 2):
            if i+1 < len(combined):
                page_id = combined[i]  # e.g., "## Page X"
                page_content = combined[i] + "\n" + combined[i+1]
            else:
                page_id = combined[i]
                page_content = combined[i]
            m = re.search(r"## Page (\d+)", page_id)
            if m:
                pid = f"page_{m.group(1)}"
            else:
                pid = f"page_{i//2 + 1}"
            pages.append((pid, page_content))
    except Exception as e:
        print(f"Error splitting markdown into pages: {e}")
    return pages

def group_pages_for_summary(pages, x):
    """
    Group pages based on variable x.

    If the number of pages is less than x, group each page individually (group size = 1).
    Otherwise, set the group size as:
       group_size = ceil(total_pages / x)
    so that if total_pages is 2*x then each group contains 2 pages, if 3*x then 3 pages, etc.

    Args:
        pages (list): List of tuples (page_id, page_text).
        x (int): Threshold variable.

    Returns:
        list: List of tuples (group_id, combined_text) for each group.
    """
    num_pages = len(pages)
    if num_pages < x:
        group_size = 1
    else:
        group_size = math.ceil(num_pages / x)
        group_size = min(60, group_size)
    groups = []
    num_groups = math.ceil(num_pages / group_size)
    for i in range(num_groups):
        group_pages = pages[i * group_size : (i + 1) * group_size]
        group_id = i+1
        combined_text = "\n\n".join(text for _, text in group_pages)
        groups.append((group_id, combined_text))
    return groups, group_size

def summarize_markdown_groups(markdown_file, api_key, x=10, output_dir=None):
    """
    Given a markdown file, split it into pages, group the pages based on variable x,
    and concurrently generate a summary for each group.
    
    If the total number of pages N is less than x then generate one summary per page.
    If N is 2*x then group pages in groups of 2, if N is 3*x then group in groups of 3, etc.
    
    The summaries (group IDs and summary text) are saved in a CSV file.

    Args:
        markdown_file (str): Path to the markdown file.
        api_key (str): Gemini API key.
        x (int): Threshold for grouping pages.
        output_dir (str): Directory to save the CSV output.

    Returns:
        str: Path to the CSV file with summaries.
    """
    try:
        genai.configure(api_key=api_key)
        if output_dir is None:
            base_dir = os.path.dirname(markdown_file)
            output_dir = os.path.join(base_dir, "summaries")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize Gemini model once and share it among threads.
        try:
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}\nTrying 'gemini-pro-vision'...")
            try:
                model = genai.GenerativeModel("gemini-pro-vision")
            except Exception as e2:
                print(f"Error with alternate model: {e2}")
                raise Exception("Failed to initialize Gemini model for summarization.")

        # Split markdown file into pages.
        pages = split_markdown_into_pages(markdown_file)
        total_pages = len(pages)
        print(f"Found {total_pages} pages in the markdown file.")

        # Group pages based on threshold x.
        groups, estimated_group_size = group_pages_for_summary(pages, x)
        estimated_group_size = math.ceil(total_pages / x) if total_pages >= x else 1
        print(f"Created {len(groups)} groups with group size determined by x = {x} (group size ≈ {estimated_group_size}).")

        summaries = []

        # Use ThreadPoolExecutor for concurrent summarization.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_group = {
                executor.submit(generate_group_summary, model, group_text): group_id
                for group_id, group_text in groups
            }
            for future in concurrent.futures.as_completed(future_to_group):
                group_id = future_to_group[future]
                try:
                    summary_text = future.result()
                except Exception as e:
                    summary_text = "Summary generation error."
                    print(f"Error summarizing {group_id}: {e}")
                summaries.append((group_id, summary_text))
                print(f"Summary for {group_id} generated.")

        # Save summaries to CSV.
        csv_file = os.path.join(output_dir, os.path.basename(markdown_file).replace(".md", "_summaries.csv"))
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Group ID", "Summary"])
            for item in summaries:
                writer.writerow(item)

        print(f"Summaries successfully saved to: {csv_file}")
        return estimated_group_size

    except Exception as e:
        print(f"Error occurred during grouped markdown summarization: {e}")
        print(traceback.format_exc())
        return None

#########################
# Main Function
#########################

def main():
    # Get API key from environment variable.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set the GOOGLE_API_KEY environment variable.")
        return

    pdf_path = input("Enter the path to the PDF file: ").strip()
    if not pdf_path:
        print("No PDF path provided.")
        return
        
    # Get maximum number of worker threads
    try:
        max_workers = int(input("Enter maximum number of parallel workers (default: 4): ").strip() or "4")
    except ValueError:
        max_workers = 4
        print("Invalid input, using default of 4 workers.")
        
    # Get output file name (optional)
    output_file = input("Enter output file name without extension (leave empty for default): ").strip()
    if not output_file:
        output_file = None
    else:
        # Store the output file path for later use
        md_file_path = f"{output_file}.md"

    print(f"\nProcessing PDF to generate Markdown: {pdf_path} with {max_workers} workers")
    generate_md(pdf_path, api_key, output_file, max_workers=max_workers)
    print("\nMarkdown file created successfully.")

    # Set variable x as the threshold for grouping.
    # For example, if x=10 and the PDF has 20 pages then group_size becomes ceil(20/10)=2 (i.e., one summary per 2 pages).
    # If the PDF has less than x pages then each page gets its own summary.
    x = int(input("Enter the summary threshold value (x): ").strip())
    
    # Get requests per minute limit (rpm) and compute delay.
    rpm = int(input("Enter the maximum requests per minute (rpm): ").strip())
    global global_delay
    global_delay = 60.0 / rpm  # delay in seconds between requests
    print(f"Rate limiting set to {rpm} rpm ({global_delay:.2f} sec delay between requests).")
    
    # Determine the markdown file path based on whether output_file was provided
    if output_file is None:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        md_file_path = os.path.join(
            os.path.dirname(pdf_path), 
            f"{pdf_name}_markdown", 
            f"{pdf_name}.md"
        )
    else:
        md_file_path = f"{output_file}.md"
    
    print(f"\nNow generating grouped summaries from the Markdown file using threshold x = {x}...")
    csv_file = summarize_markdown_groups(md_file_path, api_key, x)
    
    if csv_file:
        print(f"CSV with grouped page summaries created successfully: {csv_file}")
    else:
        print("Failed to create CSV with summaries.")

if __name__ == "__main__":
    main()

