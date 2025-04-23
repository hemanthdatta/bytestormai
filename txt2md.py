import os
import re
import traceback

def create_safe_filename(text):
    """Create a safe filename from text by removing invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "", text.replace(' ', '_'))[:50]

def generate_md_from_text(text_path, tokens_per_page=500, output_file=None):
    """
    Convert a plain text file to markdown with pagination.
    
    Args:
        text_path (str): Path to the text file
        tokens_per_page (int): Approximate number of tokens per page
        output_file (str): Optional output file name (without .md extension)
    """
    try:
        # Check if file exists
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found: {text_path}")
            
        # Setup output file
        text_name = os.path.splitext(os.path.basename(text_path))[0]
        
        if output_file is None:
            output_file = text_name
            
        # Read the text file
        with open(text_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Initialize markdown content
        markdown_content = f"# {text_name}\n\n"
        
        # Simple tokenization by splitting on whitespace
        # This is an approximation; more sophisticated tokenization could be used
        tokens = content.split()
        total_tokens = len(tokens)
        
        # Calculate number of pages
        num_pages = max(1, (total_tokens + tokens_per_page - 1) // tokens_per_page)
        
        # Process content by pages
        for page in range(num_pages):
            # Add page header
            markdown_content += f"## Page {page + 1}\n\n"
            
            # Calculate token range for this page
            start_token = page * tokens_per_page
            end_token = min((page + 1) * tokens_per_page, total_tokens)
            
            # Get tokens for this page
            page_tokens = tokens[start_token:end_token]
            
            # Convert back to text, preserving some paragraph structure
            page_text = ' '.join(page_tokens)
            
            # Simple paragraph detection (double newlines)
            # This attempts to preserve paragraph breaks from the original text
            paragraphs = re.split(r'\n\s*\n', page_text)
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    markdown_content += f"{paragraph.strip()}\n\n"
        
        # Save the final markdown file
        markdown_file = f"{output_file}.md"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"Markdown file saved to: {markdown_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert text file to markdown with pagination")
    parser.add_argument("text_file", help="Path to the text file to convert")
    parser.add_argument("--tokens", type=int, default=500, help="Approximate tokens per page (default: 500)")
    parser.add_argument("--output", help="Output file name (without .md extension)")
    
    args = parser.parse_args()
    
    generate_md_from_text(args.text_file, args.tokens, args.output) 