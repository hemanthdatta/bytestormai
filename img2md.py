import os
import io
import google.generativeai as genai
from PIL import Image
import time
import re

def create_safe_filename(text):
    """Create a safe filename from text by removing invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "", text.replace(' ', '_'))[:50]

def generate_md(image_path, api_key, output_file=None):
    """
    Extract content from an image file into a markdown file with OCR text.
    
    Args:
        image_path (str): Path to the image file
        api_key (str): Google API key for Gemini
        output_file (str): Optional output file name (without .md extension)
    """
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Setup output directory
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if output_file is None:
            output_file = image_name
            output_dir = os.path.join(os.path.dirname(image_path), f"{image_name}_markdown")
        else:
            output_dir = os.path.dirname(os.path.abspath(output_file))
            if not output_dir:  # If output_file has no directory component
                output_dir = os.path.join(os.path.dirname(image_path), f"{output_file}_markdown")
        
        # Create output directories
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Initialize markdown content
        markdown_content = f"# {image_name}\n\n"
        
        # Initialize Gemini model - with error handling
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            print("Trying with model name 'gemini-pro-vision'...")
            try:
                model = genai.GenerativeModel('gemini-pro-vision')
            except Exception as e2:
                print(f"Error with alternate model: {str(e2)}")
                raise Exception("Failed to initialize Gemini model. Check your API key and network connection.")
        
        # Process the image
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Save a copy of the image in the images directory
            img_filename = f"{image_name}{os.path.splitext(image_path)[1]}"
            img_path = os.path.join(images_dir, img_filename)
            image.save(img_path)
            
            # Add image to markdown
            markdown_content += f"## Original Image\n\n"
            markdown_content += f"![{image_name}](images/{img_filename})\n\n"
            
            # Extract text using OCR
            markdown_content += f"## Extracted Text\n\n"
            
            # Use Gemini for OCR
            try:
                response = model.generate_content(["Extract all text from this image", image])
                ocr_text = response.text.strip()
                
                if ocr_text:
                    markdown_content += f"```\n{ocr_text}\n```\n\n"
                else:
                    markdown_content += "No text was extracted from this image.\n\n"
                    
                # Also attempt to get image description/content
                time.sleep(1)  # Brief pause to avoid rate limiting
                try:
                    desc_response = model.generate_content(["Describe what you see in this image in detail", image])
                    description = desc_response.text.strip()
                    
                    if description:
                        markdown_content += f"## Image Description\n\n{description}\n\n"
                except Exception as desc_e:
                    print(f"Error generating image description: {str(desc_e)}")
                
            except Exception as ocr_e:
                markdown_content += f"**OCR Failed:** {str(ocr_e)}\n\n"
                
        except Exception as e:
            markdown_content += f"**Error processing image:** {str(e)}\n\n"
            
        # Save the final markdown file
        markdown_file = f"{output_file}.md"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"Markdown file saved to: {markdown_file}")
        return markdown_file
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

def main():
    # Get API key from environment variable
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Please set the GOOGLE_API_KEY environment variable")
        return
        
    # Get image path from user
    image_path = input("Enter the path to the image file: ")
    
    # Generate markdown from image
    generate_md(image_path, api_key)

if __name__ == "__main__":
    main() 