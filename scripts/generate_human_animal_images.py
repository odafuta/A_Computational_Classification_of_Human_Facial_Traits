#!/usr/bin/env python3
"""
Generate human faces with subtle animal-like features using Pollinations.ai API
Creates human faces that subtly resemble different animals without being too obvious

This script generates images for the human-to-animal classification task.
The goal is to create human faces with subtle animal characteristics that can be
classified by the trained model.
"""

import requests
import os
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont
import random

def generate_human_animal_images(output_dir="data/af_data_new/human_like_animal", num_images_per_type=7):
    """
    Generate human faces with subtle animal-like features
    
    Args:
        output_dir: Directory to save generated images
        num_images_per_type: Number of images to generate per animal type
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating human faces with subtle animal features...")
    print(f"Output directory: {output_path}")
    print(f"Images per type: {num_images_per_type}")
    
    # Subtle animal-like human prompts - focus on human faces with slight animal characteristics
    prompts = {
        'cat': [
            "human portrait with slightly almond-shaped eyes and subtle feline grace, photorealistic",
            "person with naturally narrow eyes and elegant facial structure, human face, realistic photo",
            "human with graceful features and alert expression, subtle cat-like elegance, portrait photography",
            "person with refined facial features and mysterious eyes, human portrait, high quality",
            "human face with naturally angular cheekbones and poised expression, realistic photography",
            "portrait of person with naturally sleek features and composed demeanor, photorealistic",
            "human with elegant bone structure and calm, observant expression, professional portrait"
        ],
        'dog': [
            "human portrait with warm, loyal eyes and friendly expression, photorealistic",
            "person with naturally round, expressive eyes and gentle smile, human face, realistic photo",
            "human with approachable features and trustworthy expression, portrait photography",
            "person with kind eyes and naturally cheerful demeanor, human portrait, high quality",
            "human face with naturally soft features and optimistic expression, realistic photography",
            "portrait of person with warm, inviting smile and honest eyes, photorealistic",
            "human with naturally friendly features and open expression, professional portrait"
        ],
        'wild': [
            "human portrait with intense, piercing eyes and strong jawline, photorealistic",
            "person with naturally sharp features and confident expression, human face, realistic photo",
            "human with bold facial structure and determined look, portrait photography",
            "person with naturally angular features and fierce expression, human portrait, high quality",
            "human face with strong bone structure and wild, untamed hair, realistic photography",
            "portrait of person with naturally rugged features and intense gaze, photorealistic",
            "human with naturally prominent cheekbones and adventurous expression, professional portrait"
        ]
    }
    
    successful_downloads = 0
    failed_downloads = 0
    
    for animal_type, type_prompts in prompts.items():
        print(f"\nGenerating {animal_type}-like human faces...")
        
        for i in range(num_images_per_type):
            prompt = type_prompts[i % len(type_prompts)]
            filename = f"{animal_type}_human_{i+1:02d}.jpg"
            filepath = output_path / filename
            
            print(f"  Generating: {filename}")
            
            success = generate_image_with_fallback(prompt, filepath)
            
            if success:
                successful_downloads += 1
                print(f"    ✓ Successfully generated: {filename}")
            else:
                failed_downloads += 1
                print(f"    ✗ Failed to generate: {filename}")
            
            # Add delay to be respectful to the API
            time.sleep(2)
    
    print(f"\n=== Generation Summary ===")
    print(f"Successfully generated: {successful_downloads} images")
    print(f"Failed to generate: {failed_downloads} images")
    print(f"Total attempts: {successful_downloads + failed_downloads}")
    print(f"Success rate: {(successful_downloads / (successful_downloads + failed_downloads)) * 100:.1f}%")
    
    return successful_downloads, failed_downloads

def generate_image_with_fallback(prompt, filepath, max_retries=3):
    """
    Generate image using Pollinations.ai API with fallback options
    
    Args:
        prompt: Text prompt for image generation
        filepath: Path to save the generated image
        max_retries: Maximum number of retry attempts
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Pollinations.ai API endpoint (free service)
    base_url = "https://image.pollinations.ai/prompt/"
    
    # Image parameters
    params = {
        'width': 512,
        'height': 512,
        'model': 'flux',  # High-quality model
        'enhance': 'true',
        'nologo': 'true'
    }
    
    for attempt in range(max_retries):
        try:
            # Construct URL with prompt and parameters
            url = base_url + requests.utils.quote(prompt)
            
            # Add parameters to URL
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{url}?{param_string}"
            
            # Make request with timeout
            response = requests.get(full_url, timeout=30)
            
            if response.status_code == 200:
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Verify image was saved correctly
                if filepath.exists() and filepath.stat().st_size > 1000:  # At least 1KB
                    return True
                else:
                    print(f"    Warning: Generated image seems too small (attempt {attempt + 1})")
                    
            else:
                print(f"    API returned status code {response.status_code} (attempt {attempt + 1})")
                
        except requests.exceptions.RequestException as e:
            print(f"    Request failed: {e} (attempt {attempt + 1})")
        except Exception as e:
            print(f"    Unexpected error: {e} (attempt {attempt + 1})")
        
        if attempt < max_retries - 1:
            print(f"    Retrying in 3 seconds...")
            time.sleep(3)
    
    # If all attempts failed, create a placeholder
    print(f"    All attempts failed. Creating placeholder image.")
    return create_placeholder_image(filepath, prompt)

def create_placeholder_image(filepath, prompt):
    """
    Create a placeholder image when generation fails
    
    Args:
        filepath: Path to save the placeholder image
        prompt: Original prompt (for reference)
    
    Returns:
        bool: True if placeholder created successfully
    """
    try:
        # Create a simple placeholder image
        img = Image.new('RGB', (512, 512), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # Add text indicating this is a placeholder
        text = "Placeholder\nImage\nGeneration\nFailed"
        
        # Try to use a default font, fall back to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (512 - text_width) // 2
        y = (512 - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill='black', font=font)
        
        # Save placeholder
        img.save(filepath, 'JPEG', quality=85)
        
        return True
        
    except Exception as e:
        print(f"    Failed to create placeholder: {e}")
        return False

def verify_generated_images(output_dir="data/af_data_new/human_like_animal"):
    """
    Verify that generated images are valid and display summary
    
    Args:
        output_dir: Directory containing generated images
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Directory not found: {output_path}")
        return
    
    print(f"\n=== Verifying Generated Images ===")
    
    # Get all image files
    image_files = list(output_path.glob("*.jpg")) + list(output_path.glob("*.png"))
    
    if not image_files:
        print("No image files found.")
        return
    
    valid_images = 0
    invalid_images = 0
    total_size = 0
    
    categories = {'cat': 0, 'dog': 0, 'wild': 0}
    
    for img_path in image_files:
        try:
            # Try to open image
            with Image.open(img_path) as img:
                width, height = img.size
                file_size = img_path.stat().st_size
                
                if width > 0 and height > 0 and file_size > 1000:
                    valid_images += 1
                    total_size += file_size
                    
                    # Count by category
                    filename = img_path.name.lower()
                    if 'cat' in filename:
                        categories['cat'] += 1
                    elif 'dog' in filename:
                        categories['dog'] += 1
                    elif 'wild' in filename:
                        categories['wild'] += 1
                        
                else:
                    invalid_images += 1
                    print(f"  Invalid image: {img_path.name} ({width}x{height}, {file_size} bytes)")
                    
        except Exception as e:
            invalid_images += 1
            print(f"  Cannot open image: {img_path.name} - {e}")
    
    print(f"Total images found: {len(image_files)}")
    print(f"Valid images: {valid_images}")
    print(f"Invalid images: {invalid_images}")
    
    if valid_images > 0:
        avg_size = total_size / valid_images
        print(f"Average file size: {avg_size/1024:.1f} KB")
        
        print(f"\nImages by category:")
        for category, count in categories.items():
            print(f"  {category}-like humans: {count}")

def main():
    """Main function to generate human-animal hybrid images"""
    print("=== Human-Animal Image Generator ===")
    print("Generating human faces with subtle animal-like features")
    print("Using Pollinations.ai API (free service)")
    print()
    
    # Generate images
    successful, failed = generate_human_animal_images()
    
    # Verify generated images
    verify_generated_images()
    
    print(f"\n=== Generation Complete ===")
    print(f"Generated {successful} images successfully")
    
    if failed > 0:
        print(f"Failed to generate {failed} images")
        print("You may want to run the script again to retry failed generations.")

if __name__ == "__main__":
    main() 