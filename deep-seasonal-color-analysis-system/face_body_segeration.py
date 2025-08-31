import cv2
import numpy as np
from PIL import Image
import os
import argparse

def detect_face_opencv(image_path, scale_factor=1.1, min_neighbors=5):
    """
    Detect face using OpenCV's Haar Cascade (lightweight, CPU-friendly)
    
    Args:
        image_path (str): Path to input image
        scale_factor (float): Parameter specifying how much the image size is reduced at each scale
        min_neighbors (int): Parameter specifying how many neighbors each face rectangle should retain
    
    Returns:
        tuple: (image_array, face_coordinates) where face_coordinates is (x, y, w, h) or None
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to RGB for consistency
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    # Return the largest face if multiple faces are detected
    if len(faces) > 0:
        # Find the largest face by area
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        return image_rgb, largest_face
    else:
        return image_rgb, None

def crop_face(image, face_coords, padding=0.1):
    """
    Crop the face region from the image with optional padding
    
    Args:
        image (numpy.ndarray): Input image array
        face_coords (tuple): Face coordinates (x, y, w, h)
        padding (float): Padding ratio to add around the face (0.1 = 10% padding)
    
    Returns:
        numpy.ndarray: Cropped face image
    """
    x, y, w, h = face_coords
    
    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    # Calculate new coordinates with padding
    x_start = max(0, x - pad_w)
    y_start = max(0, y - pad_h)
    x_end = min(image.shape[1], x + w + pad_w)
    y_end = min(image.shape[0], y + h + pad_h)
    
    # Crop the face region
    face_crop = image[y_start:y_end, x_start:x_end]
    
    return face_crop

def create_body_image(image, face_coords, method='black'):
    """
    Create body image by removing/masking the face region
    
    Args:
        image (numpy.ndarray): Input image array
        face_coords (tuple): Face coordinates (x, y, w, h)
        method (str): Method to handle face area ('black', 'blur', 'white')
    
    Returns:
        numpy.ndarray: Body image with face area processed
    """
    # Create a copy of the original image
    body_image = image.copy()
    
    x, y, w, h = face_coords
    
    if method == 'black':
        # Black out the face area
        body_image[y:y+h, x:x+w] = [0, 0, 0]
    elif method == 'white':
        # White out the face area
        body_image[y:y+h, x:x+w] = [255, 255, 255]
    elif method == 'blur':
        # Blur the face area
        face_region = body_image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        body_image[y:y+h, x:x+w] = blurred_face
    
    return body_image

def save_image(image_array, output_path, quality=95):
    """
    Save image array to file using PIL for better quality control
    
    Args:
        image_array (numpy.ndarray): Image array to save
        output_path (str): Output file path
        quality (int): JPEG quality (1-100)
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Save with specified quality
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        pil_image.save(output_path, 'JPEG', quality=quality)
    else:
        pil_image.save(output_path)
    
    print(f"Saved: {output_path}")

def process_image(input_path, output_dir="output", face_padding=0.1, body_method='black'):
    """
    Main function to process the input image and create face/body separations
    
    Args:
        input_path (str): Path to input image
        output_dir (str): Directory to save output images
        face_padding (float): Padding around face crop (0.1 = 10%)
        body_method (str): Method to handle face area in body image
    
    Returns:
        dict: Dictionary with paths to created files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    try:
        # Detect face in the image
        print("Detecting face...")
        image, face_coords = detect_face_opencv(input_path)
        
        if face_coords is None:
            print("No face detected in the image!")
            return None
        
        x, y, w, h = face_coords
        print(f"Face detected at: x={x}, y={y}, width={w}, height={h}")
        
        # Create face crop
        print("Cropping face...")
        face_crop = crop_face(image, face_coords, padding=face_padding)
        
        # Create body image (with face area processed)
        print(f"Creating body image (method: {body_method})...")
        body_image = create_body_image(image, face_coords, method=body_method)
        
        # Define output paths
        face_path = os.path.join(output_dir, f"{base_name}_face.jpg")
        body_path = os.path.join(output_dir, f"{base_name}_body.jpg")
        original_copy_path = os.path.join(output_dir, f"{base_name}_original.jpg")
        
        # Save images
        print("Saving images...")
        save_image(face_crop, face_path)
        save_image(body_image, body_path)
        save_image(image, original_copy_path)  # Save original copy
        
        result = {
            'face': face_path,
            'body': body_path,
            'original_copy': original_copy_path,
            'face_coordinates': face_coords
        }
        

        # Convert NumPy arrays to PIL Images (in memory)
        face_pil = Image.fromarray(face_crop)
        body_pil = Image.fromarray(body_image)
        original_pil = Image.fromarray(image)

        

        
        print("\nProcessing completed successfully!")
        print(f"Face image: {face_path}")
        print(f"Body image: {body_path}")
        print(f"Original copy: {original_copy_path}")
        
        return {
            'face_image': face_pil,
            'body_image': body_pil,
            'original_image': original_pil,
            'face_coordinates': face_coords
        }
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def main():
    """
    Command line interface for the face/body separation script
    """
    parser = argparse.ArgumentParser(description='Separate face and body from a full-body photo')
    parser.add_argument('input_path', help='Path to input image')
    parser.add_argument('--output_dir', default='output', help='Output directory (default: output)')
    parser.add_argument('--face_padding', type=float, default=0.1, 
                       help='Padding around face crop as ratio (default: 0.1)')
    parser.add_argument('--body_method', choices=['black', 'white', 'blur'], default='black',
                       help='Method to handle face area in body image (default: black)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' not found!")
        return
    
    # Process the image
    result = process_image(
        input_path=args.input_path,
        output_dir=args.output_dir,
        face_padding=args.face_padding,
        body_method=args.body_method
    )
    
    if result is None:
        print("Processing failed!")
    else:
        print("\nAll files created successfully!")

# Example usage as a module
def example_usage():
    """
    Example of how to use this script as a module
    """
    input_image = "your_photo.jpg"  # Replace with your image path
    
    # Basic usage
    result = process_image(input_image)
    
    if result:
        print("Success! Files created:")
        for key, path in result.items():
            if key != 'face_coordinates':
                print(f"  {key}: {path}")
    
    # Advanced usage with custom parameters
    result = process_image(
        input_path=input_image,
        output_dir="my_output",
        face_padding=0.2,  # 20% padding around face
        body_method='blur'  # Blur face instead of blacking out
    )

if __name__ == "__main__":
    main()

    """# Basic usage - default 1.8x head expansion
python script.py photo.jpg

# More aggressive head detection
python script.py photo.jpg --head_expansion 2.0

# Custom settings
python script.py photo.jpg --head_expansion 2.2 --head_padding 0.1 --body_method blur
    """