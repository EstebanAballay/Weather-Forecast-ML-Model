from PIL import Image
import sys

img_path = 'assets/logo.png'
try:
    img = Image.open(img_path)
    print(f"Original size: {img.size}")
    width, height = img.size
    crop_height = int(height * 0.70)
    img_cropped = img.crop((0, 0, width, crop_height))
    img_cropped.save(img_path)
    print(f"Cropped size: {img_cropped.size}")
except Exception as e:
    print(f"Error: {e}")
