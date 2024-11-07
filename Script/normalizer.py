import os
import sys
from PIL import Image, ImageOps

os.makedirs('Temp', exist_ok=True)

for filename in os.listdir(sys.argv[1]):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        file_path = os.path.join(sys.argv[1], filename)
        
        image = Image.open(file_path)
        image = ImageOps.exif_transpose(image).convert('RGB')
        
        final_size = (600, 600)
        
        image.thumbnail(final_size, Image.LANCZOS)
        
        background = Image.new('RGB', final_size, (255, 255, 255))
        offset = ((final_size[0] - image.width) // 2, (final_size[1] - image.height) // 2)
        background.paste(image, offset)
        
        base_output_path = os.path.join('Temp', os.path.splitext(filename)[0])
        background.save(base_output_path + '_resized.jpg', format='JPEG')
        
