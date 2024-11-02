import os
import sys
from PIL import Image

    
for filename in os.listdir('Work'):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        file_path = os.path.join('Work', filename)
        
        image = Image.open(file_path).convert('RGB')
        image_resized = image.resize((128, 128))
        
        output_file_path = os.path.join('Temp', os.path.splitext(filename)[0] + '_resized.jpg')
        image_resized.save(output_file_path, format='JPEG')
        
        os.remove(file_path)

        