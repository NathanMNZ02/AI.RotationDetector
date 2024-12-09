import json

class CocoAnnotationsReader:
    def __init__(self, path):
        f = open(path)
    
        self.data = json.load(f)

        f.close()
                
    def get_images(self):
        filenames = []
        for img in self.data['images']:
            filenames.append({
                "id": img["id"],
                "width": img["width"],
                "height": img["height"],
                "file_name": img["file_name"]
                })
            
        return filenames
    
    def get_categories(self):
        categories = []
        for cat in self.data["categories"]:
            categories.append({
                "id": cat["id"],
                "name": cat["name"]
            })
            
        return categories

    def get_targets(self):
        targets = []
        for annotation in self.data['annotations']:
            target = {
                "image_id": annotation["image_id"],
                "bbox": annotation['bbox'],
                "category_id": annotation['category_id']        
            }
            
            # Aggiunge 'segmentation' solo se esiste (es. Instance Segmentation)
            if 'segmentation' in annotation:
                target["segmentation"] = annotation['segmentation']
            
            targets.append(target)
        return targets        
    