import json

class JsonReader:
    def __init__(self, path):
        f = open(path)
    
        self.data = json.load(f)

        f.close()
                
    def get_images(self):
        filenames = []
        for img in self.data['images']:
            filenames.append({
                "id": img["id"],
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

    def get_target(self):
        targets = []
        
        for annotation in self.data['annotations']:            
            targets.append({
                    "image_id": annotation["image_id"],
                    "bbox": annotation['bbox'],
                    "category_id": annotation['category_id']        
                    })

        return targets
    