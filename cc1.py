import os
import random

def delete_half_images(base_path):
    if not os.path.exists(base_path):
        print(f"Path '{base_path}' does not exist.")
        return
    
    folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    for folder in folders:
        images = [os.path.join(folder, img) for img in os.listdir(folder) if os.path.isfile(os.path.join(folder, img))]
        num_to_delete = len(images) // 2
        
        images_to_delete = random.sample(images, num_to_delete)
        
        for img in images_to_delete:
            os.remove(img)
            print(f"Deleted: {img}")
        
        print(f"Deleted {num_to_delete} images from {folder}.")

# Usage
delete_half_images("./data/cfp_fp/cfp-dataset/Protocol/Split/FP")