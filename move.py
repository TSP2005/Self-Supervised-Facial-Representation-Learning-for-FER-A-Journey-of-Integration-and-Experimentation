import os
import random
import shutil

source_root = "./data/affectnet/val"
target_root = "./data/affectnet/train"
categories = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

for category in categories:
    source_dir = os.path.join(source_root, category)
    target_dir = os.path.join(target_root, category)

    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists

    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(files)  # Shuffle files

    num_to_move = len(files) // 2  # Select half
    files_to_move = files[:num_to_move]

    for file in files_to_move:
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))

    print(f"Moved {num_to_move} images from {source_dir} to {target_dir}")
