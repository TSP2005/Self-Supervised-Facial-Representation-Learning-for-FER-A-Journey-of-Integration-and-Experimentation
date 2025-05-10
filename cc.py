import kagglehub

# Download latest version into the specified directory
path = kagglehub.dataset_download("shuvoalok/raf-db-dataset", path="./data/rafdb")

print("Path to dataset files:", path)
