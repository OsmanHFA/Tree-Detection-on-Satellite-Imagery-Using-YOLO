from google.cloud import storage
import os
import rasterio
from matplotlib import pyplot as plt

def download_images_and_labels(bucket_name, prefix, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Verify bucket existence
    if not bucket.exists():
        print(f"Bucket {bucket_name} does not exist.")
        return
    
    blobs = bucket.list_blobs(prefix=prefix)
    found_any_blob = False
    
    for blob in blobs:
        found_any_blob = True
        print(f"Found blob: {blob.name}")  # Debugging 
        if any(blob.name.endswith(ext) for ext in ['.jpeg', '.jpg', '.json', '.png', '.csv', '.txt']):
            
            # Create local directory structure if it doesn't exist
            local_path = os.path.join(local_dir, os.path.relpath(blob.name, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")
    
    if not found_any_blob:
        print("No blobs found with the given prefix.")
            
def download_tiff_images(bucket_name, prefix, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith('.tif'):
            destination_path = os.path.join(local_dir, os.path.basename(blob.name))
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            blob.download_to_filename(destination_path)
            print(f"Downloaded {blob.name} to {destination_path}")
            
def display_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:
        image_data = src.read(1)  # Read the first band
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image_data, cmap='gray')
    plt.title('TIFF Image')
    plt.axis('off')
    plt.show()