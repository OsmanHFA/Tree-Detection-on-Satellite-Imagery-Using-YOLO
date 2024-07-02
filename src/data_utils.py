from google.cloud import storage
import os
import rasterio
from matplotlib import pyplot as plt

def download_images(bucket_name, prefix, local_dir):
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