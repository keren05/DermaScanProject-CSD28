import os
import zipfile
import requests
from tqdm import tqdm


# 1. Download the dataset
def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


# ISIC 2019 dataset URLs
files = {
    "images": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
    "labels": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
}

# Create data directory
os.makedirs("ISIC_2019", exist_ok=True)

# Download files
for name, url in files.items():
    filename = os.path.join("ISIC_2019", url.split('/')[-1])
    download_file(url, filename)

# 2. Unzip the images
print("Extracting images...")
with zipfile.ZipFile(os.path.join("ISIC_2019", "ISIC_2019_Training_Input.zip"), 'r') as zip_ref:
    zip_ref.extractall("ISIC_2019")

print("Download complete! Files are in ISIC_2019/ directory")
