import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
import os
import urllib.request

# Sample images for calibration (diverse set)
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/1200px-Kittyply_edit1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/1200px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Siberian_Tiger_by_Malene_Thyssen.jpg/1200px-Siberian_Tiger_by_Malene_Thyssen.jpg"
]

def download_calibration_images(cache_dir="calibration_images"):
    os.makedirs(cache_dir, exist_ok=True)
    image_paths = []
    
    for i, url in enumerate(IMAGE_URLS):
        filename = f"calib_{i}.jpg"
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading calibration image {i+1}/{len(IMAGE_URLS)}...")
            try:
                # Add User-Agent header
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                continue
        image_paths.append(path)
    return image_paths

class CalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

def get_calibration_loader(batch_size=1):
    paths = download_calibration_images()
    dataset = CalibrationDataset(paths)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    loader = get_calibration_loader()
    print(f"Prepared {len(loader)} calibration samples.")
