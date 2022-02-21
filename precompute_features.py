import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip


model_name = "ViT-B/16"   # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
root_path = r"E:\data\renminribao"
imgdata_path = Path(root_path) / "images"
features_path = Path(root_path) / "features" / model_name
features_path.mkdir(parents=True, exist_ok=True)

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Running on {device}.')
model_path = "models"
model, preprocess = clip.load(model_name, device=device, download_root=model_path)

# Function that computes the feature vectors for a batch of images
def compute_clip_features(img_path_batch):
    imgs = [Image.open(img_path) for img_path in img_path_batch]
    imgs_preprocessed = torch.stack([preprocess(img) for img in imgs]).to(device)
    with torch.no_grad():
        img_features = model.encode_image(imgs_preprocessed)
        img_features /= img_features.norm(dim=-1, keepdim=True)
    return img_features.cpu().numpy()

img_files = list(imgdata_path.glob("*.jpg"))
print(f"Images found: {len(img_files)}")

batch_size = 32
batches = math.ceil(len(img_files) / batch_size)

for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    batch_ids_path = features_path / f"{i:010d}.csv"
    batch_features_path = features_path / f"{i:010d}.npy"
    
    if not batch_features_path.exists():
        try:
            img_path_batch = img_files[i*batch_size : (i+1)*batch_size]

            # Compute the features and save to a numpy file
            batch_features = compute_clip_features(img_path_batch)
            np.save(batch_features_path, batch_features)

            # Save the image IDs to a CSV file
            img_ids = [img_path.stem for img_path in img_path_batch]
            img_ids_data = pd.DataFrame(img_ids, columns=['image_id'])
            img_ids_data.to_csv(batch_ids_path, index=False)
        except:
            print(f'Problem with batch {i}')


# Load all numpy files
features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]

# Concatenate the features and store in a merged file
features = np.concatenate(features_list)
np.save(features_path / "features.npy", features)

# Load all the image IDs
img_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))])
img_ids.to_csv(features_path / "image_ids.csv", index=False)