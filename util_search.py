import numpy as np
import pandas as pd
from pathlib import Path
import torch
import clip
from PIL import Image
from IPython.display import Image as disImage
import ipyplot


class CLIPSearch():
    def __init__(self, model_name, model_path='models', device='cpu'):
        # model_name: 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
        self.model_name = model_name
        self.model_path = model_path
        self.device = device #"cuda" if torch.cuda.is_available() else "cpu"
        self._load_model(model_name)


    def load_dataset(self, root_path='data/unsplash-dataset/lite'):
        # Load the features and the corresponding IDs
        self.imgdata_path = Path(root_path) / "images"
        features_path = Path(root_path) / "features" / self.model_name
        self.img_features = np.load(features_path / "features.npy")
        image_ids = pd.read_csv(features_path / "image_ids.csv")
        self.image_ids = list(image_ids['image_id'])


    def _load_model(self, model_name):
        self.model, self.preprocess = clip.load(
            model_name, 
            device=self.device, 
            download_root=self.model_path)


    def _encode_image(self, img_path):
        img = Image.open(img_path)
        img_preprocessed = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Encode the imgs batch to compute the feature vectors and normalize them
            img_features = self.model.encode_image(img_preprocessed)
            img_features /= img_features.norm(dim=-1, keepdim=True)
        return img_features #.cpu().numpy()


    def _encode_text(self, search_query):
        with torch.no_grad():
            # Encode and normalize the description using CLIP
            text_encoded = self.model.encode_text(clip.tokenize(search_query).to(self.device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        return text_encoded


    def _find_best_matches(self, search_features, results_count=3):
        search_features = search_features.cpu().numpy()
        similarities = list((search_features @ self.img_features.T).squeeze(0))
        best_imgs = sorted(zip(similarities, self.image_ids), key=lambda x: x[0], reverse=True)
        return best_imgs[:results_count]


    def _display_imgs(self, best_imgs):
        img_paths = [str(self.imgdata_path / (str(img[1]) + '.jpg')) for img in best_imgs]
        ipyplot.plot_images(img_paths, custom_texts=best_imgs, img_width=300)


    def search_by_text(self, text, results_count=3, display_in_ipython=True):
        text_feature = self._encode_text(text)
        search_results = self._find_best_matches(text_feature, results_count)
        if display_in_ipython:
            self._display_imgs(search_results)


    def search_by_image(self, img_path, results_count=3, display_in_ipython=True):
        img_feature = self._encode_image(img_path)
        search_results = self._find_best_matches(img_feature, results_count)
        if display_in_ipython:
            print('Query image:')
            display(disImage(img_path, width=300))
            print('Results:')
            self._display_imgs(search_results)


    def search_by_text2(self, text1, text2, weight=0.5, results_count=3, display_in_ipython=True):
        text_feature1 = self._encode_text(text1)
        text_feature2 = self._encode_text(text2)
        search_features = weight * text_feature1 + (1 - weight) * text_feature2
        search_features /= search_features.norm(dim=-1, keepdim=True)
        search_results = self._find_best_matches(search_features, results_count)
        if display_in_ipython:
            self._display_imgs(search_results)


    def search_by_text_and_image(self, img_path, text, weight=0.5, results_count=3, display_in_ipython=True):
        img_feature = self._encode_image(img_path)
        text_feature = self._encode_text(text)
        search_features = weight * img_feature + (1 - weight) * text_feature
        search_results = self._find_best_matches(search_features, results_count)
        if display_in_ipython:
            print('Query image:')
            display(disImage(img_path, width=300))
            print('Results:')
            self._display_imgs(search_results)


if __name__ == "__main__":
    search = CLIPSearch('lite')
    search.load_model('ViT-B/32')
    search.search_by_text('five man standing under a tree')
    # search.search_by_image(r'.\unsplash-dataset\lite\photos\__G2yFuW7jQ.jpg')
