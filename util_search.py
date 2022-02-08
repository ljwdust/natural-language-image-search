import numpy as np
import pandas as pd
from pathlib import Path
import torch
import clip
from PIL import Image
from IPython.display import Image as disImage
import ipyplot


class CLIPSearch():
    def __init__(self, dataset_version, model_path='models', device='cpu'):
        # dataset_version: "lite" or "full"
        self.dataset_version = dataset_version
        self.model_path = model_path
        self.device = device #"cuda" if torch.cuda.is_available() else "cpu"
        self.photos_path = Path("unsplash-dataset") / dataset_version / "photos"

    def load_model(self, model_name):
        # model_name: 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
        self.features_path = Path("unsplash-dataset") / self.dataset_version / "features" / model_name
        # Load the features and the corresponding IDs
        self.photo_features = np.load(self.features_path / "features.npy")
        photo_ids = pd.read_csv(self.features_path / "photo_ids.csv")
        self.photo_ids = list(photo_ids['photo_id'])

        # Load the open CLIP model
        self.model, self.preprocess = clip.load(
            model_name, 
            device=self.device, 
            download_root=self.model_path)


    def _encode_photo(self, photo_path):
        photo = Image.open(photo_path)
        photo_preprocessed = self.preprocess(photo).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Encode the photos batch to compute the feature vectors and normalize them
            photos_features = self.model.encode_image(photo_preprocessed)
            photos_features /= photos_features.norm(dim=-1, keepdim=True)
        return photos_features #.cpu().numpy()


    def _encode_text(self, search_query):
        with torch.no_grad():
            # Encode and normalize the description using CLIP
            text_encoded = self.model.encode_text(clip.tokenize(search_query).to(self.device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        return text_encoded


    def _find_best_matches(self, search_features, results_count=3):
        search_features = search_features.cpu().numpy()
        similarities = list((search_features @ self.photo_features.T).squeeze(0))
        best_photos = sorted(zip(similarities, self.photo_ids), key=lambda x: x[0], reverse=True)
        return best_photos[:results_count]


    def _display_photos(self, best_photos):
        photo_paths = [str(self.photos_path / (photo[1] + '.jpg')) for photo in best_photos]
        ipyplot.plot_images(photo_paths, custom_texts=best_photos, img_width=300)


    def search_by_text(self, text, results_count=3, display_in_ipython=True):
        text_feature = self._encode_text(text)
        search_results = self._find_best_matches(text_feature, results_count)
        if display_in_ipython:
            self._display_photos(search_results)


    def search_by_photo(self, photo_path, results_count=3, display_in_ipython=True):
        photo_feature = self._encode_photo(photo_path)
        search_results = self._find_best_matches(photo_feature, results_count)
        if display_in_ipython:
            print('Query image:')
            display(disImage(photo_path, width=300))
            print('Results:')
            self._display_photos(search_results)


    def search_by_text2(self, text1, text2, weight=0.5, results_count=3, display_in_ipython=True):
        text_feature1 = self._encode_text(text1)
        text_feature2 = self._encode_text(text2)
        search_features = weight * text_feature1 + (1 - weight) * text_feature2
        search_features /= search_features.norm(dim=-1, keepdim=True)
        search_results = self._find_best_matches(search_features, results_count)
        if display_in_ipython:
            self._display_photos(search_results)


    def search_by_text_and_photo(self, photo_path, text, weight=0.5, results_count=3, display_in_ipython=True):
        photo_feature = self._encode_photo(photo_path)
        text_feature = self._encode_text(text)
        search_features = weight * photo_feature + (1 - weight) * text_feature
        search_results = self._find_best_matches(search_features, results_count)
        if display_in_ipython:
            print('Query image:')
            display(disImage(photo_path, width=300))
            print('Results:')
            self._display_photos(search_results)


if __name__ == "__main__":
    search = CLIPSearch('lite')
    search.load_model('ViT-B/32')
    search.search_by_text('five man standing under a tree')
    # search.search_by_photo(r'.\unsplash-dataset\lite\photos\__G2yFuW7jQ.jpg')
