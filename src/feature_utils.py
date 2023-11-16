import torch as t
import numpy as np

import os
import re
from pathlib import Path
from easyocr import Reader
from spellchecker import SpellChecker
from transformers import BertTokenizer, BertModel
from src.dataset import YoloDataset
from src.model import YoloModel
from src.annoy_index import AnnoyTree

from PIL import Image
from torchvision import transforms
import pickle
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA


class TextExtractor:
    '''Text extractor from a picture'''

    def __init__(self, 
                 a_lang: str   = "ru", 
                 a_max_d: int  = 1, 
                 a_conf: float = 0.1, 
                 a_device: str = "cpu"
                 ):
        
        assert a_device in ["cpu", "cuda", "mps"]
        assert a_conf > 0.01 and a_conf <= 1.0
        assert a_max_d > 0

        self.conf    = a_conf # level of confidence
        self.device  = a_device
        self.reader  = Reader([a_lang], gpu=a_device != "cpu")
        self.checker = SpellChecker(language=a_lang, distance=a_max_d)
    
    def extract(self, page_path: Path):
        '''Return list of texts for each page in a folder'''

        assert os.path.exists(page_path)
        
        text = self.reader.readtext(page_path)
        text = [n for n in text if n[2] > self.conf]
        l_words = re.findall(
            r'\w+', " ".join([n[1].lower() for n in text])
        )

        misspelled = self.checker.unknown(l_words)
        for i, _ in enumerate(l_words):
            if l_words[i] in misspelled:
                l_words[i] = self.checker.correction(l_words[i])

        return [n for n in l_words if n is not None]
    

class BERTconverter:
    def __init__(self,                
                 a_lang: str   = "ru", 
                 a_max_d: int  = 1, 
                 a_conf: float = 0.1, 
                 a_device_ocr: str = "cpu",                 
                 ):
        self.text_extr = TextExtractor(a_lang=a_lang, 
                                       a_max_d=a_max_d,
                                       a_conf=a_conf,
                                       a_device=a_device_ocr)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model     = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True)
    
    def __call__(self, a_path):
        words   = self.text_extr.extract(a_path)
        b_inp   = ['[CLS]'] + words
        b_inp   = " ".join(b_inp)
        token_t = self.tokenizer.tokenize(b_inp)

        idx_token = self.tokenizer.convert_tokens_to_ids(token_t)

        idx_seg   = [1] * len(token_t)
        token_pt = t.tensor([idx_token])
        idx_seg_pt = t.tensor([idx_seg])

        with t.no_grad():
            outputs = self.model(token_pt, idx_seg_pt)
            res = t.sum(outputs.last_hidden_state, dim=1)[0]
        return res
    

class MangaPredictor:
    def __init__(self, n_pca=10, ocr_device = 'cpu'):
        self.model = None
        self.text = BERTconverter(a_device_ocr = ocr_device)
        self.pca   = PCA(n_components=n_pca)
        self.annoy = None
        self.train_features = None
        self.train_titles   = None
        self.train_path     = None
        
    def extract_title(self, path):
        return path.split(os.sep)[-1].split('-')[0]
        
        # self.yolo(image), self.text(path[0]), 
    @staticmethod
    def column(data, i):
        return [r[i] for r in data]
    
    @staticmethod
    def convert_to_ds(arr_1, arr_2):
        ds = np.zeros((len(arr_1), arr_1[0].shape[0] + arr_2[0].shape[0]))
        for i, _ in enumerate(arr_1):
            ds[i] = np.concatenate([arr_1[i], arr_2[i]], axis=0)
        return ds

    def train(self, data_path, shape, epochs, y_lr0=0.01, y_lr1=0.01):
        self.model = YoloModel(data_path, shape, epochs, y_lr0, y_lr1)
        self.shape = shape
        img, text, manga, path = self.get_features(Path(data_path) / "train")
        train_ds = self.convert_to_ds(img, text)
        self.pca.fit(train_ds)
        vec = self.apply_pca_lists(img, text)
        self.train_features = vec
        self.train_titles   = manga
        self.train_path     = path
        self.annoy = AnnoyTree(vec, manga, path)

    def __call__(self, data_path):
        img, text, manga, path = self.get_features(Path(data_path))
        return self.apply_pca(img, text), manga, path

    def apply_pca_lists(self, img, text):
        ds = self.convert_to_ds(img, text)
        ds_pca = self.pca.transform(ds)
        return ds_pca

    def get_features(self, folder):
        dataset = YoloDataset(folder, self.shape)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        transformed = []
        for img, path in loader:
            ph = path[0]
            try:
                transformed.append([self.model.forward(img).flatten(), self.text(ph).cpu(), self.extract_title(ph), ph])
            except Exception as e:
                print(e, ph)
        # Returns - (image_features, text_features, manga_title, path)
        return self.column(transformed, 0), self.column(transformed, 1), self.column(transformed, 2), self.column(transformed, 3)

    def get_image_text(self, path):
        img = Image.open(path).convert("RGB")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        comp = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
            normalize
        ])
        img = comp(img).unsqueeze(0)
        
        return self.apply_pca_lists([self.model.forward(img).flatten()], [self.text(path)])
    
    def get_top_rec(self, img_path):
        vec = self.get_image_text(img_path)
        return self.annoy.infer(vec)
    
    def save(self, path):
        self.annoy = None
        with open(path, "wb+") as f:
            pickle.dump(self, f)
        

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            mp = pickle.load(f)
        mp.annoy = AnnoyTree(mp.train_features, mp.train_titles, mp.train_path)
        return mp