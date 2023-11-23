# basic
import torch as t
import numpy as np
import pickle
import os
import re
from pathlib import Path

# Text processing
from easyocr import Reader
from spellchecker import SpellChecker
from transformers import BertTokenizer, BertModel

# Image processing
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

# own libraries
from src.dataset import YoloDataset
from src.model import YoloModel
from src.annoy_index import AnnoyTree

#=================================================#
# TextExtractor class                             #
#=================================================#
class TextExtractor:
    '''Extraction of a text from picture'''

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
    
    #-------------------------------------------------#
    # Inference                                       #
    #-------------------------------------------------#
    def extract(self, page_path: Path) -> list:
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
    

#=================================================#
# BERT converter                                  #
#=================================================#
class BERTconverter:
    '''Class for ruBERT API'''
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
    
    #-------------------------------------------------#
    # Inference                                       #
    #-------------------------------------------------#
    def __call__(self, a_path: Path):
        
        # extraction of words
        words   = self.text_extr.extract(a_path)
        
        # tokenization of words
        b_inp   = ['[CLS]'] + words
        b_inp   = " ".join(b_inp)
        token_t = self.tokenizer.tokenize(b_inp)
        idx_token = self.tokenizer.convert_tokens_to_ids(token_t)
        token_pt = t.tensor([idx_token])
        
        # segmentation. For our case we will have single segment
        idx_seg   = [1] * len(token_t)
        idx_seg_pt = t.tensor([idx_seg])

        # inference of BERT and returning of last hidden dim
        with t.no_grad():
            outputs = self.model(token_pt, idx_seg_pt)
            res = t.sum(outputs.last_hidden_state, dim=1)[0]
        return res
    
#=================================================#
# Manga Predictor class                           #
#=================================================#
class MangaPredictor:
    '''
    Central class of the solution.
    Responsible for combining together the YOLO model, annoyIndex search
    and text features extractor for full pipeline construction.
    '''
    def __init__(self, n_pca=10, ocr_device = 'cpu'):
        self.model = None
        self.text = BERTconverter(a_device_ocr = ocr_device)
        self.pca   = PCA(n_components=n_pca)
        self.annoy = None
        self.train_features = None
        self.train_titles   = None
        self.train_path     = None
    
    
    def extract_title(self, path):
        '''
        extraction of a title from path to a class folder
        '''
        return path.split(os.sep)[-1].split('-')[0]
    
    
    @staticmethod
    def column(data, i):
        '''
        extraction of a 'column' from 2D list
        '''
        return [r[i] for r in data]
    
    
    @staticmethod
    def convert_to_ds(arr_1: list[np.array], 
                      arr_2:list[np.array]
                      ) -> np.array:
        '''
        conversion from the two lists with 1D numpy arr to the 2D numpy 
        '''
        ds = np.zeros((len(arr_1), arr_1[0].shape[0] + arr_2[0].shape[0]))
        for i, _ in enumerate(arr_1):
            ds[i] = np.concatenate([arr_1[i], arr_2[i]], axis=0)
        return ds

    #-------------------------------------------------#
    # Training                                        #
    #-------------------------------------------------#
    def train(self, 
              data_path: Path,              # Path to data
              shape    : list[2] = [8,512], # Target shape for YOLO model
              epochs   : int     = 10,      # num of training epochs
              y_lr0    : float   = 0.01,    # start learning rate
              y_lr1    : float   = 0.01     # end learning rate
              ):
        '''
        method for training of the solution's components.
        Trains YOLO model and fots the PCA algorithm on the total
        feature vectors.
        '''
        # YOLO model training
        self.model = YoloModel(data_path, shape, epochs, y_lr0, y_lr1)
        self.shape = shape
        # PCA fit and transform
        img, text, manga, path = self.get_features(Path(data_path) / "train")
        train_ds = self.convert_to_ds(img, text)
        self.pca.fit(train_ds)
        vec = self.apply_pca_lists(img, text)
        # Annoy index creation
        self.annoy = AnnoyTree(vec, manga, path)
        # recording of data
        self.train_features = vec
        self.train_titles   = manga
        self.train_path     = path
        

    def apply_pca_lists(self, 
                        img: list[np.array], 
                        text: list[np.array]):
        '''PCA transformatio of a list of numpy array'''
        ds = self.convert_to_ds(img, text)
        ds_pca = self.pca.transform(ds)
        return ds_pca

    #-------------------------------------------------#
    # Featrures retrieval                             #
    #-------------------------------------------------#
    def get_features(self, folder):
        '''Getting feature vec, titles and paths from a folder with pictures'''
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

    #-------------------------------------------------#
    # Featrures retrieval from an image               #
    #-------------------------------------------------#
    def get_image_text(self, path):
        '''gets feature vec and title from an image'''
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
    
    #-------------------------------------------------#
    # Get Recommendations                             #
    #-------------------------------------------------#
    def get_top_rec(self, img_path):
        '''returns top-n recommendations in form of Title / Path to img / distance'''
        vec = self.get_image_text(img_path)
        return self.annoy.infer(vec)
    

    def save(self, path):
        '''Saving of the model'''
        self.annoy = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        

    @staticmethod
    def load(path):
        '''Loading of a model'''
        with open(path, "rb") as f:
            mp = pickle.load(f)
        mp.annoy = AnnoyTree(mp.train_features, mp.train_titles, mp.train_path)
        return mp