import torch as t
import numpy as np

import os
import re
from pathlib import Path
from easyocr import Reader
from spellchecker import SpellChecker
from transformers import BertTokenizer, BertModel


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

        self.conf = a_conf # level of confidence
        self.device = a_device
        self.reader = Reader([a_lang], gpu=a_device != "cpu")
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
        words = self.text_extr.extract(a_path)
        b_inp = ['[CLS]'] + words
        b_inp = " ".join(b_inp)
        token_t = self.tokenizer.tokenize(b_inp)

        idx_token = self.tokenizer.convert_tokens_to_ids(token_t)

        idx_seg   = [1] * len(token_t)
        token_pt = t.tensor([idx_token])
        idx_seg_pt = t.tensor([idx_seg])

        with t.no_grad():
            outputs = self.model(token_pt, idx_seg_pt)
            res = t.sum(outputs.last_hidden_state, dim=1)[0]
        return res