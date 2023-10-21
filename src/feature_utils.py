import torch as t
import numpy as np

import os
import re
from pathlib import Path
from easyocr import Reader
from spellchecker import SpellChecker

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
        print(text)

        text = [n for n in text if n[2] > self.conf]
        print(text)
        # detection = list(filter(lambda det: det[2] > self.confidence, detection))

        l_words = re.findall(
            r'\w+', " ".join([n[1].lower() for n in text])
        )
        print(text)
        # words = findall(r'\w+', " ".join(list(map(lambda det: det[1].lower(), detection))))

        misspelled = self.checker.unknown(l_words)
        print(l_words)
        for i, _ in enumerate(l_words):
            if l_words[i] in misspelled:
                l_words[i] = self.checker.correction(l_words[i])

        return [n for n in l_words if n is not None]
    

