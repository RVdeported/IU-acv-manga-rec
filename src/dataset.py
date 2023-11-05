from torch.utils.data import Dataset
import os
from random import shuffle
from PIL import Image
from torchvision import transforms

class YoloDataset(Dataset):
    def __init__(self, path: str, shape: list[int]):
        classes = os.listdir(path)
        res = []
        CLASSES = ["дзёсэй", "сёдзё", "сёнэн", "сэйнэн"]
        for genre in CLASSES:
            images = os.listdir(os.path.join(path, genre))
            res.extend(list(map(lambda x: os.path.join(path, genre, x), images)))
        shuffle(res)
        self.items = res
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.comp = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor(),
            normalize
        ])
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path = self.items[idx]
        im = Image.open(path).convert("RGB")
        return (self.comp(im), path)
