from ultralytics import YOLO
import torch
import functools
from sklearn.decomposition import PCA
from .dataset import YoloDataset
import os
from torch.utils.data import DataLoader

class YoloModel:
    def __init__(self, path, shape, epochs=1):
        self.path = path
        self.shape = shape
        model = YOLO('yolov8m-cls.pt', task='classification')
        model.train(data=self.path, epochs=epochs, imgsz=self.shape)
        children = list(list(list(list(model.children())[0].children())[0].children()))
        ch_new = children[:-1]
        self.model = torch.nn.Sequential(*ch_new)
        self._train_pca()
        
    def _train_pca(self):
        dataset = YoloDataset(os.path.join(self.path, "train"), self.shape)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        res = list(map(lambda x: self.model(x[0]).flatten(1), train_dataloader))
        vectors = functools.reduce(lambda x, y: torch.cat((x, y)), res)
        res_np = vectors.numpy()
        self.pca = PCA().fit(res_np)
        

    def forward(self, image):
        res = self.model(image).flatten(1).numpy()
        return self.pca.transform(res)
