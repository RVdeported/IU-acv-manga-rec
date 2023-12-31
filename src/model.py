from ultralytics import YOLO
import torch

#=================================================#
# YOLO model API                                  #
#=================================================#
class YoloModel:
    def __init__(self, path, shape, epochs=1, lr0=0.01, lr1=0.01, model_name="yolov8s-cls"):
        self.path = path
        self.shape = shape
        model = YOLO(model_name, task='classification')
        model.train(data=self.path, epochs=epochs, imgsz=self.shape, lr0=lr0, lrf=lr1, dropout=0.2, pretrained=False)
        children = list(list(list(list(model.children())[0].children())[0].children()))
        ch_new = children[:-1]
        self.model = torch.nn.Sequential(*ch_new)
        
    # inference
    def forward(self, image):
        res = self.model(image).flatten(1).numpy()
        return res
    
