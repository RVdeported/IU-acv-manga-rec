from pathlib import Path
import os
import shutil
import random

from glob import glob
def split_ds(
        a_path_pr:     Path, 
        a_path_mo:     Path, 
        a_train_split: float=0.6,
        a_val_split:   float=0.2,
        a_shuffle:     bool =False
        ):
    assert a_train_split + a_val_split <= 1.0
    assert os.path.exists(a_path_pr)

    if os.path.exists(a_path_mo):
        shutil.rmtree(a_path_mo)
    os.mkdir(a_path_mo)
    os.mkdir(a_path_mo / "train")
    os.mkdir(a_path_mo / "test")
    os.mkdir(a_path_mo / "val")

    classes = glob("./*", root_dir=a_path_pr)

    for cls in classes:
        items = glob("./*", root_dir=a_path_pr / cls)
        if a_shuffle:
            random.shuffle(items)
        train = int(len(items) * a_train_split)
        val   = int(len(items) * a_val_split)
        test  = len(items) - train - val

        for i in range(0, train):
            dest = a_path_mo / "train" / cls 
            Path(dest).mkdir( parents=True, exist_ok=True )
            shutil.copy(a_path_pr / cls / items[i] / "0.jpg", dest / (items[i]+".jpg"))
        for i in range(train, train + val):
            dest = a_path_mo / "val" / cls 
            Path(dest).mkdir( parents=True, exist_ok=True )
            shutil.copy(a_path_pr / cls / items[i] / "0.jpg", dest / (items[i]+".jpg"))
        for i in range(train + val, train + val + test):
            dest = a_path_mo / "test" / cls
            Path(dest).mkdir( parents=True, exist_ok=True )
            shutil.copy(a_path_pr / cls / items[i] / "0.jpg", dest / (items[i]+".jpg"))

    print(classes)








