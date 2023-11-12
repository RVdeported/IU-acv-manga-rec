import streamlit as st
import numpy as np
import time
import cv2
from src.annoy_index import AnnoyTree
from src.model import YoloModel
from src.feature_utils import MangaPredictor
import pickle

with open('results/text_vectors.pik', "rb") as f:
    text = pickle.load(f)
with open('results/image_vectors.pik', "rb") as f:
    img = pickle.load(f)
with open('results/manga_names.pik', "rb") as f:
    manga = pickle.load(f)
with open('results/paths.pik', "rb") as f:
    paths = pickle.load(f)

model = YoloModel.load('model.pik')
ann = AnnoyTree(img, text, manga, paths)

IMG1 = './data/demo/Sample1.jpg'
IMG2 = './data/demo/Sample2.jpg'
IMG3 = './data/demo/Sample3.jpg'

def img_launch(path):
    img, text = model.get_image_text(path)
    predictions = ann.infer(img, text)
    predictions = sorted(predictions, key=lambda x: x[2])
    st.session_state.predictions = predictions

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

st.title('Manga Recommendation system')
st.markdown('''Welcome to the prototype of Manga recommendation system. The project is developed as a part of Advanced computer vision course in Innopolis University. 
        \nThe application allows to input a Manga page and then receive a list of suggested Mangas.
        Currently the app allows to choose one of Mangas chosen for demo or put your own Manga page in
        a pucture format (.jpg, .png, etc.)''')

st.header("Choose a manga for demo")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Sample 1**")
    st.image(cv2.imread(IMG1))
with col2:
    st.markdown("**Sample 2**")
    st.image(cv2.imread(IMG2))
with col3:
    st.markdown("**Sample 3**")
    st.image(cv2.imread(IMG3))

col1, col2, col3 = st.columns(3)
with col1:
    st.button("**RUN**", on_click=img_launch, args=[IMG1], key="sample_1")
with col2:
    st.button("**RUN**", on_click=img_launch, args=[IMG2], key="sample_2")
with col3:
    st.button("**RUN**", on_click=img_launch, args=[IMG3], key="sample_3")

st.header("...or pass your own")
input_file = st.file_uploader("Choose file...", type=['png', 'jpeg', 'jpg'])

if input_file is not None:
    extension = input_file.type.split('/')[-1]
    with open("./data/demo/input." + extension, 'wb+') as f:
        f.write(input_file.getvalue())
    img_launch("./data/demo/input." + extension)

if st.session_state.predictions != None:
    predictions = st.session_state.predictions
    for idx, col in enumerate(st.columns(5)):
        with col:
            name, p, score = predictions[idx]
            st.markdown(f"**RESULT {idx+1}.**")
            st.markdown(f"**{name}**")
            st.markdown(f"Infernece score - {score}")
            st.image(cv2.imread(p))


