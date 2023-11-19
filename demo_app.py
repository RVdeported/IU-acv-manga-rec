import streamlit as st
import cv2
from src.feature_utils import MangaPredictor

#---------------------------------------------# 
# constants definition                        #
#---------------------------------------------# 
IMG1 = './samples/Sample1.jpg'
IMG2 = './samples/Sample2.png'
IMG3 = './samples/Sample3.jpg'
# loading of an object with everything
MODEL = MangaPredictor.load('./results/mp.pckl')


#---------------------------------------------# 
# launch of a model with a given path         #
#---------------------------------------------# 
def img_launch(path):
    predictions = MODEL.get_top_rec(path)
    st.session_state.predictions = predictions

# Service checks 'predictions' and if something arise, displays it.
# For now we initialize it as None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

#=============================================#
# Main page rendering                         #
#=============================================#
st.title('Manga Recommendation system')
st.markdown('''Welcome to the prototype of Manga recommendation system. The project is developed as a part of Advanced computer vision course in Innopolis University. 
        \nThe application allows to input a Manga page and then receive a list of suggested Mangas.
        Currently the app allows to choose one of Mangas chosen for demo or put your own Manga page in
        a pucture format (.jpg, .png, etc.)''')

st.header("Choose a manga for demo")


#---------------------------------------------# 
# Examples render                             #
#---------------------------------------------# 
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


#---------------------------------------------# 
# Examples render                             #
#---------------------------------------------# 
st.header("...or pass your own")
input_file = st.file_uploader("Choose file...", type=['png', 'jpeg', 'jpg'])

if input_file is not None:
    extension = input_file.type.split('/')[-1]
    with open("./input." + extension, 'wb') as f:
        f.write(input_file.getvalue())
    img_launch("./input." + extension)

#---------------------------------------------# 
# Render of model output if one was recorded  #
#---------------------------------------------# 
if st.session_state.predictions != None:
    predictions = st.session_state.predictions
    for idx, col in enumerate(st.columns(5)):
        with col:
            name, p, score = predictions[idx]
            st.markdown("**RESULT {}.**".format(idx+1))
            st.markdown("**{}**".format(name))
            st.markdown("Infernece score - {:.2f}".format(score))
            st.image(cv2.imread(p))


