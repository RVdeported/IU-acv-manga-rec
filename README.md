# IU-acv-manga-rec

### Project description:
Extensive use of static methods and Artificial Intelligence have opened the door to various recommendation systems for nearly any product. Most successful of techniques being a combination of Collaborative and Content-based techniques which allow us to search a product based on history of users' choices. However, such methods ignore the special features of a particular product, every item is a "black box" for a search algorithm. This problem drastically escalates in case of Manga search - a search among highly diversified content where hidden features of a product are essential for proper recommendations.

Our solution will focus of feature extraction from a manga magazines and search of manga with similar distinctive features. We expect that a search algorithm which utilize a user's favorite mangas' features will yield more relevant results than mentioned recommendation techniques. Our solution will transform a Russian-translated manga pictures into a combination of features based on both images and text collecting a "bag of objects". Less the distance between those features - more similar those manga should be. We utilized YOLO latent feature vector combined with BERT text vector averaged from each input Manga page.

### Launch a demo
Demo is composed using `Streamlit` utility. In order to correctly launch a demo: 
1. install packages stated in `requirements.txt`
2. download and unpack source images (required strictly for demo): https://drive.google.com/file/d/1XSYDgbXujfRGVCUcBelDd6bzuDsxkkMy/view?usp=drive_link
3. downliad and unpack trained model: https://drive.google.com/file/d/1jDKyWLLlUH1IKD4-h_UQjR_t5sUM8-1J/view?usp=drive_link
4. execute `launch.sh` script
You will need at least 2GB GPU memory to execute the demo. Currently only NVIDIA drivers are supported

### Train your own model
In order to fine-tune your own recommendation system, you will need to make the following steps:
1. Download and insert sample Mangas you would like to train on into `prepared_data/[SPLIT]` where `SPLIT` is `train`, `test` and `val`
2. Create `MangaPredictor` instance defined in `src/feature_utils.py` and use method `train` with correspnding arguments as described in `src/feature_utils.py`
3. Infer an image using `get_top_rec` method of `MangaPredictor` instance
