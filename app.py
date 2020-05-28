from __future__ import division, print_function
# coding=utf-8
import sys
import os
import pickle
import numpy as np
import cv2
from definition import ResNet50

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Flask utils
from flask import Flask

# Server Requirement
sys.path.insert(0, os.path.dirname(__file__))

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/crop_doctor.h5'

# Load your trained model
model = ResNet50(input_shape=(256, 256, 3), num_classes=31)
model = model.load_weights(MODEL_PATH)


def model_predict(img_path):
    img = cv2.resize(img_path, tuple((256,256)))

    # Preprocessing the image
    img = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = [img]
    x = np.array(x, dtype=np.float16) / 225.0

    pred = model.predict(x)
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return 'Agrifarm Crop Doctor in Service'


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)

        # Process your result for human          # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return preds.encode()
    return None


# if __name__ == '__main__':
#     app.run(debug=True)
