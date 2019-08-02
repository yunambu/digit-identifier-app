# NOTE: A lot of this is from this keras blog post:
# https: // blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

from flask import Flask, render_template, jsonify, request
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException
import traceback
import io

app = Flask(__name__)
model = None


@app.route('/')
def home():
    return render_template('home.html')


@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    print("Error!", e)
    print(traceback.format_exc())
    return jsonify(error=str(e)), code


def load_keras_model():
    global model
    # Only load model once
    if not model:
        print("------------------------>>>> loading model...")
        model = load_model('./mnistmodel.h5')

    # Error: Tensor Tensor("dense_4/Softmax:0", shape=(?, 10), dtype=float32) is not an element of this graph.
    # https://github.com/keras-team/keras/issues/6462#issuecomment-319232504
    # model._make_predict_function()
    # this is key : save the graph after loading the model
    # global graph
    # graph = tf.get_default_graph()

    return model


def valid_image_request():
    return request.method == "POST" and request.files.get("image")


def prepare_image(image, target_size):
    # if the image mode is not grayscale ("L"), convert it
    # https://pillow.readthedocs.io/en/4.1.x/handbook/concepts.html#modes
    if image.mode != "L":
        image = image.convert("L")

    #
    image = ImageOps.invert(image)

    # resize the input image and preprocess it
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # We need 1x784 shape as input for this model
    image = image.reshape(1, 784)

    # return the processed image
    return image


def valid_form_request():
    return request.method == "POST" and request.files.get("image")


@app.route('/process_form', methods=["POST"])
def process_form():
    # values = {}

    error = False
    if not valid_form_request():
        error = True

    age = request.form['age']
    bpSystolic = request.form['bpSystolic']
    bpDiastolic = request.form['bpDiastolic']
    weight = request.form['weight']  # kg
    height = request.form['height']  # cm
    # normal, aboveNormal, or wellAboveNormal
    cholesterol = request.form['cholesterol']
    cholesterolDescriptions = {
        "normal": "Normal",
        "aboveNormal": "Above Normal",
        "wellAboveNormal": "Well Above Normal",
    }

    inputValues = {
        "Age": age,
        "Blood Pressure": "%s/%s" % (bpSystolic, bpDiastolic),
        "Weight": "%s kg" % weight,
        "Height": "%s cm" % height,
        "Cholesterol": cholesterolDescriptions[cholesterol]
    }

    return render_template('results.html', prediction="asdf", inputValues=inputValues)


@app.route('/predict', methods=["POST"])
def predict():
    if not valid_image_request():
        return jsonify({
            "success": False,
            "error": "You must provide a valid `image`."
        })

    image = request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image, target_size=(28, 28))

    # Image loading from https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037

    # Decoding and pre-processing base64 image
    # img = image.img_to_array(image.load_img(BytesIO(base64.b64decode(request.form['b64'])),
    #                                         target_size=(224, 224))) / 255.
    # this line is added because of a bug in tf_serving(1.10.0-dev)
    # img = img.astype('float16')
    # img = ?
    # Pre-process the image.

    model = load_keras_model()

    # Returns an np array.. convert to list with .tolist()
    prediction_classes = model.predict_classes(image)

    # Displays probability for each number
    prediction_probabilities_list = model.predict(image).tolist()[0]
    prediction_probabilities = {}

    # Turn this into a dictionary of digit => probability
    prediction_probabilities = {}
    for num, p in enumerate(prediction_probabilities_list, start=0):
        prediction_probabilities[num] = format(p, ".5f")

    response = {
        "success": True,
        "prediction": prediction_classes.tolist(),
        "predictions": prediction_probabilities,
    }

    print(response)

    return jsonify(response)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(host='0.0.0.0', debug=True)
