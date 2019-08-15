# Dependencies used to build the web app
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


def load_keras_model():
    """
    This method loads our model, so that we can use it to perform predictions.
    """
    global model
    # Only load model once
    if not model:
        print("------------------------>>>> loading model...")
        model = load_model('./mnistmodel.h5')
    return model


# Homepage: The form
@app.route('/')
def form():
    return render_template('form.html')


@app.route('/process_form', methods=["POST"])
def process_form():
    if not valid_image_request():
        # TODO: do not jsonify.
        return jsonify({
            "success": False,
            "error": "You must provide a valid `image`."
        })

    image = request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image, target_size=(28, 28))

    model = load_keras_model()

    # Returns an np array.. convert to list with .tolist()
    prediction_classes = model.predict_classes(image).tolist()

    # Displays probability for each number
    prediction_probabilities_list = model.predict(image).tolist()[0]
    prediction_probabilities = {}

    # Turn this into a dictionary of digit => probability
    prediction_probabilities = {}
    for num, p in enumerate(prediction_probabilities_list, start=0):
        prediction_probabilities[num] = p

    # TODO: Return image that we're passing to model (28x28) but as an array, so we can build a representation on the frontend (with large pixels)

    return render_template('results.html', prediction=prediction_classes[0], probabilities=prediction_probabilities)


def valid_image_request():
    return request.method == "POST" and request.files.get("image")


def prepare_image(image, target_size):
    # if the image mode is not grayscale ("L"), convert it
    # https://pillow.readthedocs.io/en/4.1.x/handbook/concepts.html#modes
    if image.mode != "L":
        image = image.convert("L")

    # Model is trained on inverted images, so we must invert our input
    image = ImageOps.invert(image)

    # resize the input image and preprocess it
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # NOTE: CNN model does not need this
    # We need 1x784 shape as input for this model
    # image = image.reshape(1, 784)

    # return the processed image
    return image


# Start the server
if __name__ == "__main__":
    print("* Starting Flask server..."
          "please wait until server has fully started")
    # debug=True options allows us to view our changes without restarting the server.
    app.run(host='0.0.0.0', debug=True)
