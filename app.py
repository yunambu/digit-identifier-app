# Dependencies used to build the web app
from flask import Flask, render_template, jsonify, request
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException
from image_helpers import smart_crop, fix_image_rotation
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
        # TODO: Update to your model's filename
        model = load_model('./mnistmodel.h5')
    return model


# Homepage: The form
@app.route('/')
def form():
    return render_template('form.html')


# Process the uploaded image:
#  - Apply transformations to build up the input to the model
#  - Use the model to perform the prediction
@app.route('/process_form', methods=["POST"])
def process_form():
    if not valid_image_request():
        return jsonify({
            "success": False,
            "error": "You must provide a valid `image`."
        })

    # Load the uploaded image into the `image` variable
    image = request.files["image"].read()
    image = Image.open(io.BytesIO(image))

    # Note that when using images uploaded from iphones, sometimes they will be rotated incorrectly.
    # This method fixes rotation if needed.
    image = fix_image_rotation(image)

    # The `prepare_image` method will transform the input image into a 28x28 image.
    image, image_display = prepare_image(image)

    model = load_keras_model()

    # Returns an np array. Convert to list with .tolist()
    prediction_classes = model.predict_classes(image).tolist()

    # Displays probability for each number
    prediction_probabilities_list = model.predict(image).tolist()[0]
    prediction_probabilities = {}

    # Turn this into a dictionary of digit => probability
    # We will use this to display the probabilities on the result page
    prediction_probabilities = {}
    for num, p in enumerate(prediction_probabilities_list, start=0):
        prediction_probabilities[num] = p

    return render_template('results.html', prediction=prediction_classes[0], probabilities=prediction_probabilities, image=image_display)


def valid_image_request():
    return request.method == "POST" and request.files.get("image")


def prepare_image(image):
    """
    Useful Documentation:
    Image Module: https://pillow.readthedocs.io/en/stable/reference/Image.html
    ImageOps Module: https://pillow.readthedocs.io/en/3.0.x/reference/ImageOps.html
    """

    # Resize image first, so that future operations on the image are faster
    image = ImageOps.fit(image, size=(28, 28))

    # if the image mode is not grayscale ("L"), convert it
    # https://pillow.readthedocs.io/en/4.1.x/handbook/concepts.html#modes
    if image.mode != "L":
        image = image.convert("L")

    # Model is trained on inverted images, so we must invert our input
    image = ImageOps.invert(image)

    # autocontrast cleans up photos that are taken in lower light
    image = ImageOps.autocontrast(image, cutoff=2)

    # Convert into input format for model
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # 28x28 image used for display on results page.
    # (if using CNN, this is the same as the model input image)
    image_display = image

    # Convert from 28x28 image to
    # NOTE: The CNN model does not need this step (comment out if using CNN model)
    image = image.reshape(1, 784)

    # return the processed image
    return image, image_display

# Start the server
if __name__ == "__main__":
    print("* Starting Flask server..."
          "please wait until server has fully started")
    # debug=True options allows us to view our changes without restarting the server.
    app.run(host='0.0.0.0', debug=True)
