from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageOps

def smart_crop(image):
    # NOTE: We are not using this method.
    # --- Let's try to smart crop the image by finding the digit, to handle imperfect images
    # ref: https://codereview.stackexchange.com/a/132933

    # This will not work well for images that are already small (i.e. already 28x28)
    # In that case, return the image directly
    if image.size == (28, 28):
        return image

    # Assume that any pixel with value of 75 or less is black (part of background, since inverted)
    mask = img_to_array(image) > 100

    # Find coordinates of non-black pixels
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    y0, x0, _ = coords.min(axis=0)
    y1, x1, _ = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = image.crop((x0, y0, x1, y1))

    # Make the image square, with the size being the largest current size
    new_size = max(cropped.size)
    width_border = int((new_size - cropped.size[0]) / 2)
    height_border = int((new_size - cropped.size[1]) / 2)
    border = (width_border, height_border, width_border, height_border)
    image = ImageOps.expand(cropped, border=border)

    # Convert to final size, minus a border
    image = ImageOps.fit(image, size=(20, 20))

    # add border (so final size is 28x28)
    image = ImageOps.expand(image, border=4)

    return image
