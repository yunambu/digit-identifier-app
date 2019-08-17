from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageOps, ExifTags
import functools

def fix_image_rotation(im):
    """
    Fixes rotation of image, by checking EXIF data.
    This fixes issues with images uploaded directly from a smart phone camera.
    Source: https://stackoverflow.com/a/30462851/76710
    """

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [                   # Val  0th row  0th col
        [],                                        #  0    (reserved)
        [],                                        #  1   top      left
        [Image.FLIP_LEFT_RIGHT],                   #  2   top      right
        [Image.ROTATE_180],                        #  3   bottom   right
        [Image.FLIP_TOP_BOTTOM],                   #  4   bottom   left
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  #  5   left     top
        [Image.ROTATE_270],                        #  6   right    top
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  #  7   right    bottom
        [Image.ROTATE_90],                         #  8   left     bottom
    ]

    try:
        seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag]]
    except Exception:
        return im
    else:
        return functools.reduce(type(im).transpose, seq, im)


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
