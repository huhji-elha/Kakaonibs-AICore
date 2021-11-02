from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import os.path as _path
import numpy as np
import cv2 as _cv2
from PIL import ImageFont
import numpy as _np
from hashlib import md5 as _md5

_LOC = _path.realpath(_path.join(_os.getcwd(), _path.dirname(__file__)))

# https://clrs.cc/
_COLOR_NAME_TO_RGB = dict(
    navy=((0, 38, 63), (255, 255, 255)),
    blue=((83, 83, 198), (255, 225, 255)),
    aqua=((102, 102, 255), (255, 255, 255)),
    teal=((15, 205, 202), (255, 255, 255)),
    olive=((51, 153, 102), (255, 255, 255)),
    green=((0, 100, 0), (255, 255, 255)),
    lime=((1, 255, 127), (255, 255, 255)),
    yellow=((218, 165, 32), (255, 255, 255)),
    orange=((230, 92, 0), (255, 255, 255)),
    red=((255, 47, 65), (255, 255, 255)),
    maroon=((135, 13, 75), (255, 255, 255)),
    fuchsia=((51, 102, 153), (255, 255, 255)),
    purple=((204, 0, 255), (255, 255, 255)),
    black=((24, 24, 24), (255, 255, 255)),
    gray=((102, 102, 153), (255, 255, 255)),
    red2=((255, 77, 77), (255, 255, 255)),
    palegreen=((102, 153, 153), (255, 255, 255)),
    purple2=((230, 0, 172), (255, 255, 255)),
    purple3=((147, 112, 219), (255, 255, 255)),
    orchid=((218, 112, 214), (255, 255, 255)),
    slateblue=((106, 90, 205), (255, 255, 255)),
    orchid2=((153, 50, 204), (255, 255, 255)),
    violet=((199, 21, 133), (255, 255, 255)),
    indigo=((75, 0, 130), (255, 255, 255)),
    violet2=((138, 43, 226), (255, 255, 255)),
)

_COLOR_NAMES = list(_COLOR_NAME_TO_RGB)

_DEFAULT_COLOR_NAME = "green"

_FONT_PATH = _os.path.join(_LOC, "NanumGothicBold.ttf")
_FONT_HEIGHT = 15
_FONT = ImageFont.truetype(_FONT_PATH, _FONT_HEIGHT)


def _rgb_to_bgr(color):
    return list(reversed(color))


def _color_image(image, font_color, background_color):
    return background_color + (font_color - background_color) * image / 255


def _get_label_image(text, font_color_tuple_bgr, background_color_tuple_bgr):
    text_image = _FONT.getmask(text)
    shape = list(reversed(text_image.size))
    bw_image = np.array(text_image).reshape(shape)

    image = [
        _color_image(bw_image, font_color, background_color)[None, ...]
        for font_color, background_color in zip(font_color_tuple_bgr, background_color_tuple_bgr)
    ]

    return np.concatenate(image).transpose(1, 2, 0)


def add(image, left, top, right, bottom, label=None, color=None):
    if type(image) is not _np.ndarray:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    except ValueError:
        raise TypeError("'left', 'top', 'right' & 'bottom' must be a number")

    if label and type(label) is not str:
        raise TypeError("'label' must be a str")

    if label and not color:
        hex_digest = _md5(label.encode()).hexdigest()
        color_index = int(hex_digest, 24) % len(_COLOR_NAME_TO_RGB)
        color = _COLOR_NAMES[color_index]

    if not color:
        color = _DEFAULT_COLOR_NAME

    if type(color) is not str:
        raise TypeError("'color' must be a str")

    if color not in _COLOR_NAME_TO_RGB:
        msg = "'color' must be one of " + ", ".join(_COLOR_NAME_TO_RGB)
        raise ValueError(msg)

    colors = [_rgb_to_bgr(item) for item in _COLOR_NAME_TO_RGB[color]]
    color, color_text = colors

    _cv2.rectangle(image, (left, top), (right, bottom), color, 1)

    if label:
        _, image_width, _ = image.shape

        label_image = _get_label_image(label, color_text, color)
        label_height, label_width, _ = label_image.shape

        rectangle_height, rectangle_width = 1 + label_height, 1 + label_width

        rectangle_bottom = top
        rectangle_left = max(0, min(left - 1, image_width - rectangle_width))

        rectangle_top = rectangle_bottom - rectangle_height
        rectangle_right = rectangle_left + rectangle_width

        label_top = rectangle_top + 1

        if rectangle_top < 0:
            rectangle_top = top
            rectangle_bottom = rectangle_top + label_height + 1

            label_top = rectangle_top

        label_left = rectangle_left + 1
        label_bottom = label_top + label_height
        label_right = label_left + label_width

        rec_left_top = (rectangle_left, rectangle_top)
        rec_right_bottom = (rectangle_right, rectangle_bottom)

        _cv2.rectangle(image, rec_left_top, rec_right_bottom, color, -1)

        image[label_top:label_bottom, label_left:label_right, :] = label_image
