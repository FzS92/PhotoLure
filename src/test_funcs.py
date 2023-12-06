import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from funcs import (  # Make sure to replace 'your_module' with the actual module name
    preprocess,
    replace_pixels_with_mask,
)


def test_preprocess():
    # Create a dummy image for testing
    dummy_image = Image.new("RGB", (100, 100))

    # Test preprocess function
    target_shape = 64
    processed_image = preprocess(dummy_image, target_shape)

    # Check if the output has the correct shape
    assert processed_image.size == (target_shape, target_shape)


def test_replace_pixels_with_mask():
    # Create dummy images and mask for testing
    image1 = Image.new("RGB", (100, 100))
    image2 = Image.new("RGB", (100, 100))
    mask = np.ones((100, 100), dtype=np.uint8)  # All pixels are 1 in the mask

    # Test replace_pixels_with_mask function
    replaced_image = replace_pixels_with_mask(image1, image2, mask)

    # Check if the output has the correct type
    assert isinstance(replaced_image, Image.Image)

    # Check if the output image has the same size as input images
    assert replaced_image.size == image1.size
