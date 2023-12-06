# tests/test_functions.py
import os
import numpy as np
from PIL import Image
import pytest
import torch
from torchvision.transforms import ToPILImage, ToTensor

from .funcs import preprocess, replace_pixels_with_mask

# Helper function to create a sample image for testing
def create_sample_image(size=(100, 100), mode="RGB"):
    return Image.fromarray(np.random.randint(0, 255, size + (len(mode),), dtype=np.uint8), mode=mode)

# Test preprocess function
def test_preprocess():
    input_image = create_sample_image()
    target_shape = 64
    processed_image = preprocess(input_image, target_shape)
    
    # Assert that the processed image has the correct shape
    assert processed_image.size == (target_shape, target_shape)

# Test replace_pixels_with_mask function
def test_replace_pixels_with_mask():
    # Create sample images and mask
    image1 = create_sample_image()
    image2 = create_sample_image()
    mask = np.random.randint(0, 2, image1.size, dtype=np.uint8)

    # Perform pixel replacement
    replaced_image = replace_pixels_with_mask(image1, image2, mask)

    # Assert that the replaced image has the same size as the input images
    assert replaced_image.size == image1.size

    # Assert that the replaced image has the expected values based on the mask
    for i in range(image1.size[0]):
        for j in range(image1.size[1]):
            if mask[i, j] == 0:
                assert replaced_image.getpixel((i, j)) == image2.getpixel((i, j))
            else:
                assert replaced_image.getpixel((i, j)) == image1.getpixel((i, j))


