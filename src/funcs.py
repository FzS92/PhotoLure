import os

import numpy as np
import PIL
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


# Define the function to preprocess the input image
def preprocess(input_image, target_shape):
    width, height = input_image.size
    max_dimension = max(height, width)
    scale = target_shape / max_dimension
    new_width = int(width * scale)
    new_height = int(height * scale)
    if new_width % 2 != 0:  # If the number is odd
        new_width += 1  # Add 1 to make it even
    if new_height % 2 != 0:  # If the number is odd
        new_height += 1  # Add 1 to make it even

    input_image = input_image.resize((new_width, new_height))
    # input_image.save('check2.png')
    # input_image = transform_to_tensor(input_image).permute(1, 2, 0)
    # print(input_tensor.shape)
    # print(input_tensor.shape)
    # return input_tensor.unsqueeze(0)
    return input_image


# Fix faces
def replace_pixels_with_mask(image1, image2, mask):
    # Convert images to numpy arrays
    img1_array = np.array(image1)
    img2_array = np.array(image2)

    # Ensure the mask has the same shape as the image arrays
    if img1_array.shape != mask.shape:
        mask = np.expand_dims(
            mask, axis=-1
        )  # Add an extra dimension to match the image arrays

    # Replace pixels based on the mask
    replaced_array = np.where(mask == 0, img2_array, img1_array)

    # Create a PIL image from the replaced array
    replaced_image = Image.fromarray(replaced_array.astype(np.uint8))

    return replaced_image


images_examples = [
    os.path.join(os.path.abspath(""), "src/images/celeb2.jpg"),
    os.path.join(os.path.abspath(""), "src/images/selfie2.jpg"),
    os.path.join(os.path.abspath(""), "src/images/celeb1.jpg"),
    os.path.join(os.path.abspath(""), "src/images/selfie4.jpg"),
]

prompt_examples = [
    (
        "Mayan city pramid sunset ivy foliage abandoned luminiscense scultures dark sky"
        " forest stars concept landscape environment depth water waterfall river,"
        " nature, real, high quality, 4k"
    ),
    (
        "A pool full of water and there is table in the background, fancy, Real,"
        " detailed, 4k"
    ),
    (
        "A table, and in the background, scary lightning black and white, Real, nature,"
        " ultra detailed, 8k"
    ),
    (
        "A luxury huge private yacht, sailing in the bahamas with palm trees in the"
        " background and hardwood deck on the yacht, cinematic, nature,"
        " hyperrealistic, 8 k"
    ),
]

examples = [
    [
        images_examples[0],
        None,
        str(prompt_examples[0]),
        "Input Prompt",
        0.66,
        "SAM (Segment Anything)",
        "Stable Diffusion v2",
        0.0,
        7.5,
    ],
    [
        images_examples[1],
        None,
        str(prompt_examples[1]),
        "Chat GPT",
        0.66,
        "SAM (Segment Anything)",
        "Stable Diffusion v2",
        0.0,
        7.5,
    ],
    [
        images_examples[2],
        None,
        str(prompt_examples[2]),
        "Input Prompt",
        0.66,
        "SAM (Segment Anything)",
        "Stable Diffusion v2",
        0.0,
        7.5,
    ],
    [
        images_examples[3],
        None,
        str(prompt_examples[3]),
        "Chat GPT",
        0.66,
        "SAM (Segment Anything)",
        "Stable Diffusion v1",
        0.0,
        7.5,
    ],
]
