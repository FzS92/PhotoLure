import os
import gradio as gr
import torch
from torchvision.transforms import ToTensor, ToPILImage
import PIL
from PIL import Image

import numpy as np
from pathlib import Path
from segment_anything import segment_SAM
from deeplab import segment_torch
from stable_diffusion import stable_diffusion
from Dalle import dalle

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
    os.path.join(os.path.abspath(""), "images/celeb2.jpg"),
    os.path.join(os.path.abspath(""), "images/selfie2.jpg"),
    os.path.join(os.path.abspath(""), "images/celeb1.jpg"),
    os.path.join(os.path.abspath(""), "images/selfie4.jpg"),
]

prompt_examples = [
        "Mayan city pramid sunset ivy foliage abandoned luminiscense scultures dark sky forest stars concept landscape environment depth water waterfall river, nature, real, high quality, 4k",
        "A pool full of water and there is table in the background, fancy, Real, detailed, 4k",
        "A table, and in the background, scary lightning black and white, Real, nature, ultra detailed, 8k",
        "A luxury huge private yacht, sailing in the bahamas with palm trees in the background and hardwood deck on the yacht, cinematic, nature, hyperrealistic, 8 k",
]

examples = [
    [
        images_examples[0],
        None,
        str(prompt_examples[0]),
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
        0.66,
        "SAM (Segment Anything)",
        "Stable Diffusion v1",
        0.0,
        7.5,
    ],
]


def main():
    # Define the function to make predictions

    def predict(
        input_image,
        input_image_camera,
        prompt,
        ratio_of_image,
        Background_detector,
        version,
        up,
        guidance_scale,
    ):
        if input_image_camera:
            input_image = input_image_camera
        else:
            input_image = input_image

        target_shape = ratio_of_image * 512

        input_tensor = preprocess(input_image, target_shape=target_shape)
        if Background_detector == "SAM (Segment Anything)":
            new_image, new_mask, mask_stable = segment_SAM(
                input_tensor,
                is_top_background=True,
                is_left_background=True,
                is_right_background=True,
                is_bottom_background=True,
                go_up=up,
                target_shape=256,
                sam_model="large",
            )
        else:
            new_image, new_mask, mask_stable = segment_torch(
                image=input_tensor,
                model=Background_detector,
                go_up=up,
                target_shape=256,
            )

        if 'Stable Diffusion' in version:
            new_image = PIL.Image.fromarray(new_image)

            # print(new_mask)
            new_mask = PIL.Image.fromarray(new_mask)
            new_image_2 = stable_diffusion(
                new_image, mask_stable, version, prompt, guidance_scale
            )
            new_image_2 = replace_pixels_with_mask(new_image_2, new_image, mask_stable)

        elif 'DallE' in version:
            folder_path = Path('data/dalle_generated')
            folder_path.mkdir(parents=True, exist_ok=True)
            
            image_path = 'data/dalle_generated/dalle_img.png'
            dalle_image = np.zeros((*new_image.shape, 4))
            dalle_image = new_image
            dalle_image = Image.fromarray(np.uint8(dalle_image))
            dalle_image.save(image_path)
            
            
            mask_path = 'data/dalle_generated/dalle_mask.png'
            dalle_mask = np.zeros((*new_mask.shape[:2], 4))
            dalle_mask[:, :, -1] = new_mask[:, :, 0]
            dalle_mask = Image.fromarray(np.uint8(dalle_mask))
            dalle_mask.save(mask_path)
            
            new_image_2 = dalle(image_path, mask_path, prompt)


            # new_image = pass
            # new_mask = 
            
            
        else:
            raise NotImplementedError(f'Given version: {version} is not implemented.')
        
        return new_image, new_mask, new_image_2

    # Create the Gradio interface
    input_image = gr.Image(type="pil", source="upload")
    input_image_camera = gr.Image(type="pil", source="webcam")

    """
    examples= [
    ['Mayan city pramid sunset ivy foliage abandoned luminiscense scultures dark sky forest stars concept landscape environment depth water waterfall river, nature, real, high quality, 4k'],
    ['A pool full of water and there is table in the background, fancy, Real, detailed, 4k'],
    ['A table, and in the background, scary lightning black and white, Real, nature, ultra detailed, 8k'],
    ['A luxury huge private yacht, sailing in the bahamas with palm trees in the background and hardwood deck on the yacht, cinematic, nature, hyperrealistic, 8 k']
            ]
    """

    Background_detector = gr.Dropdown(
        [
            "SAM (Segment Anything)",
            "deeplabv3_resnet101",
            "deeplabv3_resnet50",
            "fcn_resnet101",
            "fcn_resnet50",
            "deeplabv3_mobilenet_v3_large",
        ],
        value="SAM (Segment Anything)",
    )

    prompt = gr.Textbox(
        value="A luxury huge private yacht, sailing in the bahamas with palm trees in the background and hardwood deck on the yacht, cinematic, nature, hyperrealistic, 8 k"
    )
    ratio_of_image = gr.Slider(
        0.25,
        1,
        value=0.66,
        label="Ratio of the image to the background.",
        info="Default value: 0.66. Increase to have a smalller background",
    )

    version = gr.Dropdown(
        ["Stable Diffusion v1", "Stable Diffusion v2", "DallE"], value="Stable Diffusion v2"
    )
    up = gr.Slider(0.0, 1.0, value=0.0, label="up", info="Go up 0 to 100%")
    guidance_scale = gr.Slider(
        1.0,
        100.0,
        value=7.5,
        label="guidance_scale",
        info="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.",
    )

    large_image = gr.Image(type="pil")
    mask_output = gr.Image(type="pil")
    new_image = gr.Image(type="pil")
    article = """
            Description:
            Discover your backdground with SAM (Segment Anything)
            and seamlessly replace it with a vast background using
            your desired prompt and other configurable settings
            through a stable diffusion model or DALLE
            (DALLE is available on GitHub).
            
            Usage:
            Simply provide your image and a prompt,specify the model
            for generating your desired background. Upwards your photo
            using the "up" feature. Guidance_scale is a parameter
            specifically designed for stable diffusion models, default
            value of 7.5.
            
            GitHub:
            Due to the free version of hosting space, you might experience
            huge waiting time. However, you have the option to enhance your
            experience by downloading the updated code with additional
            features from https://github.com/FzS92/Dalle and utilizing Gradio
            from https://github.com/FzS92/Dalle/tree/main/app to achieve faster
            speeds on your local machine.
            
    """

    description = "Change your image's background using stable diffusion or DALLE"

    gr.Interface(
        fn=predict,
        inputs=[
            input_image,
            input_image_camera,
            prompt,
            ratio_of_image,
            Background_detector,
            version,
            up,
            guidance_scale,
        ],
        outputs=[large_image, mask_output, new_image],
        title="Photo Lure",
        description=description,
        article=article,
        examples=examples,
    ).launch(share=False)


if __name__ == "__main__":
    main()
