import gradio as gr
import numpy as np
import PIL
from PIL import Image
from pathlib import Path


from src.chat import chat_gpt
from src.Dalle import dalle
from src.deeplab import segment_torch
from src.funcs import examples, preprocess, replace_pixels_with_mask
from src.segment_anything import segment_SAM
from src.stable_diffusion import stable_diffusion


def main():
    # Define the function to make predictions

    def predict(
        input_image,
        input_image_camera,
        prompt,
        chatgpt,
        ratio_of_image,
        Background_detector,
        version,
        up,
        guidance_scale,
    ):
        # prompt = gpt_prompt(prompt)

        if chatgpt == "Input Prompt":
            print("Using given input prompt")
            prompt = prompt
        elif chatgpt == "Chat GPT":
            print("Using GPT generated prompt")
            prompt = chat_gpt(prompt)
            # gpt_prompt.lunch

        if input_image_camera:
            input_image = input_image_camera
        else:
            input_image = input_image

        target_shape = ratio_of_image * 512

        input_tensor = preprocess(
            input_image, target_shape=target_shape
        )  # Reshape input image to target shape
        if Background_detector == "SAM (Segment Anything)":
            # Find the background using SAM
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
            # Find the background using torch models
            new_image, new_mask, mask_stable = segment_torch(
                image=input_tensor,
                model=Background_detector,
                go_up=up,
                target_shape=256,
            )

        if "Stable Diffusion" in version:
            # Change the background using Stable Diffusion
            new_image = PIL.Image.fromarray(new_image)

            # print(new_mask)
            new_mask = PIL.Image.fromarray(new_mask)
            new_image_2 = stable_diffusion(
                new_image, mask_stable, version, prompt, guidance_scale
            )
            new_image_2 = replace_pixels_with_mask(new_image_2, new_image, mask_stable)

        elif "DallE" in version:
            # Change the background using DallE
            folder_path = Path("data/dalle_generated")
            folder_path.mkdir(parents=True, exist_ok=True)

            image_path = "data/dalle_generated/dalle_img.png"
            dalle_image = np.zeros((*new_image.shape, 4))
            dalle_image = new_image
            dalle_image = Image.fromarray(np.uint8(dalle_image))
            dalle_image.save(image_path)

            mask_path = "data/dalle_generated/dalle_mask.png"
            dalle_mask = np.zeros((*new_mask.shape[:2], 4))
            dalle_mask[:, :, -1] = new_mask[:, :, 0]
            dalle_mask = Image.fromarray(np.uint8(dalle_mask))
            dalle_mask.save(mask_path)

            new_image_2 = dalle(image_path, mask_path, prompt)

            # new_image = pass
            # new_mask =

        else:
            raise NotImplementedError(f"Given version: {version} is not implemented.")

        return prompt, new_image, new_mask, new_image_2

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
        value=(
            "A luxury huge private yacht, sailing in the bahamas with palm trees in the"
            " background and hardwood deck on the yacht, cinematic, nature,"
            " hyperrealistic, 8 k"
        ),
    )
    Used_Prompt = gr.Textbox(
        value="Used Prompt",
    )
    chatgpt = gr.Dropdown(["Chat GPT", "Input Prompt"], value="Chat GPT")

    ratio_of_image = gr.Slider(
        0.25,
        1,
        value=0.66,
        label="Ratio of the image to the background.",
        info="Default value: 0.66. Increase to have a smalller background",
    )

    version = gr.Dropdown(
        ["Stable Diffusion v1", "Stable Diffusion v2", "DallE"], value="DallE"
    )
    up = gr.Slider(0.0, 1.0, value=0.0, label="up", info="Go up 0 to 100%")
    guidance_scale = gr.Slider(
        1.0,
        100.0,
        value=7.5,
        label="guidance_scale",
        info=(
            "Higher guidance scale encourages to generate images that are closely"
            " linked to the text prompt, usually at the expense of lower image quality."
        ),
    )

    large_image = gr.Image(type="pil")
    mask_output = gr.Image(type="pil")
    new_image = gr.Image(type="pil")
    article = """
            Description:
            Discover your backdground with SAM (Segment Anything) or other models
            and seamlessly replace it with a vast background using
            your desired prompt by the user or ChatGPT and other configurable settings
            through a stable diffusion model or DALLE.
            
            
            Usage:
            Simply provide your image and a prompt,specify the model
            for generating your desired background. Upwards your photo
            using the "up" feature. Guidance_scale is a parameter
            specifically designed for stable diffusion models, default
            value of 7.5.            
    """

    description = "Change your image's background using stable diffusion or DALLE"

    gr.Interface(
        fn=predict,
        inputs=[
            input_image,
            input_image_camera,
            prompt,
            chatgpt,
            ratio_of_image,
            Background_detector,
            version,
            up,
            guidance_scale,
        ],
        outputs=[Used_Prompt, large_image, mask_output, new_image],
        title="Photo Lure",
        description=description,
        article=article,
        examples=examples,
    ).launch(share=True)


if __name__ == "__main__":
    main()
