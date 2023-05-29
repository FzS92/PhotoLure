import gradio as gr
import torch
from torchvision.transforms import ToTensor, ToPILImage
import PIL

from segment_anything import segment
from stable_diffusion import stable_diffusion
# Load your pre-trained PyTorch model
# Define the transformation functions for image conversion
transform_to_tensor = ToTensor()
transform_to_pil = ToPILImage()


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


# Define the function to postprocess the output image


def postprocess(output_tensor):
    # output_tensor = output_tensor.squeeze(0)
    output_image = transform_to_pil(output_tensor.permute(2, 0, 1))
    return output_image

# Define the function to make predictions


def predict(input_image, prompt, version, up=0.0, target_shape=256):
    input_tensor = preprocess(input_image, target_shape=target_shape)
    new_image, new_mask, mask_stable = segment(input_tensor, is_top_background=True, is_left_background=True, is_right_background=True,
                                               is_bottom_background=True, go_up=up, target_shape=256, sam_model="large")
    # print(new_image.shape)
    # print(new_image)
    new_image = PIL.Image.fromarray(new_image)

    # print(new_mask)
    new_mask = PIL.Image.fromarray(new_mask)
    new_image_2 = stable_diffusion(
        new_image, mask_stable, version=version, prompt=prompt)
    return new_image, new_mask, new_image_2


# Create the Gradio interface
input_image = gr.Image(type="pil")
prompt = gr.Textbox(
    value='A luxury huge private yacht, sailing in the bahamas with palm trees in the background and hardwood deck on the yacht, cinematic, nature, hyperrealistic, 8 k')
version = gr.Dropdown(
    ["Stable Diffusion v1", "Stable Diffusion v2"], value="Stable Diffusion v2")
up = gr.Slider(0.0, 1.0, value=0.0, label="up",
               info="Go up 0 to 100%")
large_image = gr.Image(type="pil")
mask_output = gr.Image(type="pil")
new_image = gr.Image(type="pil")


a = gr.Interface(fn=predict, inputs=[input_image, prompt, version, up], outputs=[
    large_image, mask_output, new_image]).launch(share=False)
