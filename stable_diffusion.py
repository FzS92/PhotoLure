import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import PIL


img = 'data/celeb2.png'
mask = 'data/mask_celeb2.png'
out = 'data/out_celeb2'


prompt = 'Mayan city pramid sunset ivy foliage abandoned luminiscense scultures dark sky forest stars concept landscape environment depth water waterfall river, nature, real, high quality, 4k'
negative_prompt = "face, mouth, teeth, hand, ears, fingers, body, head, hair, people, person, low quality, poor image"


device = "cpu"
server_or_drive = "drive"  # "server" or "drive"


num_samples = 3


def convert_to_rgb(image_path):
    # Open the image file
    image = Image.open(image_path)

    # Convert the image to RGB
    rgb_image = image.convert("RGB")

    return rgb_image


def extract_last_dimension(image_path):
    # Open the RGBA image
    image = Image.open(image_path)

    # Extract the last dimension
    last_dimension = image.split()[-1]

    # Apply the transformation: alpha channel = 255 - alpha
    transformed_alpha = last_dimension.point(lambda a: 255 - a)

    # Create a new image with the last dimension as grayscale
    last_dimension_image = Image.new('L', image.size)
    last_dimension_image.putdata(transformed_alpha.getdata())

    return last_dimension_image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


img = convert_to_rgb(img)
mask = extract_last_dimension(mask)

model_path = "runwayml/stable-diffusion-inpainting"

if server_or_drive == "server":  # load and save
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    ).to(device)
    pipe.save_pretrained("./pretrained_stablediffusion")
elif server_or_drive == "drive":
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "./pretrained_stablediffusion",
        torch_dtype=torch.float32,
    ).to(device)


# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is


guidance_scale = 7.5
# generator = torch.Generator(device=device).manual_seed(
#     0)  # change the seed to get different results
num_inference_steps = 50


images = pipe(
    prompt=prompt,
    image=img,
    mask_image=mask,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
    # generator=generator,
    num_images_per_prompt=num_samples,
    num_inference_steps=num_inference_steps,
).images

# insert initial image in the list so we can compare side by side
# images.insert(0, img)

# image_grid(images, 1, num_samples + 1).save("OUT1.png")

counter = 0
for img in images:
    img.save(out+str(counter)+"png")
