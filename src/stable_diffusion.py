import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def stable_diffusion(img, mask_stable, version, prompt, guidance_scale):
    # # celeb2
    # prompt = 'Mayan city pramid sunset ivy foliage abandoned luminiscense scultures dark sky forest stars concept landscape environment depth water waterfall river, nature, real, high quality, 4k'
    # # banana
    # prompt = 'A table, and in the background, scary lightning black and white, Real, nature, ultra detailed, 8k'
    # # dog
    # prompt = 'A pool full of water and there is table in the background, fancy, Real, detailed, 4k'
    # # drunk
    # prompt = 'A luxury huge private yacht, sailing in the bahamas with palm trees in the background and hardwood deck on the yacht, cinematic, nature, hyperrealistic, 8 k'

    negative_prompt = (
        "blurry, duplicate, low quality, unreal, ugly, cropped, gross proportions,"
        " malformed limbs, lowres, mutation, mutilated, morbid, watermark, worst"
        " quality, cloned face, out of frame, signature, bad anatomy, cropped,"
        " disfigured, error, deformed,dehydrated,  bad proportions, extra arms, long"
        " neck, extra fingers,   text, extra legs, animation, cartoon, deformed, ugly,"
        " face, mouth, teeth, hand, ears, fingers, body, head, hair, people, person,"
        " poor image, poorly Rendered face, username, too many fingers, poorly drawn"
        " hands, poorly drawn face, missing legs"
    )

    device = "cuda"
    server_or_drive = "server"  # "server" or "drive"

    num_samples = 1

    def convert_to_rgb(image):
        # Open the image file
        # image = Image.open(image_path)

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
        last_dimension_image = Image.new("L", image.size)
        last_dimension_image.putdata(transformed_alpha.getdata())

        return last_dimension_image

    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = PIL.Image.new("RGB", size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    img = convert_to_rgb(img)
    mask = mask_stable
    if version == "Stable Diffusion v1":
        model_path = "runwayml/stable-diffusion-inpainting"
    if version == "Stable Diffusion v2":
        model_path = "stabilityai/stable-diffusion-2-inpainting"

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

    guidance_scale = guidance_scale
    # generator = torch.Generator(device=device).manual_seed(
    #     0)  # change the seed to get different results
    num_inference_steps = 50
    # print(mask)
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

    # counter = 0
    # for img in images:
    #     img.save(out + "_" + str(counter)+"_stablediffusion.png")
    #     counter += 1

    return images[0]
