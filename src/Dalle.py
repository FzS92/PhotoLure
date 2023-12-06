import os

import matplotlib.pyplot as plt
import openai
import torchvision.transforms as transforms
import wget
from PIL import Image
from torchvision.utils import save_image


def dalle(img_path, mask_path, prompt):
    out = "data/dalle_generated/dalle_out.png"

    openai_key = "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"
    openai.api_key = openai_key

    # generate the images
    response = openai.Image.create_edit(
        image=open(img_path, "rb"),
        mask=open(mask_path, "rb"),
        prompt=prompt,
        n=1,
        size="512x512",
    )

    image_url = response["data"][0]["url"]
    if os.path.exists(out):
        os.remove(out)
    filename = wget.download(image_url, out=out)
    new_image_2 = Image.open(out)

    return new_image_2
