import openai
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Parameters
img = 'data/IMG-20220730-WA0037.png'
mask = 'data/mask_IMG-20220730-WA0037.png'
# prompt = 'A sunlit indoor lounge area with a pool containing a dog'
# prompt = 'two young men on his luxury huge private yacht, sailing in the bahamas with palm trees in the background and hardwood deck on the yacht, cinematic, nature, hyperrealistic, 8 k'
prompt = 'At the background scary lightning black and white, Real, nature, ultra detailed, 8k'

# setting up openai
openai_key = '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'
openai.api_key = openai_key

# generate the images
response = openai.Image.create_edit(
    image=open(img, "rb"),
    mask=open(mask, "rb"),
    prompt=prompt,
    n=1,
    size="512x512"
)
image_url = response['data'][0]['url']


print(image_url)
