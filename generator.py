import openai
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Parameters
img = 'data/image.png'
mask = 'data/mask2.png'
prompt = 'A sunlit indoor lounge area with a pool containing a dog'

# setting up openai
openai_key = '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'
openai.api_key = openai_key

# generate the images
response = openai.Image.create_edit(
  image=open(img, "rb"),
  mask=open(mask, "rb"),
  prompt=prompt,
  n=1,
  size="256x256"
)
image_url = response['data'][0]['url']


print(image_url)

