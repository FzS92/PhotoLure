import openai
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
import wget

# Parameters
img = 'data/celeb2.png'
mask = 'data/mask_celeb2.png'
out = 'data/out_celeb2.png'

# out_banana1
prompt = 'A table, and in the background, scary lightning black and white, Real, nature, ultra detailed, 8k'
# out_dog1
prompt = 'A pool full of water and there is table in the background, fancy, Real, detailed, 4k'
# out celeb
prompt = 'Mayan city pramid sunset ivy foliage abandoned luminiscense scultures dark sky forest stars concept landscape environment depth water waterfall river, nature, real, high quality, 4k'

# prompt = 'A sunlit indoor lounge area with a pool containing a dog'
# prompt = 'A sunlit indoor lounge area with a pool'
# prompt = 'two young men on his luxury huge private yacht, sailing in the bahamas with palm trees in the background and hardwood deck on the yacht, cinematic, nature, hyperrealistic, 8 k'
# prompt = 'At the background scary lightning black and white, Real, nature, ultra detailed, 8k'
# prompt = 'we are in a luxury huge private yacht, sailing in the bahamas with palm trees in the background and hardwood deck on the yacht, cinematic, nature, hyperrealistic, 8 k'
# prompt = 'we are in a luxury huge private yacht on water and sunshine in the background'

# prompt = 'Two scary dogs are looking at the camera at night, real, hidden cammera, night 4k'
# prompt = 'Mayan city pramid sunset ivy foliage abandoned luminiscense scultures dark sky forest stars concept landscape environment depth water waterfall river realistyc'


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
filename = wget.download(image_url, out=out)
print(image_url)
