from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path

# Functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def pil_save(addr, img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.save(addr)

# Parameters
img_path = "images/man1.jpg"
target_path = "./data"
target_shape = (256,256)

# make the target directory
target_path = Path(target_path)
target_path.mkdir(exist_ok=True)


# load the image
image = cv2.imread(img_path) 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 503 x 406 x 3
image =  cv2.resize(image,dsize=target_shape, interpolation=cv2.INTER_CUBIC)


### The Segment Anything model
model = "vit_h"
checkpoint = "./sam_vit_h_4b8939.pth"
device = "cuda"

input_point = np.array([[0, 0]])
input_label = np.array([1])

sam = sam_model_registry[model](checkpoint=checkpoint)
sam.to(device=device)

# segment the image
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)


## Print all masks for the first time to choose the right mask ###
image_id = Path(img_path).stem
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    # mask = torchvision.transforms.Resize((406,503))(mask)
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f"results/result{i}{image_id}.png")  


# choose the right mask ***************
mask = 1- masks[-1]

# make the mask RGBA
new_mask = np.zeros((*mask.shape, 4))
new_mask[:,:,-1] = mask
print(image.shape, new_mask.shape)

# save mask and image
image_id = Path(img_path).stem
pil_save(str(target_path / f'{image_id}.png'), image)
pil_save(str(target_path / f'mask_{image_id}.png'), new_mask*255)
