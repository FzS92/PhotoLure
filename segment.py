from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path
import os

# Specify image path, y position of image in the larger image, and  background lines
img_path = "./images/drunk1.jpg"
is_top_background = True
is_left_background = True
is_right_background = True
is_bottom_background = True
go_up = 0.0


target_path = "./data"
target_shape = 350
output_shape = 512

# Check if the results directory exists
if not os.path.exists("results"):
    # If it doesn't exist, create it
    os.makedirs("results")


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
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                 facecolor=(0, 0, 0, 0), lw=2))


def pil_save(addr, img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.save(addr)


def resize_image(img_path):
    # Read the image
    image = cv2.imread(img_path)

    # Get the current width and height
    height, width = image.shape[:2]

    # Determine the maximum dimension
    max_dimension = max(height, width)

    # Calculate the scaling factor
    scale = target_shape / max_dimension

    # Resize the image with the scaling factor
    new_width = int(width * scale)
    new_height = int(height * scale)

    if new_width % 2 != 0:  # If the number is odd
        new_width += 1  # Add 1 to make it even
    if new_height % 2 != 0:  # If the number is odd
        new_height += 1  # Add 1 to make it even

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, dsize=(
        new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return resized_image, new_width, new_height


# make the target directory
target_path = Path(target_path)
target_path.mkdir(exist_ok=True)


# load and resize the image
image, width, height = resize_image(img_path)


# The Segment Anything model
model = "vit_l"  # vit_b, vit_l, vit_h
checkpoint = "./sam_vit_l_0b3195.pth"
device = "cpu"

# input_point = np.array([[0, 0], [255, 0]])
# input_label = np.array([1, 1])
input_point = np.empty((0, 2))

if is_top_background:
    input_point_top = np.zeros((width, 2), dtype=np.int32)
    input_point_top[:, 0] = np.arange(width)
    input_point_top[:, 1] = np.ones(width)
    input_point = np.append(input_point, input_point_top, axis=0)

if is_left_background:
    input_point_left = np.zeros((height, 2), dtype=np.int32)
    input_point_left[:, 1] = np.arange(height)
    input_point_left[:, 0] = np.ones(height)
    input_point = np.append(input_point, input_point_left, axis=0)

if is_right_background:
    input_point_right = np.zeros((height, 2), dtype=np.int32)
    input_point_right[:, 1] = np.arange(height)
    input_point_right[:, 0] = np.ones(height) * (width-2)
    input_point = np.append(input_point, input_point_right, axis=0)

if is_bottom_background:
    input_point_bottom = np.zeros((width, 2), dtype=np.int32)
    input_point_bottom[:, 0] = np.arange(width)
    input_point_bottom[:, 1] = np.ones(width) * (height-2)
    input_point = np.append(input_point, input_point_bottom, axis=0)

if (not is_top_background) and (not is_right_background) and (not is_bottom_background) and (not is_left_background):
    raise ValueError("At least one edge is needed.")

input_point = np.unique(input_point, axis=0)

input_label = np.ones(input_point.shape[0], dtype=np.int32)

print(f"Loading the model on {device}")
sam = sam_model_registry[model](checkpoint=checkpoint)
sam.to(device=device)
print("Done!")

# segment the image
print("Finding the background")
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
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    # mask = torchvision.transforms.Resize((406,503))(mask)
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f"results/result{i}{image_id}.png")


# Increasing backgound space
# Create a new image with extra pixels
print("adding additional backgound space")
x_offset, y_offset = int((output_shape - width)/2), output_shape - height
new_height, new_width = output_shape, output_shape
new_image = np.ones((new_height, new_width, 3), np.uint8) * 255  # White pixels
start_y = int(y_offset * (1 - go_up))
new_image[start_y:height+start_y, x_offset:width+x_offset] = image

# Create a new mask with extra pixels
new_mask = np.ones((new_width, new_height), dtype=bool)
new_mask[start_y:height+start_y,
         x_offset:width+x_offset] = masks[-1]

# choose the right mask ***************
# mask = 1 - masks[-1]
mask = 1 - new_mask


# make the mask RGBA
new_mask = np.zeros((*mask.shape, 4))

new_mask[:, :, -1] = mask

# print(image.shape, new_mask.shape)

# save mask and image
image_id = Path(img_path).stem
pil_save(str(target_path / f'{image_id}.png'), new_image)
pil_save(str(target_path / f'mask_{image_id}.png'), new_mask*255)

print("Files are saved")
