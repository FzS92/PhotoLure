# Ref: https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb

# from segment_anything import SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from transformers import SamModel, SamProcessor


def segment_SAM(
    image,
    is_top_background=True,
    is_left_background=True,
    is_right_background=True,
    is_bottom_background=True,
    go_up=0.0,
    target_shape=256,
    sam_model="large",
):
    # Specify image path, y position of image in the larger image, and  background lines
    # img_path = "./images/drunk1.jpg"
    is_top_background = is_top_background
    is_left_background = is_left_background
    is_right_background = is_right_background
    is_right_background = is_right_background
    go_up = go_up

    server_or_drive = "server"  # "drive" or "server"
    device = "cuda"

    # target_path = "./data"
    target_shape = target_shape
    output_shape = 512

    sam_model = sam_model  # "small" "large", huge"

    # # Check if the results directory exists
    # if not os.path.exists("results"):
    #     # If it doesn't exist, create it
    #     os.makedirs("results")

    # Functions

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )

    def show_boxes_on_image(raw_image, boxes):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis("on")
        plt.show()

    def show_points_on_image(raw_image, input_points, input_labels=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        show_points(input_points, labels, plt.gca())
        plt.axis("on")
        plt.show()

    def show_points_and_boxes_on_image(
        raw_image, boxes, input_points, input_labels=None
    ):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        show_points(input_points, labels, plt.gca())
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis("on")
        plt.show()

    def show_points_and_boxes_on_image(
        raw_image, boxes, input_points, input_labels=None
    ):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        show_points(input_points, labels, plt.gca())
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis("on")
        plt.show()

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

    # image_id = Path(img_path).stem

    def show_masks_on_image(raw_image, masks, scores):
        if len(masks.shape) == 4:
            masks = masks.squeeze()
        if scores.shape[0] == 1:
            scores = scores.squeeze()

        nb_predictions = scores.shape[-1]
        fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask = mask.cpu().detach()
            axes[i].imshow(np.array(raw_image))
            show_mask(mask, axes[i])
            axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
            axes[i].axis("off")
            # plt.savefig(f"results/result{i}{image_id}.png")

    def pil_save(addr, img):
        pil_image = Image.fromarray(np.uint8(img))
        pil_image.save(addr)

    def resize_image(image):
        # Read the image
        # print(image)
        # Get the current width and height
        # print(image.shape)
        height, width = image.shape[:2]
        # print(height, width)
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

        # image = np.array(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # print(image)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC
        )
        # print(image)
        pil_save("check.png", image * 255)
        return image, new_width, new_height

    # make the target directory
    # target_path = Path(target_path)
    # target_path.mkdir(exist_ok=True)

    # load and resize the image
    # image, width, height = resize_image(image)
    width, height = image.size

    # The Segment Anything model
    # model = "vit_l"  # vit_b, vit_l, vit_h
    # checkpoint = "./sam_vit_l_0b3195.pth"

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
        input_point_right[:, 0] = np.ones(height) * (width - 2)
        input_point = np.append(input_point, input_point_right, axis=0)

    if is_bottom_background:
        input_point_bottom = np.zeros((width, 2), dtype=np.int32)
        input_point_bottom[:, 0] = np.arange(width)
        input_point_bottom[:, 1] = np.ones(width) * (height - 2)
        input_point = np.append(input_point, input_point_bottom, axis=0)

    if (
        (not is_top_background)
        and (not is_right_background)
        and (not is_bottom_background)
        and (not is_left_background)
    ):
        raise ValueError("At least one edge is needed.")

    input_point = np.unique(input_point, axis=0)
    input_point = np.expand_dims(input_point, axis=0)

    # input_label = np.ones(input_point.shape[0], dtype=np.int32)

    print(f"Loading the model on {device}")
    if server_or_drive == "server":  # load and save
        model = SamModel.from_pretrained(f"facebook/sam-vit-{sam_model}").to(device)
        processor = SamProcessor.from_pretrained(f"facebook/sam-vit-{sam_model}")
        model.save_pretrained(f"./sam_model_{sam_model}.pth")
        processor.save_pretrained(f"./sam_processor_{sam_model}.pth")
    elif server_or_drive == "drive":
        model = SamModel.from_pretrained(f"./sam_model_{sam_model}.pth").to(device)
        processor = SamProcessor.from_pretrained(f"./sam_processor_{sam_model}.pth")
    else:
        raise ValueError("set tp device or server.")

    print("Done!")

    # segment the image
    print("Finding the background")

    inputs = processor(image, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    input_point = np.array(input_point, dtype=int)
    input_point = input_point.tolist()
    inputs = processor(image, input_points=input_point, return_tensors="pt").to(device)

    # pop the pixel_values as they are not neded
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    # scores = outputs.iou_scores

    # print(masks)

    ## Print all masks for the first time to choose the right mask ###
    # show_masks_on_image(image, masks[0], scores)

    # Increasing backgound space
    # Create a new image with extra pixels
    print("adding additional backgound space")
    x_offset, y_offset = int((output_shape - width) / 2), output_shape - height
    new_height, new_width = output_shape, output_shape
    new_image = np.ones((new_height, new_width, 3), np.uint8) * 255  # White pixels
    start_y = int(y_offset * (1 - go_up))
    transform_to_tensor = ToTensor()
    numpy_image = transform_to_tensor(image).permute(1, 2, 0)
    new_image[start_y : height + start_y, x_offset : width + x_offset] = (
        numpy_image * 255
    )

    # Create a new mask with extra pixels
    new_mask = np.ones((new_width, new_height), dtype=bool)
    new_mask[start_y : height + start_y, x_offset : width + x_offset] = masks[
        0
    ].squeeze()[-1]

    # choose the right mask ***************
    # mask = 1 - masks[-1]
    mask = 1 - new_mask

    # make the mask RGBA
    new_mask_2 = np.zeros((*mask.shape, 4))
    new_mask_print = np.zeros((*mask.shape, 3))
    new_mask_print[:, :, 0] = mask
    new_mask_print[:, :, 1] = mask
    new_mask_print[:, :, 2] = mask

    new_mask_2[:, :, -1] = mask

    # print(image.shape, new_mask.shape)

    # save mask and image
    # image_id = Path(img_path).stem
    # pil_save(str(target_path / f'{image_id}.png'), new_image)
    # pil_save(str(target_path / f'mask_{image_id}.png'), new_mask*255)

    # print("Files are saved")

    return (
        new_image,
        (new_mask_print * 255).astype(np.uint8),
        (new_mask * 255).astype(np.uint8),
    )
