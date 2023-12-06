import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import ToTensor


def segment_torch(
    image,
    model,
    go_up=0.0,
    target_shape=256,
):
    print("Finding the background started")
    # Determine the maximum width and height for resizing
    # max_size = target_shape
    # image_path = "./images/celeb4.jpg"

    # Check if CUDA is available, and if not, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Load the pre-trained DeepLabV3 model
    if model == "lraspp_mobilenet_v3_large":
        model = models.segmentation.lraspp_mobilenet_v3_large
    elif model == "deeplabv3_mobilenet_v3_large":
        model = models.segmentation.deeplabv3_mobilenet_v3_large
    elif model == "deeplabv3_resnet50":
        model = models.segmentation.deeplabv3_resnet50
    elif model == "deeplabv3_resnet101":
        model = models.segmentation.deeplabv3_resnet101
    elif model == "fcn_resnet50":
        model = models.segmentation.fcn_resnet50
    elif model == "fcn_resnet101":
        model = models.segmentation.fcn_resnet101
    else:
        raise ValueError("Model name should be correct")

    model = model(pretrained=True)
    model = model.to(device)
    model.eval()

    # Preprocess the input image
    preprocess = transforms.Compose(
        [
            # transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Move the input tensor to the appropriate device
    input_batch = input_batch.to(device)

    # Enable mixed precision for GPU execution
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast():
                output = model(input_batch)["out"]
        else:
            output = model(input_batch)["out"]

    # Move the output tensor to the CPU
    output = output.to("cpu")

    # Find the predicted background class (assuming it's the class with the highest probability)
    background_class = torch.argmax(output, dim=1)[0]

    # Create a mask for the background class
    background_mask = background_class == 0

    # print(background_mask)
    output_shape = 512
    width, height = image.size
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
    new_mask[start_y : height + start_y, x_offset : width + x_offset] = background_mask

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
    print("Finding the background ended")

    return (
        new_image,
        (new_mask_print * 255).astype(np.uint8),
        (new_mask * 255).astype(np.uint8),
    )
