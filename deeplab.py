import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

# Determine the maximum width and height for resizing
max_size = 256
image_path = "./images/celeb4.jpg"

# Check if CUDA is available, and if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(
    pretrained=True
)  # Available models: [lraspp_mobilenet_v3_large, deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101]
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

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")


# Get the current width and height of the image
width, height = image.size

# Calculate the new width and height while maintaining the aspect ratio
if width > height:
    new_width = max_size
    new_height = int(height * (max_size / width))
else:
    new_width = int(width * (max_size / height))
    new_height = max_size

# Resize the image using the calculated dimensions
image.thumbnail((new_width, new_height))


input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Move the input tensor to the appropriate device
input_batch = input_batch.to(device)

# Enable mixed precision for GPU execution
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

print(background_mask.size())

# # Apply the mask to the original image
# background = image.copy()
# background.putdata(
#     [
#         (0, 0, 0) if mask else pixel
#         for mask, pixel in zip(background_mask.flatten(), background.getdata())
#     ]
# )

# # Save the resulting background image
# background.save("background_image.jpg")
