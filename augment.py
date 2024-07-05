import albumentations as A
import cv2
import os

# Input folder with 1 image
input_folder = 'C:/Users/User/Desktop/input1'

# Output folder for augmented images
output_folder = 'C:/Users/User/Desktop/output1'

# Load the image from the input folder
image_path = os.path.join(input_folder, os.listdir(input_folder)[0])
image = cv2.imread(image_path)

# Define the augmentation pipeline
transform = A.Compose([
    A.RandomRotate90(p=1.0),
    A.Flip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit=0.3, p=1.0),
    #A.RandomGamma(gamma_limit=(1.0, 2.0), p=1.0),
    #  A.ChannelShuffle(p=0.5),
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0)
])

# Generate 10 augmented images
for i in range(10):
    augmented_image = transform(image=image)['image']
    cv2.imwrite(os.path.join(output_folder, f'augmented_{i}.jpg'), augmented_image)