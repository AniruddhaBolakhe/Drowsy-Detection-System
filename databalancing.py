import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from random import choice

def augment_images(image_paths, augmenters, num_augmented_images):
    augmented_images = []
    for i in range(num_augmented_images):
        image_path = choice(image_paths)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aug_image = augmenters(image=image)
        augmented_images.append(aug_image)
    return augmented_images

def save_images(images, output_folder, base_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"{base_name}aug--{i}.png")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image)

def dynamic_balancing_augmentation(input_folder_class1, input_folder_class2, output_folder_class1, output_folder_class2, augmenters, target_count):
    class1_images = [os.path.join(input_folder_class1, f) for f in os.listdir(input_folder_class1) if f.endswith(('png', 'jpg', 'jpeg'))]
    class2_images = [os.path.join(input_folder_class2, f) for f in os.listdir(input_folder_class2) if f.endswith(('png', 'jpg', 'jpeg'))]
    
    num_class1 = len(class1_images)
    num_class2 = len(class2_images)
    
    if num_class1 < target_count:
        num_augmentations_class1 = (target_count+num_class1) - num_class1
        augmented_class1 = augment_images(class1_images, augmenters, num_augmentations_class1)
        save_images(augmented_class1, output_folder_class1, "class1")
    
    if num_class2 < target_count:
        num_augmentations_class2 = (target_count+num_class2) - num_class2 
        augmented_class2 = augment_images(class2_images, augmenters, num_augmentations_class2)
        save_images(augmented_class2, output_folder_class2, "class2")

# Define augmenters using imgaug
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Affine(rotate=(-20, 20)),  # rotate images by -20 to 20 degrees
    iaa.Multiply((0.8, 1.2)),  # change brightness of images
    # iaa.GaussianBlur(sigma=(0.0, 3.0)),  # blur images with a sigma between 0 and 3.0
    # iaa.AddToHueAndSaturation((-50, 50)),  # change hue and saturation
    # iaa.ContrastNormalization((0.5, 2.0))  # apply contrast normalization
])

# Define input and output folders
input_folder_class1 = 'C:/Users/User/Desktop/input1'
input_folder_class2 = 'C:/Users/User/Desktop/input2'
output_folder_class1 = 'C:/Users/User/Desktop/output1'
output_folder_class2 = 'C:/Users/User/Desktop/output2'

# Target number of images for each class
target_count = 6000

# Perform dynamic balancing augmentation
dynamic_balancing_augmentation(input_folder_class1, input_folder_class2, output_folder_class1, output_folder_class2, augmenters, target_count)