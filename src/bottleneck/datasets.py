from PIL import Image, ImageDraw
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy

image_size = 224
max_objects_in_row = 4


def place_objects_in_image(color_grid, patch_size=(16, 16)):
    patch_width, patch_height = patch_size

    img = numpy.full((image_size, image_size, 3), 255)

    for i, object in enumerate(color_grid):
        # x_start = j * patch_width
        # y_start = i * patch_height
        x_start = (i % max_objects_in_row) * patch_width
        y_start = (i // max_objects_in_row) * patch_width
        img[y_start:y_start + patch_height, x_start:x_start + patch_width] = object
        # draw.rectangle([x_start, y_start, x_start + patch_width, y_start + patch_height], fill=color)

    return img


def generate_random_objects_images(num_images, num_objects, object_options, patch_size=(16, 16)):
    """
    Generate an image with randomly arranged objects in a grid and return the color names.

    :param object_options: Dictionary of objects names and their image
    :return: An Image object with randomly arranged objects and a list of lists with the color names
    """

    object_names_list = [[random.choice(list(object_options.keys())) for _ in range(num_objects)] for _ in
                         range(int(num_images * 1.5))]
    object_names_list = [list(x) for x in set(tuple(x) for x in object_names_list)][:num_images]
    assert len(
        object_names_list) == num_images, f"Could not generate {num_images} unique images with {num_objects} objects"

    objects_list = [[object_options[obj] for obj in object_names] for object_names in object_names_list]
    images = [place_objects_in_image(objects_names, patch_size=patch_size) for objects_names in objects_list]
    return list(zip(images, object_names_list))


class ColorGridDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Initialize the ColorGridDataset.

        :param data: List of (image, color_names_grid) tuples
        :param transform: Optional torchvision transforms to be applied to the images
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, color_names_grid = self.data[idx]

        if self.transform:
            image = self.transform(image)

        return image, color_names_grid


fruits = ['apple', 'banana', 'grapes', 'kiwi']

patch_size = (image_size // max_objects_in_row, image_size // max_objects_in_row)
fruit_options = {fruit: Image.open(f'images/{fruit}.jpeg').resize(patch_size, Image.BICUBIC) for fruit in fruits}
fruit_options = {fruit: numpy.array(fruit_options[fruit]) for fruit in fruits}

num_objects = 6
num_images = 1000
data = generate_random_objects_images(num_images, num_objects, fruit_options, patch_size)

# Create the ColorGridDataset
transform = ToTensor()
dataset = ColorGridDataset(data, transform=transform)

# Example usage
for i in range(len(dataset)):
    image, color_names_grid = dataset[i]
    print(f"Image shape: {image.shape}, Color names grid: {color_names_grid}")
