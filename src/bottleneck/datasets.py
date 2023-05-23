from PIL import Image, ImageDraw
import random

def generate_color_grid_image(color_grid, patch_size=(20, 20)):
    """
    Generate an image with colors arranged in a grid according to the input nested list.

    :param color_grid: List of lists of colors in RGB tuples
                       (e.g., [[(255, 0, 0), (0, 255, 0)], [(0, 0, 255), (255, 255, 0)]])
    :param patch_size: Tuple representing the size of each color patch (width, height)
    :return: An Image object with the colors arranged in a grid
    """

    patch_width, patch_height = patch_size
    grid_rows = len(color_grid)
    grid_cols = len(color_grid[0])

    img_width = grid_cols * patch_width
    img_height = grid_rows * patch_height
    img = Image.new("RGB", (img_width, img_height))
    draw = ImageDraw.Draw(img)

    for i, row in enumerate(color_grid):
        for j, color in enumerate(row):
            x_start = j * patch_width
            y_start = i * patch_height
            draw.rectangle([x_start, y_start, x_start + patch_width, y_start + patch_height], fill=color)

    return img


def generate_random_color_grid_image(grid_size, color_options, patch_size=(20, 20)):
    """
    Generate an image with randomly arranged colors in a grid and return the color names.

    :param grid_size: Tuple representing the size of the grid (rows, cols)
    :param color_options: Dictionary of color names and their RGB tuples
                          (e.g., {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255)})
    :param patch_size: Tuple representing the size of each color patch (width, height)
    :return: An Image object with randomly arranged colors in a grid, and a list of lists with the color names
    """

    def create_random_color_grid(grid_size, color_options):
        grid_rows, grid_cols = grid_size
        color_names_grid = [[random.choice(list(color_options.keys())) for _ in range(grid_cols)] for _ in
                            range(grid_rows)]
        color_grid = [[color_options[color_name] for color_name in row] for row in color_names_grid]
        return color_grid, color_names_grid

    color_grid, color_names_grid = create_random_color_grid(grid_size, color_options)
    image = generate_color_grid_image(color_grid, patch_size=patch_size)
    return image, color_names_grid


color_options = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
                 'sky': (0, 255, 255), 'purple': (255, 0, 255)}
grid_size = (3, 3)

import torch
from PIL import Image
from torch.utils.data import Dataset


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


from torchvision.transforms import ToTensor

data = [generate_random_color_grid_image(grid_size, color_options) for _ in range(10)]

data[0][0].show()

# Create the ColorGridDataset
transform = ToTensor()
dataset = ColorGridDataset(data, transform=transform)

# Example usage
for i in range(len(dataset)):
    image, color_names_grid = dataset[i]
    print(f"Image shape: {image.shape}, Color names grid: {color_names_grid}")
