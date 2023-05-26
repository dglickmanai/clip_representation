from PIL import Image, ImageDraw
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy
from open_clip import tokenize

image_size = 224
max_objects_in_row = 8
fruits = ['apple', 'banana', 'grapes', 'kiwi']
# fruits = ['apple', 'banana']
patch_size = (image_size // max_objects_in_row, image_size // max_objects_in_row)
fruit_options = {fruit: Image.open(f'resources/images/{fruit}.jpeg').resize(patch_size, Image.BICUBIC) for fruit in
                 fruits}
fruit_options = {fruit: numpy.array(fruit_options[fruit]) for fruit in fruits}


def place_objects_in_image(color_grid, patch_size=(16, 16)):
    patch_width, patch_height = patch_size

    img = numpy.full((image_size, image_size, 3), 255)

    for i, object in enumerate(color_grid):
        if object is None:
            continue
        # x_start = j * patch_width
        # y_start = i * patch_height
        x_start = (i % max_objects_in_row) * patch_width
        y_start = (i // max_objects_in_row) * patch_width
        img[y_start:y_start + patch_height, x_start:x_start + patch_width] = object
        # draw.rectangle([x_start, y_start, x_start + patch_width, y_start + patch_height], fill=color)

    img = img / 255
    return img


def random_insert_unique(source, target, unique_indexes):
    for index, item in zip(unique_indexes, source):
        target[index] = item
    return target


def permute_lists_together(list1, list2):
    zipped_list = list(zip(list1, list2))
    random.shuffle(zipped_list)
    permuted_list1, permuted_list2 = zip(*zipped_list)
    permuted_list1, permuted_list2 = list(permuted_list1), list(permuted_list2)

    return permuted_list1, permuted_list2


class ObjectsDataset(Dataset):
    def __init__(self, num_objects, num_samples, num_hard_negatives, object_options=fruit_options,
                 patch_size=patch_size, transform=None,
                 with_spaces=True, ):
        self.with_spaces = with_spaces
        num_images = num_samples // num_hard_negatives
        data = self.generate_random_objects_names(num_images, num_objects, object_options)
        self.num_hard_negatives = num_hard_negatives
        self.data = data
        self.object_options = object_options
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        object_names = self.data[idx]

        objects_images = [self.object_options[obj] for obj in object_names]
        num_items = (image_size // self.patch_size[0]) ** 2
        unique_indexes = random.sample(range(num_items), len(objects_images)) if self.with_spaces else None

        samples = [self.create_text_img(num_items, object_names_permuted, objects_images_permuted, unique_indexes)
                   for object_names_permuted, objects_images_permuted in
                   [permute_lists_together(object_names, objects_images) for _ in range(self.num_hard_negatives)]]
        images = [sample[0] for sample in samples]
        texts = [sample[1] for sample in samples]
        return torch.stack(images), torch.stack(texts)

    def create_text_img(self, num_items, object_names, objects_images, unique_indexes):
        if self.with_spaces:
            target = [None] * num_items
            objects_images = random_insert_unique(objects_images, target, unique_indexes)
        image = place_objects_in_image(objects_images, patch_size=self.patch_size)
        if self.transform:
            image = self.transform(image).float()
        text = ' '.join(object_names)
        text = tokenize([text])[0]
        return image, text

    def generate_random_objects_names(self, num_images, num_objects, object_options):
        """
        Generate an image with randomly arranged objects in a grid and return the color names.

        :param object_options: Dictionary of objects names and their image
        :return: An Image object with randomly arranged objects and a list of lists with the color names
        """

        object_names_list = [[random.choice(list(object_options.keys())) for _ in range(num_objects)] for _ in
                             range(num_images * 2)]
        if not self.with_spaces:
            # make objects unique
            object_names_list = [list(x) for x in set(tuple(x) for x in object_names_list)][:num_images]
            assert len(
                object_names_list) == num_images, f"Could not generate {num_images} unique images with {num_objects} objects"

        return object_names_list

#
# num_images = 1_000
#
# transform = ToTensor()
# num_objects = 4
# dataset = ObjectsDataset(num_objects, 1000, 8, fruit_options, patch_size, transform=transform)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# l = []
# for i in loader:
#     l.extend(i.tolist())
# print('a')
# for i in range(len(dataset)):
#     image, color_names_grid = dataset[i]
#     print(f"Image shape: {image.shape}, Color names grid: {color_names_grid}")
