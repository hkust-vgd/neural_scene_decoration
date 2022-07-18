import numpy as np
import torch
import torchvision
import cv2
from matplotlib.cm import get_cmap


def convert_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


def convert_onehot_tensor(label, class_count=10):
    label_np = np.array(label)
    onehot_label = np.zeros((class_count, *label.size))
    for i in range(class_count):
        onehot_label[i] = (label_np == (i + 1)).astype(float)

    return torch.FloatTensor(onehot_label)


class LabelVisualizer:
    def __init__(self, image_size):
        cmap = get_cmap('tab10')

        self.image_size = image_size
        self.colors = [cmap(i)[:3] for i in range(10)]
        self.bg_color = (0, 0, 0)

        self.cmap = [self.bg_color, *self.colors]

        self.cmap = np.array(self.cmap)
        self.weights = torch.FloatTensor(range(1, 11)).reshape((-1, 1, 1))

    def convert_labels(self, label_tensor, dilate=None):
        size = (self.image_size, self.image_size)

        output_labels = []
        for label in label_tensor.cpu():
            label = (label > 0.5)
            flattened_label, _ = torch.max(label * self.weights, dim=0)
            flattened_label = flattened_label.type(torch.ByteTensor)

            label_image = self.cmap[flattened_label]
            if label_image.shape[:2] != size:
                label_image = cv2.resize(self.cmap[flattened_label], size, interpolation=cv2.INTER_NEAREST)
            if dilate is not None:
                kernel = np.ones((dilate, dilate), np.uint8)
                label_image = cv2.dilate(label_image, kernel, iterations=1)
            output_labels.append(label_image)

        output_np = np.stack(output_labels, axis=0).transpose([0, 3, 1, 2])
        output_tensor = torch.FloatTensor(output_np)
        return output_tensor
