import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from lightweight_gan.lightweight_gan import AugWrapper, expand_greyscale
from scenegen.dataset import SceneDataset
from scenegen.helpers import LabelsDownsampler
from scenegen.utils import LabelVisualizer

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, labels):
        return x, labels


@torch.no_grad()
def DiffAugmentTest(image_size=256, split_file=None, data='./data/0.jpg', label_mode='point',
                    anchor_label=None, resize_mode='random', types=[], batch_size=10, rank=0, nrow=5):
    model = DummyModel()
    aug_wrapper = AugWrapper(model, image_size)

    if os.path.exists(data):
        dataset = SceneDataset(data, image_size, split_file=split_file+'/train_split.csv', label_mode=label_mode,
                               anchor_label=anchor_label, resize_mode=resize_mode, _expand_grayscale=expand_greyscale)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        image_dict = next(iter(dataloader))
        label_visualizer = LabelVisualizer(image_size)

        empty_image_batch = image_dict['empty_image'].cuda(rank)
        full_image_batch = image_dict['full_image'].cuda(rank)
        masks_batch = image_dict['bg_mask'].cuda(rank)
        labels_batch = image_dict['objs_label'].cuda(rank)

        full_aug_images, aug_labels = aug_wrapper(full_image_batch, labels_batch, prob=1, types=types, detach=True)

        torchvision.utils.save_image(empty_image_batch, 'empty.png', nrow=nrow)
        torchvision.utils.save_image(full_image_batch, 'full.png', nrow=nrow)
        torchvision.utils.save_image(masks_batch, 'mask.png', nrow=nrow)
        torchvision.utils.save_image(full_aug_images, 'full_aug.png', nrow=nrow)

        downsampler = LabelsDownsampler([2, 3, 4, 5], image_size=image_size, label_mode=label_mode)

        labels = downsampler(labels_batch)
        for i, idx in enumerate(range(5, 1, -1)):
            label_rgb = label_visualizer.convert_labels(labels[i].cpu())
            torchvision.utils.save_image(label_rgb, 'label_{:d}.png'.format(idx), nrow=nrow)

        labels_aug = downsampler(aug_labels)
        label_aug_rgb = label_visualizer.convert_labels(labels_aug[0].cpu())
        torchvision.utils.save_image(label_aug_rgb, 'label_5_aug.png', nrow=nrow)
