import json
import os
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as F

from scenegen.utils import convert_rgb, resize_to_minimum_size, convert_onehot_tensor

HEIGHT = 720
WIDTH = 1280
SIZE = 256
MARGIN = 25
# SIZE = 512
# MARGIN = 50
MAX_X = int((WIDTH - HEIGHT) / HEIGHT * SIZE)

CLASS_IDS = [3, 4, 5, 6, 7, 11, 16, 25, 32, 35]


def get_crop_pos(obj_json, anchor_label, margin=MARGIN):
    if anchor_label is None:
        return 0, MAX_X

    anchors = [obj for obj in obj_json if obj['class_id'] == anchor_label]

    if len(anchors) > 0:
        anchor = anchors[0]
        anchor_xpos = anchor['centroid'][1] / HEIGHT * SIZE

        right_xpos = max(min(np.floor(anchor_xpos - margin), MAX_X), 0)
        left_xpos = min(max(np.ceil(anchor_xpos + margin - SIZE), 0), MAX_X)
        return int(left_xpos), int(right_xpos)
    else:
        return 0, MAX_X


class FixedParamsRandomCrop(transforms.RandomCrop):
    def __init__(self, size, seed):
        super(FixedParamsRandomCrop, self).__init__(size)
        self.params = None
        self.seed = seed

    def reset(self, xrange):
        if self.seed is not None:
            np.random.seed(self.seed)

        xpos = np.random.randint(xrange[0], xrange[1]+1)
        self.params = (0, xpos, SIZE, SIZE)

    def forward(self, img):
        return F.crop(img, *self.params)


class ToOnehotTensor(torch.nn.Module):
    def __init__(self):
        super(ToOnehotTensor, self).__init__()
        self.class_ids = torch.tensor(CLASS_IDS)
        self.total_classes = 41

    def forward(self, label):
        label_tensor = torch.from_numpy(np.array(label, np.int64, copy=False)).unsqueeze(dim=0)
        label_onehot_full = torch.zeros((self.total_classes, *label.size), dtype=torch.int64)
        label_onehot_full.scatter_(dim=0, index=label_tensor, src=torch.ones_like(label_tensor))
        label_onehot = torch.index_select(label_onehot_full, dim=0, index=self.class_ids)
        return label_onehot.type(torch.FloatTensor)


class BoxesToTensor(torch.nn.Module):
    def __init__(self):
        super(BoxesToTensor, self).__init__()
        self.class_ids = CLASS_IDS
        self.class_id_map = {idx: i for i, idx in enumerate(self.class_ids)}
        self.total_classes = 41

    def forward(self, objs_json):
        label_np = np.zeros((len(self.class_ids), HEIGHT, WIDTH), dtype=np.float32)

        for obj in objs_json:
            if obj['class_id'] not in self.class_ids:
                continue
            channel_id = self.class_id_map[obj['class_id']]
            x1, x2, y1, y2 = obj['bbox']
            label_np[channel_id, y1:y2+1, x1:x2+1] = 1.0

        label_tensor = torch.from_numpy(label_np)
        return label_tensor


class PointsToTensor(torch.nn.Module):
    def __init__(self):
        super(PointsToTensor, self).__init__()
        self.class_ids = CLASS_IDS
        self.class_id_map = {idx: i for i, idx in enumerate(self.class_ids)}

        x_coords = np.tile(np.arange(WIDTH).reshape((1, -1)), (HEIGHT, 1))
        y_coords = np.tile(np.arange(HEIGHT).reshape((-1, 1)), (1, WIDTH))
        self.coords = np.stack((y_coords, x_coords), axis=2)
    
    def forward(self, objs_json, k=0.4):
        label_np = np.zeros((len(self.class_ids), HEIGHT, WIDTH), dtype=np.float32)
        
        for obj in objs_json:
            if obj['class_id'] not in self.class_ids:
                continue
            channel_id = self.class_id_map[obj['class_id']]
            cy, cx = obj['centroid']
            rad = obj['mask_area'] ** 0.5 * k

            dists = self.coords - np.array([cy, cx]).reshape((1, 1, 2))
            obj_val = (dists[:, :, 0] ** 2 + dists[:, :, 1] ** 2) / (rad ** 2)
            obj_val = np.exp(obj_val * -0.5)
            label_np[channel_id, :, :] += obj_val
        
        label_tensor = torch.from_numpy(label_np)
        return label_tensor


class EvalDataset(Dataset):
    def __init__(self, folder, split_file=None):
        super(EvalDataset, self).__init__()

        self.folder = folder
        split_path = os.path.join(self.folder, split_file)
        paths_np = np.genfromtxt(split_path, dtype='|U', delimiter=',')

        self.paths = {}
        items = ['empty', 'label']
        for i, item in enumerate(items):
            self.paths[item] = [os.path.join(self.folder, record[i]) for record in paths_np]
        assert len(self.paths[items[0]]) != 0

        self.transform = transforms.Compose([
            transforms.Lambda(convert_rgb),
            transforms.ToTensor(),
            # transforms.Lambda(expand_fn)
        ])

    def __len__(self):
        item = list(self.paths.keys())[0]
        return len(self.paths[item])

    def __getitem__(self, index):
        empty_path = self.paths['empty'][index]
        label_path = self.paths['label'][index]

        label_tensor = torch.from_numpy(np.load(label_path))
        empty_img = Image.open(empty_path)

        img_dict = {
            'full_image': 0,
            'empty_image': self.transform(empty_img),
            'bg_mask': 0,
            'objs_label': label_tensor
        }

        return img_dict


class SceneDataset(Dataset):
    def __init__(self, folder, image_size, split_file=None, label_mode='point', anchor_label=None,
                 resize_mode='random', crop_seed=None, _expand_grayscale=None):
        super(SceneDataset, self).__init__()

        self.folder = folder
        self.image_size = image_size
        self.label_mode = label_mode
        self.anchor_label = anchor_label

        split_path = os.path.join(self.folder, 'splits', split_file)
        paths_np = np.genfromtxt(split_path, dtype='|U', delimiter=',')

        paths_fmts = {
            'full': '{}/data/{}/2D_rendering/{}/perspective/full/{}/rgb_rawlight.png',
            'empty': '{}/data/{}/2D_rendering/{}/perspective/empty/{}/rgb_rawlight.png',
            'semantic': '{}/data/{}/2D_rendering/{}/perspective/full/{}/semantic.png',
            'mask': '{}/data/{}/2D_rendering/{}/perspective/full/{}/bg_mask.png',
            'objs': '{}/data/{}/2D_rendering/{}/perspective/full/{}/objects.png',
            'objs_json': '{}/data/{}/2D_rendering/{}/perspective/full/{}/objects.json'
        }

        self.paths = {}
        items = ['full', 'empty', 'mask', 'objs_json']
        if label_mode == 'point':
            items.append('objs')
        elif label_mode == 'semantic':
            items.append('semantic')

        for item in items:
            self.paths[item] = [Path(paths_fmts[item].format(self.folder, *record)) for record in paths_np]
        assert len(self.paths[items[0]]) != 0

        self.random_crop = FixedParamsRandomCrop(image_size, crop_seed)

        expand_fn = _expand_grayscale(False)
        self.resize_mode = resize_mode
        if resize_mode == 'random':
            resize_transform = [
                transforms.Resize(image_size),
                self.random_crop
            ]
        elif resize_mode == 'full':
            resize_transform = [
                transforms.Resize((image_size, image_size))
            ]
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.Lambda(convert_rgb),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            *resize_transform,
            transforms.ToTensor(),
            transforms.Lambda(expand_fn)
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            self.random_crop,
            transforms.ToTensor()
        ])

        if label_mode == 'point':
            # label_transforms = [
            #     self.random_crop,
            #     convert_onehot_tensor
            # ]
            label_transforms = [
                PointsToTensor(),
                transforms.Resize(image_size),
                self.random_crop
            ]
        elif label_mode == 'box':
            label_transforms = [
                BoxesToTensor(),
                transforms.Resize(image_size),
                self.random_crop
            ]
        elif label_mode == 'semantic':
            label_transforms = [
                transforms.Resize(image_size),
                self.random_crop,
                ToOnehotTensor()
            ]
        else:
            raise NotImplementedError

        self.label_transform = transforms.Compose(label_transforms)

    def __len__(self):
        return len(self.paths['full'])

    def __getitem__(self, index):
        full_path = self.paths['full'][index]
        empty_path = self.paths['empty'][index]
        mask_path = self.paths['mask'][index]
        objs_json_path = self.paths['objs_json'][index]

        full_img = Image.open(full_path)
        empty_img = Image.open(empty_path)
        mask_img = Image.open(mask_path)

        with open(objs_json_path, 'r') as f:
            objs_json = json.load(f)
        xpos_range = get_crop_pos(objs_json, anchor_label=self.anchor_label)

        self.random_crop.reset(xpos_range)

        labels = None
        if self.label_mode == 'point':
            # objs_path = self.paths['objs'][index]
            # objects_img = Image.open(objs_path)
            # labels = self.label_transform(objects_img)
            labels = self.label_transform(objs_json)
        elif self.label_mode == 'box':
            labels = self.label_transform(objs_json)
        elif self.label_mode == 'semantic':
            semantic_path = self.paths['semantic'][index]
            semantic_img = Image.open(semantic_path)
            labels = self.label_transform(semantic_img)

        img_dict = {
            'full_image': self.transform(full_img),
            'empty_image': self.transform(empty_img),
            'bg_mask': self.mask_transform(mask_img),
            'objs_label': labels
        }

        return img_dict