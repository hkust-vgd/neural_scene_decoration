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

# HEIGHT = 720
# WIDTH = 1280
# SIZE = 256
# MARGIN = 25
# SIZE = 512
# MARGIN = 50
# MAX_X = int((WIDTH - HEIGHT) / HEIGHT * SIZE)

# Panorama
HEIGHT = 512
WIDTH = 1024
SIZE = 512
MARGIN = -1
MAX_X = -1

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


class PanoramaHorizontalAugment(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.pos = 0
        self.flip = False
        self.width = width
    
    def reset(self):
        self.pos = np.random.randint(0, self.width)
        self.flip = np.random.rand() > 0.5
    
    def forward(self, img):
        if self.flip:
            img = F.hflip(img)
        
        img = torch.cat((img[..., self.pos:], img[..., :self.pos]), dim=-1)
        return img


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


class FixedParamsStaticCrop(torch.nn.Module):
    def __init__(self, x_crop):
        super().__init__()
        self.x_crop = x_crop
        self.cur_id = None
    
    def forward(self, img):
        assert self.cur_id is not None
        xpos = self.x_crop[self.cur_id] * (MAX_X + SIZE) / WIDTH
        xpos = np.floor(xpos).astype(int)
        return F.crop(img, 0, xpos, SIZE, SIZE)


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
                 resize_mode='random', crop_seed=None, _expand_grayscale=None, panorama=False):
        super(SceneDataset, self).__init__()

        self.folder = folder
        self.image_size = image_size
        self.label_mode = label_mode
        self.anchor_label = anchor_label
        self.panorama = panorama

        if panorama:
            self.paths = {}
            self.paths['empty'] = sorted(list((Path(self.folder) / 'bedroom_empty_only_remove_wrongs' / split_file).glob('*.png')))
            self.paths['full'] = sorted(list((Path(self.folder) / 'bedroom_full_only_remove_wrongs' / split_file).glob('*.png')))
            # self.paths['objs_json'] = sorted(list((Path(self.folder) / 'objects' / split_file).glob('*.json')))
            def obj_repl(p):
                return str(p).replace('bedroom_full_only_remove_wrongs', 'objects').replace('full_rgb', 'objects').replace('png', 'json')
            self.paths['objs_json'] = [Path(obj_repl(p)) for p in self.paths['full']]
            self.crop = torch.nn.Identity()
        else:
            split_path = os.path.join(self.folder, 'splits', split_file)
            paths_np = np.genfromtxt(split_path, dtype='|U', delimiter=',')

            assert len(paths_np.shape) == 2 and paths_np.shape[1] in [3, 4]
            x_crop = None
            if paths_np.shape[1] == 4:
                x_crop = [int(i) for i in paths_np[:, -1]]
                paths_np = paths_np[:, :3]

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

            if x_crop is None:
                self.crop = FixedParamsRandomCrop(image_size, crop_seed)
            else:
                self.crop = FixedParamsStaticCrop(x_crop)

        expand_fn = _expand_grayscale(False)
        self.resize_mode = resize_mode
        if resize_mode == 'random':
            resize_transform = [
                transforms.Resize(image_size),
                self.crop
            ]
        elif resize_mode == 'full':
            resize_transform = [
                transforms.Resize((image_size, image_size))
            ]
        else:
            raise NotImplementedError

        img_transforms = [
            transforms.Lambda(convert_rgb),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            *resize_transform,
            transforms.ToTensor(),
            transforms.Lambda(expand_fn)
        ]

        if label_mode == 'point':
            label_transforms = [
                PointsToTensor(),
                transforms.Resize((image_size, image_size)),
                self.crop
            ]
        elif label_mode == 'box':
            label_transforms = [
                BoxesToTensor(),
                transforms.Resize((image_size, image_size)),
                self.crop
            ]
        elif label_mode == 'semantic':
            label_transforms = [
                transforms.Resize((image_size, image_size)),
                self.crop,
                ToOnehotTensor()
            ]
        else:
            raise NotImplementedError

        if self.panorama and 'train' in split_file:
            self.augment = PanoramaHorizontalAugment(image_size)
            img_transforms.append(self.augment)
            label_transforms.append(self.augment)

        self.transform = transforms.Compose(img_transforms)
        self.label_transform = transforms.Compose(label_transforms)

    def __len__(self):
        return len(self.paths['full'])

    def __getitem__(self, index):
        full_path = self.paths['full'][index]
        empty_path = self.paths['empty'][index]
        objs_json_path = self.paths['objs_json'][index]

        full_img = Image.open(full_path)
        empty_img = Image.open(empty_path)

        with open(objs_json_path, 'r') as f:
            objs_json = json.load(f)
        
        if not self.panorama:
            xpos_range = get_crop_pos(objs_json, anchor_label=self.anchor_label)

            if isinstance(self.crop, FixedParamsRandomCrop):
                self.crop.reset(xpos_range)
            else:
                self.crop.cur_id = index
        else:
            if hasattr(self, 'augment'):
                self.augment.reset()

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
            'objs_label': labels
        }

        return img_dict
