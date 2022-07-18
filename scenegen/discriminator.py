from functools import partial
from math import log2, floor
from random import random

import torch
from einops import rearrange
from gsa_pytorch import GSA
from torch import nn
import torch.nn.functional as F

from scenegen.helpers import Rezero, Residual, SumBranches, LabelsDownsampler


class LabelDiscriminator(nn.Module):
    def __init__(self, image_size, fmap_max=256, fmap_inverse_coef=12, label_mode='point'):
        super(LabelDiscriminator, self).__init__()

        resolutions = range(5, 1, -1)
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        chan_in_out = list(zip(features[:-1], features[1:]))
        label_chan = 10

        self.layers = nn.ModuleList([])
        for (res, ((_, chan_in), (_, chan_out))) in zip(resolutions, chan_in_out):
            self.layers.append(nn.Sequential(
                nn.Conv2d(chan_in + label_chan, chan_out, 4, stride=2, padding=1),
                nn.BatchNorm2d(chan_out),
                nn.LeakyReLU(0.1)
            ))

        last_chan = features[-1][1]
        self.to_logits = nn.Conv2d(last_chan + label_chan, 1, 4)
        self.layers.append(self.to_logits)

        # (32, 128), (16, 256), (8, 256), (4, 256)

        self.downsampler = LabelsDownsampler([2, 3, 4, 5], image_size, label_mode)

    def forward(self, x, labels):
        resized_labels = self.downsampler(labels)

        for layer, label in zip(self.layers, resized_labels):
            x = torch.cat((x, label), dim=1)
            x = layer(x)

        return x


class SceneDiscriminator(nn.Module):
    def __init__(self, *, image_size, fmap_max=512, fmap_inverse_coef=12, attn_res_layers=None,
                 use_label_disc=False, label_mode='point', _blur=None, _decoder=None):
        super(SceneDiscriminator, self).__init__()

        resolution = log2(image_size)
        resolution = int(resolution)

        init_channel = 3

        num_non_residual_layers = max(0, resolution - 8)
        residual_resolutions = range(min(8, resolution), 2, -1)
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        # features: [(8, 16), (7, 32), ..., (3, 512)]

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        chan_in_out = list(zip(features[:-1], features[1:]))

        # Non residual layers (512 -> 256)

        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            last_layer = (ind == (num_non_residual_layers - 1))
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(nn.Sequential(
                _blur(),
                nn.Conv2d(init_channel, chan_out, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1)
            ))

        # Residual layers (256 -> 8)

        self.residual_layers = nn.ModuleList([])
        for (res, ((_, chan_in), (_, chan_out))) in zip(residual_resolutions, chan_in_out):
            image_width = 2 ** res

            attn = None
            if image_width in attn_res_layers:
                attn = Rezero(GSA(dim=chan_in, batch_norm=False, norm_queries=True))

            self.residual_layers.append(nn.ModuleList([
                SumBranches([
                    nn.Sequential(
                        _blur(),
                        nn.Conv2d(chan_in, chan_out, 4, stride=2, padding=1),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(chan_out, chan_out, 3, padding=1),
                        nn.LeakyReLU(0.1)
                    ),
                    nn.Sequential(
                        _blur(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(chan_in, chan_out, 1),
                        nn.LeakyReLU(0.1)
                    )
                ]),
                attn
            ]))

        # Output layers (8 -> 1)

        last_chan = features[-1][-1]

        self.to_logits = nn.Sequential(
            _blur(),
            nn.Conv2d(last_chan, last_chan, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(last_chan, 1, 4)
        )

        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(init_channel, 64, 3, padding=1),
            Residual(Rezero(GSA(dim=64, norm_queries=True, batch_norm=False))),
            SumBranches([
                nn.Sequential(
                    _blur(),
                    nn.Conv2d(64, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    _blur(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(64, 32, 1),
                    nn.LeakyReLU(0.1)
                )
            ]),
            Residual(Rezero(GSA(dim=32, norm_queries=True, batch_norm=False))),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(32, 1, 4)
        )

        self.label_disc = LabelDiscriminator(image_size, label_mode=label_mode) if use_label_disc else None

        self.decoder1 = _decoder(chan_in=last_chan, chan_out=init_channel)
        self.decoder2 = _decoder(chan_in=features[-2][-1], chan_out=init_channel) if resolution >= 9 else None

    def forward(self, x, labels, calc_aux_loss=False):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (net, attn) in self.residual_layers:
            if attn is not None:
                x = attn(x) + x

            x = net(x)
            layer_outputs.append(x)

        out = self.to_logits(x).flatten(1)

        img_32x32 = F.interpolate(orig_img, size=(32, 32))
        out_32x32 = self.to_shape_disc_out(img_32x32)

        if self.label_disc is not None:
            layer_32x32 = layer_outputs[-3]
            layer_out = self.label_disc(layer_32x32, labels).flatten(1)
        else:
            layer_out = None

        if not calc_aux_loss:
            return out, layer_out, out_32x32, None

        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]

        recon_img_8x8 = self.decoder1(layer_8x8)
        aux_loss = F.mse_loss(recon_img_8x8, F.interpolate(orig_img, size=recon_img_8x8.shape[2:]))

        if self.decoder2 is not None:
            select_random_quadrant = lambda rand_quarant, img: \
                rearrange(img, 'b c (m h) (n w) -> (m n) b c h w', m=2, n=2)[rand_quarant]
            crop_image_fn = partial(select_random_quadrant, floor(random() * 4))
            img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))

            recon_img_16x16 = self.decoder2(layer_16x16_part)
            aux_loss_16x16 = F.mse_loss(recon_img_16x16, F.interpolate(img_part, size=recon_img_16x16.shape[2:]))

            aux_loss = aux_loss + aux_loss_16x16

        return out, layer_out, out_32x32, aux_loss
