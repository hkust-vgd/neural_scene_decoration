from math import log2

from einops import rearrange
from gsa_pytorch import GSA
import torch
from torch import nn
import torch.nn.functional as F

from scenegen.helpers import LabelsDownsampler, Rezero


class Spade(nn.Module):
    def __init__(self, feat_nc, label_nc, hidden_nc, ks):
        super(Spade, self).__init__()

        pw = ks // 2

        self.norm = nn.BatchNorm2d(feat_nc, affine=False)

        self.shared = nn.Conv2d(label_nc, hidden_nc, kernel_size=ks, padding=pw)
        self.gamma = nn.Conv2d(hidden_nc, feat_nc, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(hidden_nc, feat_nc, kernel_size=ks, padding=pw)

    def forward(self, feat, label):
        hidden = self.shared(label)
        gamma = self.gamma(hidden)
        beta = self.beta(hidden)

        norm_feat = self.norm(feat)
        out_feat = norm_feat * (1 + gamma) + beta
        return out_feat


class SpadeBlock(nn.Module):
    def __init__(self, feat_nc, label_nc, hidden_nc, ks, blocks_count=2, skip=True):
        super(SpadeBlock, self).__init__()

        self.count = blocks_count
        self.skip = skip
        self.layers = nn.ModuleList([])

        for i in range(blocks_count):
            self.layers.append(Spade(feat_nc, label_nc, hidden_nc, ks))

    def forward(self, feat, label):
        out_feat = F.relu(self.layers[0](feat, label))
        for i in range(1, self.count):
            out_feat = F.relu(self.layers[i](out_feat, label))

        if self.skip:
            out_feat = out_feat + feat

        return out_feat


class AttnBlock(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim=8):
        super(AttnBlock, self).__init__()

        self.x0_conv = nn.Conv2d(x1_dim, x1_dim, kernel_size=1)
        self.x1_conv = nn.Conv2d(x1_dim, hidden_dim, kernel_size=1)
        self.x2_conv = nn.Conv2d(x2_dim, hidden_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        n, c, h, w = x1.shape

        x1_feat = self.x1_conv(x1).view(n, -1, h * w).permute(0, 2, 1)
        x2_feat = self.x2_conv(x2).view(n, -1, h * w)
        attn_map = self.softmax(torch.bmm(x1_feat, x2_feat))

        x0_feat = self.x0_conv(x1).view(n, -1, h * w)
        out = torch.bmm(x0_feat, attn_map.permute(0, 2, 1)).view(n, c, h, w)
        out = self.gamma * out + x1
        return out


class SceneGenerator(nn.Module):
    def __init__(self, image_size, latent_dim=256, fmap_max=512, fmap_inverse_coef=12, attn_res_layers=None,
                 freq_chan_attn=False, spade_skip=True, label_mode='point', bg_cat=None,
                 _fcanet=None, _global_context=None, _blur=None):
        super(SceneGenerator, self).__init__()

        self.image_size = image_size
        self.label_channels = 10
        if image_size == 256:
            self.m_layer = 5
            self.m_channels = 128
        elif image_size == 512:
            self.m_layer = 6
            self.m_channels = 64
        else:
            raise NotImplementedError

        resolution = log2(image_size)
        fmap_max = fmap_max if fmap_max is not None else latent_dim

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            nn.BatchNorm2d(latent_dim * 2),
            nn.GLU(dim=1)
        )

        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]  # [256, 512, ..., 512, 256, 128, 64, 32, 3]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)  # 2, 3, ..., 8
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))  # {2: (256, 512), 3: (512, 512) ...}

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)  # {3: 7, 4: 8, 5: 9}

        self.label_injections = list(range(2, self.m_layer + 1))
        self.downsampler = LabelsDownsampler(self.label_injections, image_size, label_mode)

        self.bg_cat = bg_cat

        # Background encoder
        bg_encoder_map = [(3, 32), (32, self.m_channels)]
        self.merge_block = SpadeBlock(feat_nc=self.m_channels, label_nc=self.m_channels, hidden_nc=64, ks=3, blocks_count=2, skip=True)

        self.encoder_layers = nn.ModuleList([])
        for (chan_in, chan_out) in bg_encoder_map:
            layer = nn.Sequential(
                nn.Conv2d(chan_in, chan_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(chan_out),
                nn.LeakyReLU(0.1)
            )
            self.encoder_layers.append(layer)

        if self.bg_cat == 'attn':
            self.bg_pool = list(range(self.m_layer, 1, -1))
            self.bg_pool_layers = nn.ModuleList([])
            for _ in self.bg_pool:
                pool_layer = nn.AvgPool2d(kernel_size=2)
                self.bg_pool_layers.append(pool_layer)
            
            self.attn_blocks = nn.ModuleDict({
                str(res): AttnBlock(x1_dim=self.label_channels, x2_dim=self.m_channels) for res in self.label_injections
            })

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2 ** res

            label_injector = None
            if res in self.label_injections:
                hidden_nc = chan_out // 2
                label_nc = self.label_channels
                if self.bg_cat == 'concat':
                    label_nc += 3

                label_injector = SpadeBlock(feat_nc=chan_in, label_nc=self.label_channels, hidden_nc=hidden_nc, ks=3,
                                            blocks_count=1, skip=spade_skip)

            attn = None
            if image_width in attn_res_layers:
                attn = Rezero(GSA(dim=chan_in, norm_queries=True))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = _fcanet(chan_in=chan_out, chan_out=sle_chan_out, width=2**(res+1))
                else:
                    sle = _global_context(chan_in=chan_out, chan_out=sle_chan_out)

            layer = nn.ModuleList([
                label_injector,
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    _blur(),
                    nn.Conv2d(chan_in, chan_out*2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(chan_out * 2),
                    nn.GLU(dim=1)
                ),
                sle,
                attn
            ])
            self.layers.append(layer)

        self.out_conv = nn.Conv2d(features[-1], 3, kernel_size=3, padding=1)

    def forward(self, x, labels, empty):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)
        x = F.normalize(x, dim=1)

        resized_labels = self.downsampler(labels)
        labels_ptr = len(resized_labels) - 1

        residuals = dict()
        bg_pool_feats = {}

        bg_feat = empty
        res = int(log2(self.image_size))
        for down in self.encoder_layers:
            bg_feat = down(bg_feat)
            res -= 1
            if res in self.label_injections:
                bg_pool_feats[res] = bg_feat

        if self.bg_cat == 'attn':
            bg_pool_feat = bg_feat
            for res, pool in zip(self.bg_pool, self.bg_pool_layers):
                bg_pool_feat = pool(bg_pool_feat)
                if res in self.label_injections:
                    bg_pool_feats[res] = bg_pool_feat

        for (res, (label_injector, up, sle, attn)) in zip(self.res_layers, self.layers):
            if attn is not None:
                x = attn(x) + x

            if label_injector is not None:
                if self.bg_cat == 'concat':
                    resized_bg = F.interpolate(empty, 2 ** res)
                    labels_with_bg = torch.cat((resized_bg, resized_labels[labels_ptr]), dim=1)
                    x = label_injector(x, labels_with_bg)
                elif self.bg_cat == 'attn':
                    attn_labels = self.attn_blocks[str(res)](resized_labels[labels_ptr], bg_pool_feats[res])
                    x = label_injector(x, attn_labels)
                else:
                    x = label_injector(x, resized_labels[labels_ptr])
                labels_ptr -= 1

            x = up(x)

            if res == self.m_layer:
                x = self.merge_block(bg_feat, x)

            if sle is not None:
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

        return self.out_conv(x)
