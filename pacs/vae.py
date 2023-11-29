import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from torchmetrics import Accuracy
from torchvision.models import resnet50
from utils.nn_utils import ResidualMLP, one_hot, arr_to_cov, sample_mvn


UNET_DEPTH = 6
IMG_EMBED_SIZE = 2048


class Posterior(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.mu_causal = ResidualMLP(IMG_EMBED_SIZE + N_ENVS, h_sizes, z_size)
        self.low_rank_causal = ResidualMLP(IMG_EMBED_SIZE + N_ENVS, h_sizes, z_size * rank)
        self.diag_causal = ResidualMLP(IMG_EMBED_SIZE + N_ENVS, h_sizes, z_size)
        self.mu_spurious = ResidualMLP(IMG_EMBED_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)
        self.low_rank_spurious = ResidualMLP(IMG_EMBED_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size * rank)
        self.diag_spurious = ResidualMLP(IMG_EMBED_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)

    def forward(self, x_embed, y, e):
        batch_size = len(x_embed)
        y_one_hot = one_hot(y, N_CLASSES)
        e_one_hot = one_hot(e, N_ENVS)
        # Causal
        mu_causal = self.mu_causal(x_embed, e_one_hot)
        low_rank_causal = self.low_rank_causal(x_embed, e_one_hot)
        low_rank_causal = low_rank_causal.reshape(batch_size, self.z_size, self.rank)
        diag_causal = self.diag_causal(x_embed, e_one_hot)
        cov_causal = arr_to_cov(low_rank_causal, diag_causal)
        # Spurious
        mu_spurious = self.mu_spurious(x_embed, y_one_hot, e_one_hot)
        low_rank_spurious = self.low_rank_spurious(x_embed, y_one_hot, e_one_hot)
        low_rank_spurious = low_rank_spurious.reshape(batch_size, self.z_size, self.rank)
        diag_spurious = self.diag_spurious(x_embed, y_one_hot, e_one_hot)
        cov_spurious = arr_to_cov(low_rank_spurious, diag_spurious)
        # Block diagonal
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class Prior(nn.Module):
    def __init__(self, z_size, rank, init_sd):
        super().__init__()
        self.z_size = z_size
        self.mu_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        self.low_rank_causal = nn.Parameter(torch.zeros(N_ENVS, z_size, rank))
        self.diag_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        nn.init.normal_(self.mu_causal, 0, init_sd)
        nn.init.normal_(self.low_rank_causal, 0, init_sd)
        nn.init.normal_(self.diag_causal, 0, init_sd)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.low_rank_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, rank))
        self.diag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_spurious, 0, init_sd)
        nn.init.normal_(self.low_rank_spurious, 0, init_sd)
        nn.init.normal_(self.diag_spurious, 0, init_sd)

    def forward(self, y, e):
        batch_size = len(y)
        # Causal
        mu_causal = self.mu_causal[e]
        cov_causal = arr_to_cov(self.low_rank_causal[e], self.diag_causal[e])
        # Spurious
        mu_spurious = self.mu_spurious[y, e]
        cov_spurious = arr_to_cov(self.low_rank_spurious[y, e], self.diag_spurious[y, e])
        # Block diagonal
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None):
        super().__init__()
        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels
        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class VAE(pl.LightningModule):
    def __init__(self, task, z_size, rank, h_sizes, y_mult, beta, reg_mult, init_sd, lr, weight_decay, lr_infer,
            n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.y_mult = y_mult
        self.beta = beta
        self.reg_mult = reg_mult
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        self.val_acc = Accuracy('multiclass', num_classes=N_CLASSES)
        self.test_acc = Accuracy('multiclass', num_classes=N_CLASSES)

        # UNet down components
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        resnet = resnet50(weights='IMAGENET1K_V1')
        down_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(2048, 2048, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # VAE components
        self.posterior = Posterior(z_size, rank, h_sizes)
        self.prior = Prior(z_size, rank, init_sd)
        self.classifier = ResidualMLP(z_size, h_sizes, N_CLASSES)

        # UNet up components
        self.unpool = ResidualMLP(2 * z_size, h_sizes, 2048 * 3 * 3)
        self.tconv = nn.ConvTranspose2d(2048, 2048, 3, 2)
        up_blocks = []
        up_blocks.append(UpBlock(2048, 1024))
        up_blocks.append(UpBlock(1024, 512))
        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(
            UpBlock(in_channels=128 + 64, out_channels=128, up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlock(in_channels=64 + 3, out_channels=64, up_conv_in_channels=128, up_conv_out_channels=64))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1, stride=1)

    def elbo(self, x, y, e):
        x_embed = self.normalize(x)
        pre_pools = dict()
        pre_pools[f'layer_0'] = x_embed
        x_embed = self.input_block(x_embed)
        pre_pools[f'layer_1'] = x_embed
        x_embed = self.input_pool(x_embed)

        for i, block in enumerate(self.down_blocks, 2):
            x_embed = block(x_embed)
            if i == (UNET_DEPTH - 1):
                continue
            pre_pools[f'layer_{i}'] = x_embed

        x_embed = self.conv(x_embed) # (2048, 7, 7) -> (2048, 3, 3)
        x_embed = self.avgpool(x_embed).flatten(start_dim=1)

        # z_c,z_s ~ q(z_c,z_s|x)
        posterior_dist = self.posterior(x_embed, y, e)
        z = sample_mvn(posterior_dist)
        z_c, z_s = torch.chunk(z, 2, dim=1)

        # E_q(z_c|x)[log p(y|z_c)]
        y_pred = self.classifier(z_c)
        log_prob_y_zc = -F.cross_entropy(y_pred, y)

        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        prior_norm = (prior_dist.loc ** 2).mean()

        x_embed = self.unpool(z).reshape(-1, 2048, 3, 3)
        x_embed = self.tconv(x_embed) # (2048, 3, 3) -> (2048, 7, 7)
        for i, block in enumerate(self.up_blocks, 1):
            key = f'layer_{UNET_DEPTH - 1 - i}'
            x_embed = block(x_embed, pre_pools[key])
        x_pred = self.out_conv(x_embed)  # (3, 224, 224)
        del pre_pools
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred.flatten(start_dim=1), x.flatten(start_dim=1),
            reduction='none').sum(dim=1).mean()
        return log_prob_x_z, log_prob_y_zc, kl, prior_norm

    def training_step(self, batch, batch_idx):
        x, y, e = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_norm = self.elbo(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.reg_mult * prior_norm
        self.log('train_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('train_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('train_kl', kl, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def init_z(self, x, y_value, e_value):
        batch_size = len(x)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        pre_pools = dict()
        pre_pools[f'layer_0'] = x
        x_embed = self.input_block(x)
        pre_pools[f'layer_1'] = x_embed
        x_embed = self.input_pool(x_embed)

        for i, block in enumerate(self.down_blocks, 2):
            x_embed = block(x_embed)
            if i == (UNET_DEPTH - 1):
                continue
            pre_pools[f'layer_{i}'] = x_embed

        x_embed = self.avgpool(x_embed).flatten(start_dim=1)
        return nn.Parameter(self.posterior(x_embed, y, e).loc.detach())

    def classify_loss(self, x, y, e, z):
        pre_pools = dict()
        pre_pools[f'layer_0'] = x
        x_embed = self.input_block(x)
        pre_pools[f'layer_1'] = x_embed
        x_embed = self.input_pool(x_embed)

        for i, block in enumerate(self.down_blocks, 2):
            x_embed = block(x_embed)
            if i == (UNET_DEPTH - 1):
                continue
            pre_pools[f'layer_{i}'] = x_embed

        x_embed = self.avgpool(x_embed).flatten(start_dim=1)

        # log q(z|x,y,e)
        log_prob_z_xye = self.posterior(x_embed, y, e).log_prob(z)

        # E_q(z_c|x)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c)
        log_prob_y_zc = -F.cross_entropy(y_pred, y, reduction='none')

        x_embed = self.unpool(z).reshape(-1, 2048, 7, 7)
        for i, block in enumerate(self.up_blocks, 1):
            key = f'layer_{UNET_DEPTH - 1 - i}'
            x_embed = block(x_embed, pre_pools[key])
        x_pred = self.out_conv(x_embed)  # (3, 224, 224)
        del pre_pools
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred.flatten(start_dim=1), x.flatten(start_dim=1),
            reduction='none').sum(dim=1)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc - log_prob_z_xye
        return loss

    def opt_classify_loss(self, x, y_value, e_value):
        batch_size = len(x)
        z_param = self.init_z(x, y_value, e_value)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        optim = Adam([z_param], lr=self.lr_infer)
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            loss = self.classify_loss(x, y, e, z_param)
            loss.mean().backward()
            optim.step()
        return loss.detach().clone()

    def classify(self, x):
        loss_candidates = []
        y_candidates = []
        for y_value in range(N_CLASSES):
            for e_value in range(N_ENVS):
                loss_candidates.append(self.opt_classify_loss(x, y_value, e_value)[:, None])
                y_candidates.append(y_value)
        loss_candidates = torch.hstack(loss_candidates)
        y_candidates = torch.tensor(y_candidates, device=self.device)
        opt_loss = loss_candidates.min(dim=1)
        y_pred = y_candidates[opt_loss.indices]
        return opt_loss.values.mean(), y_pred

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y, e = batch
        with torch.set_grad_enabled(True):
            loss, y_pred = self.classify(x)
            if dataloader_idx == 0:
                self.val_acc.update(y_pred, y)
            elif dataloader_idx == 1:
                self.test_acc.update(y_pred, y)
            else:
                raise ValueError

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())
        self.log('test_acc', self.test_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y, e = batch
        with torch.set_grad_enabled(True):
            loss, y_pred = self.classify(x)
            self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)