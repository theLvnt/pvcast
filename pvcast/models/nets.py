"""Deep forecasters: sky-image CNN and multimodal fusion network.

`MultiModalNet` has two branches that can be switched on/off:
  * image branch  — stack of sky images (t-15 … t) + last-frame difference (motion cue)
  * tabular branch — PV history, clear-sky index, weather, solar geometry
and predicts the clear-sky index at t+h (output = k̂ · PV_clearsky(t+h)).

  image only  + PV history  -> a SUNSET-style image model (Sun et al., 2019)
  image + tabular           -> the multimodal model
Modality dropout (randomly blanking the image branch during training) keeps the
fused model usable when the camera is offline.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.skippd import PV_CAPACITY_KW
from ..features import Samples

K_MAX = 1.5


class ImageEncoder(nn.Module):
    def __init__(self, in_ch: int, width: int = 24, out_dim: int = 128):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.net = nn.Sequential(blk(in_ch, width), nn.MaxPool2d(2),            # 32
                                 blk(width, 2 * width), nn.MaxPool2d(2),        # 16
                                 blk(2 * width, 4 * width), nn.MaxPool2d(2),    # 8
                                 blk(4 * width, out_dim), nn.AdaptiveAvgPool2d(1))
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x).flatten(1)


class MultiModalNet(nn.Module):
    def __init__(self, n_frames: int, n_tab: int, use_image: bool = True, use_tab: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.use_image, self.use_tab = use_image, use_tab
        d = 0
        if use_image:
            self.img = ImageEncoder(3 * n_frames + 3)
            d += self.img.out_dim
        if use_tab:
            self.tab = nn.Sequential(nn.Linear(n_tab, 128), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(128, 64), nn.ReLU())
            d += 64
        self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, img, tab, drop_image: torch.Tensor | None = None):
        z = []
        if self.use_image:
            e = self.img(img)
            if drop_image is not None:
                e = e * (1 - drop_image[:, None])
            z.append(e)
        if self.use_tab:
            z.append(self.tab(tab))
        return torch.sigmoid(self.head(torch.cat(z, 1)).squeeze(1)) * K_MAX


class Batcher:
    """Builds image stacks on the fly from the stream (no duplicated frames in memory)."""

    def __init__(self, images: np.ndarray, s: Samples, tab_cols: list[str], mu=None, sd=None):
        self.images, self.s, self.cols = images, s, tab_cols
        T = s.X[tab_cols].values.astype(np.float32)
        self.mu = np.nanmean(T, 0) if mu is None else mu
        self.sd = (np.nanstd(T, 0) + 1e-6) if sd is None else sd
        self.T = np.nan_to_num((T - self.mu) / self.sd)

    def get(self, idx: np.ndarray, device="cpu"):
        fr = self.images[self.s.frame_idx[idx]]                  # (B, L, 64, 64, 3) uint8
        fr = torch.from_numpy(fr).float().div_(255.0)
        fr = fr.permute(0, 1, 4, 2, 3)                           # (B, L, 3, 64, 64)
        diff = fr[:, -1] - fr[:, -2]
        x = torch.cat([fr.flatten(1, 2), diff], 1)              # (B, 3L+3, 64, 64)
        t = torch.from_numpy(self.T[idx])
        y = torch.from_numpy(self.s.y[idx] / PV_CAPACITY_KW)
        cs = torch.from_numpy(self.s.pv_cs_target[idx] / PV_CAPACITY_KW)
        return x.to(device), t.to(device), y.to(device), cs.to(device)


class NetForecaster:
    def __init__(self, tab_cols: list[str], use_image=True, use_tab=True, n_frames=4,
                 epochs=10, lr=2e-3, batch=128, modality_dropout=0.1, seed=0, device="cpu"):
        self.tab_cols, self.n_frames = tab_cols, n_frames
        self.cfg = dict(use_image=use_image, use_tab=use_tab)
        self.epochs, self.lr, self.batch, self.mdrop, self.seed, self.device = \
            epochs, lr, batch, modality_dropout if (use_image and use_tab) else 0.0, seed, device
        torch.manual_seed(seed)
        self.net = MultiModalNet(n_frames, len(tab_cols), **self.cfg).to(device)
        self.mu = self.sd = None

    def fit(self, images, train: Samples, val: Samples | None = None, verbose=True) -> "NetForecaster":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        bt = Batcher(images, train, self.tab_cols)
        self.mu, self.sd = bt.mu, bt.sd
        bv = Batcher(images, val, self.tab_cols, self.mu, self.sd) if val is not None else None
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * ((len(train) + self.batch - 1) // self.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=steps, pct_start=0.2)
        best, best_state = np.inf, None
        for ep in range(self.epochs):
            self.net.train()
            t0, tot = time.time(), 0.0
            perm = rng.permutation(len(train))
            for i0 in range(0, len(train), self.batch):
                idx = np.sort(perm[i0:i0 + self.batch])          # sorted -> faster gather
                x, t, y, cs = bt.get(idx, self.device)
                x = x * (1 + 0.1 * torch.randn(len(idx), 1, 1, 1, device=self.device))   # exposure jitter
                drop = (torch.rand(len(idx), device=self.device) < self.mdrop).float() if self.mdrop else None
                k = self.net(x, t, drop)
                loss = F.mse_loss(k * cs, y)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item() * len(idx)
            msg = f"  epoch {ep + 1}/{self.epochs} train_rmse={np.sqrt(tot / len(train)) * PV_CAPACITY_KW:.3f} kW"
            if bv is not None:
                r = float(np.sqrt(np.mean((self.predict(images, val, bv) - val.y) ** 2)))
                msg += f" val_rmse={r:.3f} kW"
                if r < best:
                    best, best_state = r, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            if verbose:
                print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, images, s: Samples, batcher: Batcher | None = None, drop_image: bool = False) -> np.ndarray:
        self.net.eval()
        b = batcher or Batcher(images, s, self.tab_cols, self.mu, self.sd)
        out = []
        for i0 in range(0, len(s), 512):
            idx = np.arange(i0, min(i0 + 512, len(s)))
            x, t, _, cs = b.get(idx, self.device)
            d = torch.ones(len(idx), device=self.device) if drop_image and self.cfg["use_image"] else None
            out.append((self.net(x, t, d) * cs).cpu().numpy())
        return np.concatenate(out) * PV_CAPACITY_KW

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "cfg": self.cfg, "tab_cols": self.tab_cols,
                    "n_frames": self.n_frames, "mu": self.mu, "sd": self.sd}, path)

    @staticmethod
    def load(path, device="cpu") -> "NetForecaster":
        d = torch.load(path, map_location=device, weights_only=False)
        m = NetForecaster(d["tab_cols"], n_frames=d["n_frames"], device=device, **d["cfg"])
        m.net.load_state_dict(d["state"])
        m.mu, m.sd = d["mu"], d["sd"]
        return m
