#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "torch",
#   "pillow",
# ]
# ///


import argparse
import math
import time

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise SystemExit("This script requires PyTorch. Install with `pip install torch`.") from e

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("This script requires Pillow. Install with `pip install pillow`.") from e


# ------------------------------------------------------------
# Camera + scene setup
# ------------------------------------------------------------

@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_deg: float


def normalize(x: np.ndarray, axis=-1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > eps)


def look_at(camera: Camera):
    forward = normalize(camera.target - camera.eye)
    right = normalize(np.cross(forward, camera.up))
    true_up = normalize(np.cross(right, forward))
    return forward, right, true_up


def make_rays(width: int, height: int, camera: Camera):
    forward, right, up = look_at(camera)
    aspect = width / height
    fov = math.radians(camera.fov_y_deg)
    half_h = math.tan(fov / 2.0)
    half_w = aspect * half_h

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ndc_x = ((xs + 0.5) / width) * 2.0 - 1.0
    ndc_y = 1.0 - ((ys + 0.5) / height) * 2.0

    dirs = (
        forward[None, None, :]
        + ndc_x[..., None] * half_w * right[None, None, :]
        + ndc_y[..., None] * half_h * up[None, None, :]
    )
    dirs = normalize(dirs)
    origins = np.broadcast_to(camera.eye[None, None, :], dirs.shape)
    return origins, dirs


# ------------------------------------------------------------
# Intersections
# ------------------------------------------------------------

def intersect_sphere(origins, dirs, center, radius):
    oc = origins - center[None, None, :]
    b = np.sum(oc * dirs, axis=-1)
    c = np.sum(oc * oc, axis=-1) - radius * radius
    disc = b * b - c

    real = disc >= 0.0
    t = np.full(disc.shape, np.inf, dtype=np.float32)

    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[real] = np.sqrt(disc[real])

    t0 = -b - sqrt_disc
    t1 = -b + sqrt_disc

    valid0 = real & (t0 > 1e-4)
    valid1 = real & (t1 > 1e-4)

    t[valid0] = t0[valid0]
    fill = (~valid0) & valid1
    t[fill] = t1[fill]

    hit = np.isfinite(t)

    p = np.zeros_like(origins)
    p[hit] = origins[hit] + t[hit, None] * dirs[hit]

    n = np.zeros_like(origins)
    n[hit] = normalize(p[hit] - center[None, :])

    return hit, t, p, n

def intersect_plane(origins, dirs, plane_y=0.0):
    denom = dirs[..., 1]
    t = np.full(denom.shape, np.inf, dtype=np.float32)

    valid = np.abs(denom) > 1e-6
    t_candidate = np.zeros_like(denom)
    t_candidate[valid] = (plane_y - origins[..., 1][valid]) / denom[valid]

    hit = valid & (t_candidate > 1e-4)
    t[hit] = t_candidate[hit]

    p = np.zeros_like(origins)
    p[hit] = origins[hit] + t[hit, None] * dirs[hit]

    n = np.zeros_like(origins)
    n[..., 1] = 1.0
    n[~hit] = 0.0

    return hit, t, p, n


# ------------------------------------------------------------
# G-buffer generation
# ------------------------------------------------------------

def soft_shadow_for_plane(points, light_dir, sphere_center, sphere_radius):
    """
    Cheap analytic-ish shadow hint:
    cast from plane point toward light and see if it hits sphere.
    """
    shadow_origins = points + 1e-3 * light_dir[None, None, :]
    shadow_dirs = np.broadcast_to(light_dir[None, None, :], points.shape)
    hit, _, _, _ = intersect_sphere(shadow_origins, shadow_dirs, sphere_center, sphere_radius)
    return (~hit).astype(np.float32)


def build_scene(width=128, height=128, camera=None):
    if camera is None:
        camera = Camera(
            eye=np.array([2.8, 1.7, 2.8], dtype=np.float32),
            target=np.array([0.0, 0.7, 0.0], dtype=np.float32),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            fov_y_deg=50.0,
        )

    sphere_center = np.array([0.0, 0.8, 0.0], dtype=np.float32)
    sphere_radius = 0.8
    light_dir = normalize(np.array([0.6, 1.0, 0.35], dtype=np.float32))
    light_dir = light_dir.astype(np.float32)

    origins, dirs = make_rays(width, height, camera)

    s_hit, s_t, s_p, s_n = intersect_sphere(origins, dirs, sphere_center, sphere_radius)
    p_hit, p_t, p_p, p_n = intersect_plane(origins, dirs, plane_y=0.0)

    use_sphere = s_hit & (s_t < p_t)
    use_plane = p_hit & (~use_sphere)
    hit_any = use_sphere | use_plane

    pos = np.zeros((height, width, 3), dtype=np.float32)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.zeros((height, width), dtype=np.float32)
    obj_mask = np.zeros((height, width, 2), dtype=np.float32)  # [sphere, plane]

    pos[use_sphere] = s_p[use_sphere]
    pos[use_plane] = p_p[use_plane]
    normal[use_sphere] = s_n[use_sphere]
    normal[use_plane] = p_n[use_plane]
    depth[hit_any] = np.where(use_sphere, s_t, p_t)[hit_any]
    obj_mask[..., 0] = use_sphere.astype(np.float32)
    obj_mask[..., 1] = use_plane.astype(np.float32)

    # View direction at hit point
    view_dir = np.zeros_like(pos)
    view_vec = np.zeros_like(pos)
    view_vec[hit_any] = camera.eye[None, :] - pos[hit_any]
    view_dir = normalize(view_vec)

    # Baseline Lambert + plane shadow
    lambert = np.maximum(0.0, np.sum(normal * light_dir[None, None, :], axis=-1))
    shadow = np.ones_like(lambert)
    if np.any(use_plane):
        shadow_plane = soft_shadow_for_plane(p_p, light_dir, sphere_center, sphere_radius)
        shadow[use_plane] = shadow_plane[use_plane]
    baseline_light = lambert * shadow

    sphere_albedo = np.array([0.82, 0.35, 0.22], dtype=np.float32)
    plane_albedo = np.array([0.72, 0.75, 0.78], dtype=np.float32)
    albedo = (
        obj_mask[..., 0:1] * sphere_albedo[None, None, :]
        + obj_mask[..., 1:2] * plane_albedo[None, None, :]
    )

    ambient = 0.10
    baseline_rgb = albedo * (ambient + 0.90 * baseline_light[..., None])
    background = np.array([0.92, 0.95, 1.0], dtype=np.float32)
    baseline_rgb[~hit_any] = background

    # Simple screen UV
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    uv = np.stack([(xs + 0.5) / width, (ys + 0.5) / height], axis=-1)

    # Depth normalization
    if np.any(hit_any):
        d = depth[hit_any]
        depth_norm = np.zeros_like(depth)
        depth_norm[hit_any] = (d - d.min()) / max(1e-6, d.max() - d.min())
    else:
        depth_norm = depth.copy()

    # Deliberately non-physical stylised target
    target_rgb = stylised_target(
        pos=pos,
        normal=normal,
        obj_mask=obj_mask,
        view_dir=view_dir,
        baseline_rgb=baseline_rgb,
        baseline_light=baseline_light,
        uv=uv,
        hit_any=hit_any,
        sphere_center=sphere_center,
    )

    features = np.concatenate(
        [
            normal,                         # 3
            depth_norm[..., None],          # 1
            obj_mask,                       # 2
            hit_any[..., None],             # 1
            baseline_light[..., None],      # 1
            uv,                             # 2
            view_dir,                       # 3
            pos,                            # 3
        ],
        axis=-1,
    )

    # background channels sane
    features[~hit_any, :] = 0.0
    #features[..., 7:9] = uv   # restore UV everywhere if you want UV on sky

    return {
        "features": features.astype(np.float32),
        "baseline_rgb": baseline_rgb.astype(np.float32),
        "target_rgb": target_rgb.astype(np.float32),
        "hit_any": hit_any.astype(np.float32),
        "depth": depth_norm.astype(np.float32),
        "normal": normal.astype(np.float32),
    }


def stylised_target(pos, normal, obj_mask, view_dir, baseline_rgb, baseline_light, uv, hit_any, sphere_center):
    sphere = obj_mask[..., 0] > 0.5
    plane = obj_mask[..., 1] > 0.5

    rgb = baseline_rgb.copy()

    # Purple/blue shadow tint
    shadow_amt = np.clip(1.0 - baseline_light, 0.0, 1.0)
    rgb += shadow_amt[..., None] * np.array([0.10, 0.02, 0.18], dtype=np.float32)

    # Warm rim light on sphere
    ndotv = np.clip(np.sum(normal * view_dir, axis=-1), 0.0, 1.0)
    rim = np.power(1.0 - ndotv, 2.5) * sphere.astype(np.float32)
    rgb += rim[..., None] * np.array([0.95, 0.65, 0.22], dtype=np.float32) * 0.45

    # Fake contact glow under sphere on plane
    rel = pos - sphere_center[None, None, :]
    radial_xz = np.sqrt(rel[..., 0] ** 2 + rel[..., 2] ** 2)
    contact_ring = np.exp(-((radial_xz - 0.78) ** 2) / 0.015) * np.exp(-(pos[..., 1] ** 2) / 0.01)
    contact_ring *= plane.astype(np.float32)
    rgb += contact_ring[..., None] * np.array([0.2, 0.45, 0.95], dtype=np.float32) * 0.55

    # Fake bounce light from plane to bottom of sphere
    downness = np.clip(-normal[..., 1], 0.0, 1.0) * sphere.astype(np.float32)
    rgb += downness[..., None] * np.array([0.10, 0.22, 0.38], dtype=np.float32) * 0.35

    # Slight top-to-bottom grade across screen
    grade = (1.0 - uv[..., 1])[..., None]
    rgb *= 0.92 + 0.16 * grade

    # Background tinted
    bg = ~hit_any
    rgb[bg] = np.array([0.88, 0.92, 1.0], dtype=np.float32) + (1.0 - uv[bg, 1:2]) * np.array([0.04, 0.02, 0.0], dtype=np.float32)

    return np.clip(rgb, 0.0, 1.0)


# ------------------------------------------------------------
# Tiny neural shader
# ------------------------------------------------------------

class TinyNeuralShader(nn.Module):
    def __init__(self, in_dim=12, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class TinyCNNShader(nn.Module):
    def __init__(self, in_ch=12, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)
    

class PartialConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # ones kernel for mask normalization
        self.register_buffer(
            "mask_kernel",
            torch.ones(1, 1, kernel_size, kernel_size)
        )

    def forward(self, x, mask):
        # x:    [B, C, H, W]
        # mask: [B, 1, H, W], values 0/1

        x_masked = x * mask
        out = self.conv(x_masked)

        with torch.no_grad():
            mask_sum = F.conv2d(
                mask,
                self.mask_kernel,
                stride=self.stride,
                padding=self.padding,
            )

        # avoid divide-by-zero
        mask_sum_clamped = torch.clamp(mask_sum, min=1.0)

        # renormalize as if only valid inputs contributed
        out = out * (self.kernel_size * self.kernel_size / mask_sum_clamped)

        # zero where there was no valid support
        new_mask = (mask_sum > 0).float()
        out = out * new_mask

        return out, new_mask
    
class TinyPartialCNNShader(nn.Module):
    def __init__(self, in_ch=15, hidden=32):
        super().__init__()
        self.pconv1 = PartialConv2d(in_ch, hidden, kernel_size=3, padding=1)
        self.pconv2 = PartialConv2d(hidden, hidden, kernel_size=3, padding=1)
        self.pconv3 = PartialConv2d(hidden, hidden, kernel_size=3, padding=1)
        self.pconv4 = PartialConv2d(hidden, hidden, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(hidden, 3, kernel_size=1)

    def forward(self, x, mask):
        x, mask = self.pconv1(x, mask)
        x = F.gelu(x)

        x, mask = self.pconv2(x, mask)
        x = F.gelu(x)

        x, mask = self.pconv3(x, mask)
        x = F.gelu(x)

        x, mask = self.pconv4(x, mask)
        x = F.gelu(x)

        x = self.out_conv(x)
        x = torch.tanh(x)
        return x, mask
# ------------------------------------------------------------
# Training / rendering
# ------------------------------------------------------------

def save_image(path: Path, img: np.ndarray):
    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img8).save(path)


def save_triptych(path: Path, left: np.ndarray, middle: np.ndarray, right: np.ndarray):
    canvas = np.concatenate([left, middle, right], axis=1)
    save_image(path, canvas)


def train_on_single_view(features, target_rgb, steps=2000, lr=1e-3, device="cpu", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    H, W, C = features.shape
    x = torch.from_numpy(features.reshape(-1, C)).to(device)
    y = torch.from_numpy(target_rgb.reshape(-1, 3)).to(device)

    model = TinyNeuralShader(in_dim=C, hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"step {step:5d} | loss {loss.item():.6f}")

    with torch.no_grad():
        pred = model(x).reshape(H, W, 3).cpu().numpy()
    return model, pred

def train_cnn_on_single_view(
    features,
    baseline_rgb,
    target_rgb,
    steps=2000,
    lr=1e-3,
    device="cpu",
    seed=0,
    residual_scale=0.5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).to(device)         # 1,C,H,W
    baseline = torch.from_numpy(baseline_rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # 1,3,H,W
    y = torch.from_numpy(target_rgb).permute(2, 0, 1).unsqueeze(0).to(device)       # 1,3,H,W

    model = TinyCNNShader(in_ch=features.shape[-1], hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        delta = model(x)  # should be roughly in [-1, 1]
        pred = torch.clamp(baseline + residual_scale * delta, 0.0, 1.0)

        loss = torch.mean((pred - y) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"step {step:5d} | loss {loss.item():.6f}")

    with torch.no_grad():
        delta = model(x)
        pred = torch.clamp(baseline + residual_scale * delta, 0.0, 1.0)
        pred = pred[0].permute(1, 2, 0).cpu().numpy()

    return model, pred

def make_dataset(
    n_views=64,
    width=128,
    height=128,
    theta_min_deg=0.0,
    theta_max_deg=180.0,
    radius_range=(3.6, 4.4),
    height_range=(1.5, 2.1),
    seed=0,
):
    rng = np.random.default_rng(seed)

    features_list = []
    baseline_list = []
    target_list = []
    hit_list = []
    meta = []

    for _ in range(n_views):
        theta = rng.uniform(theta_min_deg, theta_max_deg)
        radius = rng.uniform(*radius_range)
        if rng.random() < 0.4:
            cam_height = rng.uniform(2.0, height_range[1])
        else:
            cam_height = rng.uniform(*height_range)

        cam = orbit_camera(theta_deg=theta, radius=radius, height=cam_height)
        scene = build_scene(width=width, height=height, camera=cam)

        features_list.append(scene["features"])
        baseline_list.append(scene["baseline_rgb"])
        target_list.append(scene["target_rgb"])
        hit_list.append(scene["hit_any"])
        meta.append(
            {
                "theta_deg": theta,
                "radius": radius,
                "height": cam_height,
            }
        )

    features = np.stack(features_list, axis=0)      # N,H,W,C
    baseline = np.stack(baseline_list, axis=0)      # N,H,W,3
    target = np.stack(target_list, axis=0)          # N,H,W,3
    hit_any = np.stack(hit_list, axis=0)            # N,H,W

    return {
        "features": features.astype(np.float32),
        "baseline_rgb": baseline.astype(np.float32),
        "target_rgb": target.astype(np.float32),
        "hit_any": hit_any.astype(np.float32),
        "meta": meta,
    }

def erode_hit_mask(hit_mask, radius=1):
    k = 2 * radius + 1
    return (F.max_pool2d(1.0 - hit_mask, kernel_size=k, stride=1, padding=radius) < 0.5).float()

def train_cnn_multiview(
    features,
    baseline_rgb,
    target_rgb,
    hit_any,
    steps=4000,
    batch_size=8,
    lr=1e-3,
    device="cpu",
    seed=0,
    residual_scale=0.5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.from_numpy(features).permute(0, 3, 1, 2).to(device)         # N,C,H,W
    baseline = torch.from_numpy(baseline_rgb).permute(0, 3, 1, 2).to(device)
    y = torch.from_numpy(target_rgb).permute(0, 3, 1, 2).to(device)
    hit = torch.from_numpy(hit_any).unsqueeze(1).to(device)                # N,1,H,W

    n = x.shape[0]

    model = TinyCNNShader(in_ch=features.shape[-1], hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rng = np.random.default_rng(seed)

    for step in range(steps):
        idx = rng.integers(0, n, size=batch_size)
        xb = x[idx]
        bb = baseline[idx]
        yb = y[idx]
        hb = hit[idx]

        delta = model(xb)
        hb_safe = erode_hit_mask(hb, radius=0)
        pred = torch.clamp(bb + residual_scale * delta * hb_safe, 0.0, 1.0)

        loss_map = (pred - yb) ** 2
        loss = (loss_map * hb_safe).sum() / (3.0 * hb_safe.sum().clamp_min(1.0))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            with torch.no_grad():
                full_delta = model(x[: min(n, batch_size)])
                full_pred = torch.clamp(
                    baseline[: min(n, batch_size)] + residual_scale * full_delta * hit[: min(n, batch_size)],
                    0.0,
                    1.0,
                )
                preview_loss = torch.mean((full_pred - y[: min(n, batch_size)]) ** 2).item()
                delta_mag = torch.mean(torch.abs(delta)).item()

            print(
                f"step {step:5d} | batch_loss {loss.item():.6f} "
                f"| preview_loss {preview_loss:.6f} | |delta| {delta_mag:.4f}"
            )

    return model

def train_partial_cnn_multiview(
    features,
    baseline_rgb,
    target_rgb,
    hit_any,
    steps=4000,
    batch_size=8,
    lr=1e-3,
    device="cpu",
    seed=0,
    residual_scale=0.5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.from_numpy(features).permute(0, 3, 1, 2).to(device)         # N,C,H,W
    baseline = torch.from_numpy(baseline_rgb).permute(0, 3, 1, 2).to(device)
    y = torch.from_numpy(target_rgb).permute(0, 3, 1, 2).to(device)
    hit = torch.from_numpy(hit_any).unsqueeze(1).to(device)                # N,1,H,W

    n = x.shape[0]

    model = TinyPartialCNNShader(in_ch=features.shape[-1], hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rng = np.random.default_rng(seed)

    for step in range(steps):
        idx = rng.integers(0, n, size=batch_size)
        xb = x[idx]
        bb = baseline[idx]
        yb = y[idx]
        hb = hit[idx]

        delta, out_mask = model(xb, hb)
        pred = torch.clamp(bb + residual_scale * delta * hb, 0.0, 1.0)

        loss_map = (pred - yb) ** 2
        loss = (loss_map * hb).sum() / (3.0 * hb.sum().clamp_min(1.0))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            with torch.no_grad():
                preview_n = min(n, batch_size)
                px = x[:preview_n]
                pb = baseline[:preview_n]
                py = y[:preview_n]
                ph = hit[:preview_n]

                pdelta, _ = model(px, ph)
                ppred = torch.clamp(pb + residual_scale * pdelta * ph, 0.0, 1.0)

                preview_loss_map = (ppred - py) ** 2
                preview_loss = (
                    (preview_loss_map * ph).sum() / (3.0 * ph.sum().clamp_min(1.0))
                ).item()
                delta_mag = torch.mean(torch.abs(delta)).item()

            print(
                f"step {step:5d} | batch_loss {loss.item():.6f} "
                f"| preview_loss {preview_loss:.6f} | |delta| {delta_mag:.4f}"
            )

    return model

def render_with_model(model, features, device="cpu"):
    H, W, C = features.shape
    x = torch.from_numpy(features.reshape(-1, C)).to(device)
    with torch.no_grad():
        pred = model(x).reshape(H, W, 3).cpu().numpy()
    return pred

def render_with_cnn(model, features, baseline_rgb, hit_any, device="cpu", residual_scale=0.5):
    x = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).to(device)
    baseline = torch.from_numpy(baseline_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    hit = torch.from_numpy(hit_any).unsqueeze(0).unsqueeze(0).to(device)
    hit_safe = erode_hit_mask(hit, radius=0)

    with torch.no_grad():
        delta = model(x)
        pred = torch.clamp(baseline + residual_scale * delta * hit_safe, 0.0, 1.0)

    return pred[0].permute(1, 2, 0).cpu().numpy()

def render_with_partial_cnn(model, features, baseline_rgb, hit_any, device="cpu", residual_scale=0.5):
    x = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).to(device)
    baseline = torch.from_numpy(baseline_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    hit = torch.from_numpy(hit_any).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        delta, _ = model(x, hit)
        pred = torch.clamp(baseline + residual_scale * delta * hit, 0.0, 1.0)

    return pred[0].permute(1, 2, 0).cpu().numpy()

def orbit_camera(theta_deg: float, radius=4.0, height=1.8):
    theta = math.radians(theta_deg)
    eye = np.array([radius * math.cos(theta), height, radius * math.sin(theta)], dtype=np.float32)
    return Camera(
        eye=eye,
        target=np.array([0.0, 0.65, 0.0], dtype=np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        fov_y_deg=50.0,
    )

def save_dataset_examples(model, dataset, outdir, prefix, device="cpu", residual_scale=0.5, count=6):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = dataset["features"].shape[0]
    chosen = np.linspace(0, n - 1, min(count, n), dtype=int)

    for i in chosen:
        feat = dataset["features"][i]
        base = dataset["baseline_rgb"][i]
        tgt = dataset["target_rgb"][i]
        hit = dataset["hit_any"][i]
        pred = render_with_partial_cnn(model, feat, base, hit, device=device, residual_scale=residual_scale)

        save_triptych(outdir / f"{prefix}_{i:03d}.png", base, tgt, pred)

def render_gif_frames(
    model,
    width,
    height,
    n_frames=48,
    radius=4.0,
    height_base=1.8,
    height_amp=0.5,
    theta_start_deg=0.0,
    theta_end_deg=360.0,
    device="cpu",
    residual_scale=0.5,
):
    baseline_frames = []
    target_frames = []
    pred_frames = []

    thetas = np.linspace(theta_start_deg, theta_end_deg, n_frames, endpoint=False)

    for theta in thetas:
        h = height_base + height_amp * math.sin(math.radians(theta))
        cam = orbit_camera(theta_deg=float(theta), radius=radius, height=h)
        scene = build_scene(width=width, height=height, camera=cam)

        pred = render_with_partial_cnn(
            model,
            scene["features"],
            scene["baseline_rgb"],
            scene["hit_any"],
            device=device,
            residual_scale=residual_scale,
        )

        baseline_frames.append(scene["baseline_rgb"])
        target_frames.append(scene["target_rgb"])
        pred_frames.append(pred)

    return baseline_frames, target_frames, pred_frames

def save_gif(path: Path, frames, duration_ms=60, loop=0):
    pil_frames = [
        Image.fromarray(np.clip(f * 255.0, 0, 255).astype(np.uint8))
        for f in frames
    ]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
    )

def concat_frames_horiz(frames_a, frames_b):
    return [np.concatenate([a, b], axis=1) for a, b in zip(frames_a, frames_b)]   


# ------------------------------------------------------------
# Temporal reprojection validation helpers
# ------------------------------------------------------------

def extract_delta_cache(
    delta_rgb: np.ndarray,
    world_pos: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    """
    Turn a per pixel residual image into a world anchored point cache.

    Parameters
    ----------
    delta_rgb : (H, W, 3) float32
        Residual in RGB space, usually target_rgb - baseline_rgb for frame 0.
    world_pos : (H, W, 3) float32
        Per pixel world space hit positions.
    valid_mask : (H, W) bool or float
        True where geometry was hit.

    Returns
    -------
    cache : dict
        {
            "points_world": (N, 3),
            "delta_rgb": (N, 3),
        }
    """
    valid = valid_mask > 0.5
    return {
        "points_world": world_pos[valid].astype(np.float32),
        "delta_rgb": delta_rgb[valid].astype(np.float32),
    }


def world_to_camera(points_world: np.ndarray, camera: Camera) -> np.ndarray:
    """
    Transform world points to camera coordinates.
    Camera forward is +z in this convention.
    """
    forward, right, up = look_at(camera)
    rel = points_world - camera.eye[None, :]
    x_cam = rel @ right
    y_cam = rel @ up
    z_cam = rel @ forward
    return np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)


def project_world_points(
    points_world: np.ndarray,
    camera: Camera,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project world points into pixel coordinates for the current camera.

    Returns
    -------
    xi : (N,) int32
        Pixel x indices.
    yi : (N,) int32
        Pixel y indices.
    z_cam : (N,) float32
        Positive distance along camera forward.
    in_bounds : (N,) bool
        True where point is in front of the camera and on screen.
    """
    cam_pts = world_to_camera(points_world, camera)
    x_cam = cam_pts[:, 0]
    y_cam = cam_pts[:, 1]
    z_cam = cam_pts[:, 2]

    aspect = width / float(height)
    fov = math.radians(camera.fov_y_deg)
    half_h = math.tan(fov / 2.0)
    half_w = aspect * half_h

    eps = 1e-6
    in_front = z_cam > eps

    ndc_x = np.full_like(z_cam, np.nan, dtype=np.float32)
    ndc_y = np.full_like(z_cam, np.nan, dtype=np.float32)

    ndc_x[in_front] = x_cam[in_front] / (z_cam[in_front] * half_w)
    ndc_y[in_front] = y_cam[in_front] / (z_cam[in_front] * half_h)

    # Inverse of make_rays():
    # ndc_x = ((x + 0.5) / width) * 2 - 1
    # ndc_y = 1 - ((y + 0.5) / height) * 2
    x_pix = ((ndc_x + 1.0) * 0.5) * width - 0.5
    y_pix = ((1.0 - ndc_y) * 0.5) * height - 0.5

    xi = np.rint(x_pix).astype(np.int32)
    yi = np.rint(y_pix).astype(np.int32)

    on_screen = (
        in_front
        & np.isfinite(x_pix)
        & np.isfinite(y_pix)
        & (xi >= 0)
        & (xi < width)
        & (yi >= 0)
        & (yi < height)
    )

    return xi, yi, z_cam.astype(np.float32), on_screen


def rasterize_reprojected_delta(
    points_world: np.ndarray,
    delta_rgb: np.ndarray,
    camera: Camera,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproject cached world points into the new camera and z buffer them.

    Returns
    -------
    delta_img : (H, W, 3) float32
        Reprojected delta at winner pixels.
    zbuf : (H, W) float32
        Camera space z of the winning reprojection.
    valid_proj : (H, W) bool
        True where at least one cached point projects and wins.
    """
    delta_img = np.zeros((height, width, 3), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    valid_proj = np.zeros((height, width), dtype=bool)

    xi, yi, z_cam, on_screen = project_world_points(
        points_world=points_world,
        camera=camera,
        width=width,
        height=height,
    )

    idx = np.nonzero(on_screen)[0]
    if idx.size == 0:
        return delta_img, zbuf, valid_proj

    # Simple z buffer splat. Closest point wins.
    for i in idx:
        x = xi[i]
        y = yi[i]
        z = z_cam[i]
        if z < zbuf[y, x]:
            zbuf[y, x] = z
            delta_img[y, x] = delta_rgb[i]
            valid_proj[y, x] = True

    return delta_img, zbuf, valid_proj


def compute_current_camera_depth_from_world_pos(
    world_pos: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    """
    Convert per pixel world positions for the current frame into camera space depth.
    Invalid or background pixels will usually be zero in world_pos, so you should mask them.
    """
    h, w, _ = world_pos.shape
    cam_pts = world_to_camera(world_pos.reshape(-1, 3), camera).reshape(h, w, 3)
    return cam_pts[..., 2].astype(np.float32)


def compute_geom_validity(
    reproj_zbuf: np.ndarray,
    current_world_pos: np.ndarray,
    current_hit_mask: np.ndarray,
    camera: Camera,
    depth_eps: float = 0.05,
) -> np.ndarray:
    """
    Reject reprojections that do not agree with the current geometry.

    The comparison is done in camera space z.
    """
    current_hit = current_hit_mask > 0.5
    current_z = compute_current_camera_depth_from_world_pos(current_world_pos, camera)

    valid_geom = np.zeros_like(current_hit, dtype=bool)
    valid_geom[current_hit] = np.abs(reproj_zbuf[current_hit] - current_z[current_hit]) <= depth_eps
    return valid_geom


def compose_reprojected_frame(
    baseline_rgb: np.ndarray,
    delta_reproj: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply the reprojected delta only where valid.
    """
    out = baseline_rgb.copy()
    valid = valid_mask > 0.5
    out[valid] = np.clip(baseline_rgb[valid] + delta_reproj[valid], 0.0, 1.0)
    return out.astype(np.float32)


def dilate_mask_3x3(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Tiny numpy only binary dilation for filling one pixel holes.
    """
    out = mask.astype(bool).copy()
    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        nbrs = []
        for dy in range(3):
            for dx in range(3):
                nbrs.append(padded[dy:dy + out.shape[0], dx:dx + out.shape[1]])
        out = np.logical_or.reduce(nbrs)
    return out


def fill_holes_from_neighbours(
    delta_img: np.ndarray,
    valid_mask: np.ndarray,
    iterations: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optional cheap post pass to reduce single pixel holes in the reprojection.

    For each invalid pixel adjacent to valid pixels, fill it with the mean of valid
    8 connected neighbours.
    """
    delta = delta_img.copy()
    valid = valid_mask.astype(bool).copy()

    h, w, c = delta.shape
    for _ in range(iterations):
        new_delta = delta.copy()
        new_valid = valid.copy()

        for y in range(h):
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            for x in range(w):
                if valid[y, x]:
                    continue
                x0 = max(0, x - 1)
                x1 = min(w, x + 2)

                nbr_valid = valid[y0:y1, x0:x1]
                if not np.any(nbr_valid):
                    continue

                nbr_delta = delta[y0:y1, x0:x1]
                vals = nbr_delta[nbr_valid]
                new_delta[y, x] = np.mean(vals, axis=0)
                new_valid[y, x] = True

        delta = new_delta
        valid = new_valid

    return delta.astype(np.float32), valid


def render_reprojected_delta_sequence(
    width: int,
    height: int,
    n_frames: int = 64,
    radius: float = 4.0,
    height_base: float = 1.8,
    height_amp: float = 0.6,
    theta_start_deg: float = 0.0,
    theta_end_deg: float = 360.0,
    depth_eps: float = 0.05,
    hole_fill_iters: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
    """
    Validation sequence:
    1. Use frame 0 residual only.
    2. Reproject that residual to every future camera.
    3. Compose it with the current baseline.
    4. Return baseline, target, reprojection result, validity visualisation, and coverage visualisation.
    """
    thetas = np.linspace(theta_start_deg, theta_end_deg, n_frames, endpoint=False)

    # Frame 0 cache
    theta0 = float(thetas[0])
    h0 = height_base + height_amp * math.sin(math.radians(theta0))
    cam0 = orbit_camera(theta_deg=theta0, radius=radius, height=h0)
    scene0 = build_scene(width=width, height=height, camera=cam0)

    # In this repo the last 3 feature channels are world position.
    world_pos0 = scene0["features"][..., -3:]
    valid0 = scene0["hit_any"] > 0.5
    delta0 = scene0["target_rgb"] - scene0["baseline_rgb"]

    cache = extract_delta_cache(
        delta_rgb=delta0,
        world_pos=world_pos0,
        valid_mask=valid0,
    )

    baseline_frames = []
    target_frames = []
    reproj_frames = []
    valid_vis_frames = []
    coverage_vis_frames = []
    reproj_ms_list = []

    for theta in thetas:
        h = height_base + height_amp * math.sin(math.radians(theta))
        cam = orbit_camera(theta_deg=float(theta), radius=radius, height=h)
        scene = build_scene(width=width, height=height, camera=cam)

        current_world_pos = scene["features"][..., -3:]
        current_hit = scene["hit_any"] > 0.5

        t0 = time.perf_counter()

        delta_reproj, reproj_zbuf, valid_proj = rasterize_reprojected_delta(
            points_world=cache["points_world"],
            delta_rgb=cache["delta_rgb"],
            camera=cam,
            width=width,
            height=height,
        )

        valid_geom = compute_geom_validity(
            reproj_zbuf=reproj_zbuf,
            current_world_pos=current_world_pos,
            current_hit_mask=current_hit,
            camera=cam,
            depth_eps=depth_eps,
        )

        valid_reproj = valid_proj & valid_geom

        if hole_fill_iters > 0:
            delta_reproj_masked = np.zeros_like(delta_reproj)
            delta_reproj_masked[valid_reproj] = delta_reproj[valid_reproj]
            delta_reproj_filled, valid_reproj_filled = fill_holes_from_neighbours(
                delta_reproj_masked,
                valid_reproj,
                iterations=hole_fill_iters,
            )
            delta_reproj = delta_reproj_filled
            valid_reproj = valid_reproj_filled & current_hit

        reproj_img = compose_reprojected_frame(
            baseline_rgb=scene["baseline_rgb"],
            delta_reproj=delta_reproj,
            valid_mask=valid_reproj,
        )

        t1 = time.perf_counter()
        reproj_ms_list.append((t1 - t0) * 1000.0)

        coverage = np.zeros((height, width, 3), dtype=np.float32)
        coverage[..., 1] = valid_reproj.astype(np.float32)  # green valid
        coverage[..., 0] = (current_hit & ~valid_reproj).astype(np.float32)  # red uncovered geometry

        valid_vis = np.zeros((height, width, 3), dtype=np.float32)
        valid_vis[..., 1] = valid_proj.astype(np.float32)
        valid_vis[..., 2] = valid_geom.astype(np.float32)

        baseline_frames.append(scene["baseline_rgb"])
        target_frames.append(scene["target_rgb"])
        reproj_frames.append(reproj_img)
        valid_vis_frames.append(valid_vis)
        coverage_vis_frames.append(coverage)

    return baseline_frames, target_frames, reproj_frames, valid_vis_frames, coverage_vis_frames, reproj_ms_list

def print_timing_stats(name: str, values_ms: list[float]) -> None:
    arr = np.asarray(values_ms, dtype=np.float64)
    print(
        f"{name}: "
        f"mean={arr.mean():.3f} ms, "
        f"median={np.median(arr):.3f} ms, "
        f"min={arr.min():.3f} ms, "
        f"max={arr.max():.3f} ms, "
        f"p95={np.percentile(arr, 95):.3f} ms"

    )

# ------------------------------------------------------------
# Temporal training data generation test
# ------------------------------------------------------------

CASE_REPROJ = 0
CASE_NONE = 1
CASE_CORRUPT = 2

CASE_NAME_TO_ID = {
    "reproj": CASE_REPROJ,
    "none": CASE_NONE,
    "corrupt": CASE_CORRUPT,
}

CASE_ID_TO_NAME = {v: k for k, v in CASE_NAME_TO_ID.items()}


def lerp(a, b, t: float):
    return (1.0 - t) * a + t * b


def make_camera(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
    fov_y_deg: float = 45.0,
) -> Camera:
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return Camera(
        eye=np.asarray(eye, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        up=np.asarray(up, dtype=np.float32),
        fov_y_deg=float(fov_y_deg),
    )


def camera_basis(camera: Camera):
    forward, right, up = look_at(camera)
    return forward.astype(np.float32), right.astype(np.float32), up.astype(np.float32)


def rotate_vec_around_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis.astype(np.float32))
    v = v.astype(np.float32)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (
        v * c
        + np.cross(axis, v) * s
        + axis * np.dot(axis, v) * (1.0 - c)
    ).astype(np.float32)


def perturb_camera_orientation(
    camera: Camera,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> Camera:
    forward, right, up = camera_basis(camera)
    f = forward.copy()
    u = up.copy()

    if abs(yaw_deg) > 1e-9:
        f = rotate_vec_around_axis(f, u, math.radians(yaw_deg))
        right = normalize(np.cross(f, u))
        u = normalize(np.cross(right, f))

    if abs(pitch_deg) > 1e-9:
        right = normalize(np.cross(f, u))
        f = rotate_vec_around_axis(f, right, math.radians(pitch_deg))
        u = normalize(np.cross(right, f))

    if abs(roll_deg) > 1e-9:
        f = normalize(f)
        u = rotate_vec_around_axis(u, f, math.radians(roll_deg))

    return make_camera(
        eye=camera.eye.copy(),
        target=camera.eye + normalize(f),
        up=normalize(u),
        fov_y_deg=camera.fov_y_deg,
    )


def orbit_camera_with_target(
    theta_deg: float,
    radius: float = 4.0,
    height: float = 1.8,
    target: np.ndarray | None = None,
    fov_y_deg: float = 45.0,
) -> Camera:
    if target is None:
        target = np.array([0.0, 0.75, 0.0], dtype=np.float32)
    theta = math.radians(theta_deg)
    eye = np.array(
        [
            radius * math.cos(theta),
            height,
            radius * math.sin(theta),
        ],
        dtype=np.float32,
    )
    return make_camera(eye=eye, target=target, fov_y_deg=fov_y_deg)


def render_scene_frame(width: int, height: int, camera: Camera) -> dict:
    """
    Wrapper around build_scene that pulls out the fields we need for temporal samples.
    Assumes the last 3 feature channels are world position.
    """
    scene = build_scene(width=width, height=height, camera=camera)

    features = scene["features"].astype(np.float32)
    baseline_rgb = scene["baseline_rgb"].astype(np.float32)
    target_rgb = scene["target_rgb"].astype(np.float32)
    hit_any = scene["hit_any"].astype(np.float32)

    world_pos = features[..., -3:].astype(np.float32)
    delta_rgb = (target_rgb - baseline_rgb).astype(np.float32)

    out = {
        "camera": camera,
        "features": features,
        "world_pos": world_pos,
        "baseline_rgb": baseline_rgb,
        "target_rgb": target_rgb,
        "delta_rgb": delta_rgb,
        "hit_any": hit_any,
    }

    if "normal" in scene:
        out["normal"] = scene["normal"].astype(np.float32)

    return out


def masked_zero_like_hw3(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.float32)


def masked_zero_like_hw(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.float32)


def random_binary_keep_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    keep_prob: float,
) -> np.ndarray:
    return (rng.random((height, width)) < keep_prob).astype(np.float32)


def shift_image_integer(
    img: np.ndarray,
    dx: int,
    dy: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    out = np.full_like(img, fill_value)
    h, w = img.shape[:2]

    x_src0 = max(0, -dx)
    x_src1 = min(w, w - dx)
    y_src0 = max(0, -dy)
    y_src1 = min(h, h - dy)

    x_dst0 = max(0, dx)
    x_dst1 = min(w, w + dx)
    y_dst0 = max(0, dy)
    y_dst1 = min(h, h + dy)

    out[y_dst0:y_dst1, x_dst0:x_dst1] = img[y_src0:y_src1, x_src0:x_src1]
    return out


def blur3x3_mean(img: np.ndarray) -> np.ndarray:
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = np.zeros_like(img)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return acc / 9.0


def erode_mask3x3(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = (mask > 0.5)
    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        nbrs = []
        for dy in range(3):
            for dx in range(3):
                nbrs.append(padded[dy:dy + out.shape[0], dx:dx + out.shape[1]])
        out = np.logical_and.reduce(nbrs)
    return out.astype(np.float32)


def corrupt_reprojected_inputs(
    rng: np.random.Generator,
    delta_reproj: np.ndarray,
    valid_reproj: np.ndarray,
    current_hit: np.ndarray,
    noise_std: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrupts a real reprojection so the network cannot simply trust the mask.
    """
    h, w, _ = delta_reproj.shape

    delta = delta_reproj.copy()
    valid = valid_reproj.astype(np.float32).copy()

    # Random integer shift
    if rng.random() < 0.8:
        dx = int(rng.integers(-2, 3))
        dy = int(rng.integers(-2, 3))
        delta = shift_image_integer(delta, dx=dx, dy=dy, fill_value=0.0)
        valid = shift_image_integer(valid[..., None], dx=dx, dy=dy, fill_value=0.0)[..., 0]

    # Random mask dropout
    if rng.random() < 0.9:
        keep_prob = float(rng.uniform(0.75, 0.97))
        keep = random_binary_keep_mask(rng, h, w, keep_prob)
        valid *= keep
        delta *= valid[..., None]

    # Erode mask a little
    if rng.random() < 0.6:
        iters = int(rng.integers(1, 3))
        valid = erode_mask3x3(valid, iterations=iters)
        delta *= valid[..., None]

    # Local blur to smear details slightly
    if rng.random() < 0.5:
        delta = blur3x3_mean(delta)

    # Add bounded noise where still valid
    if rng.random() < 0.85:
        noise = rng.normal(0.0, noise_std, size=delta.shape).astype(np.float32)
        delta = delta + noise * valid[..., None]

    # Optionally invert a small fraction of valid pixels into invalid and vice versa
    if rng.random() < 0.4:
        flip = random_binary_keep_mask(rng, h, w, keep_prob=float(rng.uniform(0.97, 0.995)))
        valid = np.where(flip > 0.5, 1.0 - valid, valid).astype(np.float32)

    # Do not let corrupted mask extend outside currently hit geometry too much
    valid *= (current_hit > 0.5).astype(np.float32)
    delta *= valid[..., None]

    delta = np.clip(delta, -1.0, 1.0).astype(np.float32)
    return delta, valid.astype(np.float32)


def choose_temporal_case(
    rng: np.random.Generator,
    p_reproj: float,
    p_none: float,
    p_corrupt: float,
) -> str:
    probs = np.array([p_reproj, p_none, p_corrupt], dtype=np.float64)
    probs = probs / probs.sum()
    case_id = int(rng.choice(3, p=probs))
    return CASE_ID_TO_NAME[case_id]


def make_temporal_training_sample(
    rng: np.random.Generator,
    prev_frame: dict | None,
    curr_frame: dict,
    width: int,
    height: int,
    depth_eps: float,
    case_name: str,
) -> dict:
    """
    Builds one training sample for the current frame using:
    real reprojection, no reprojection, or corrupted reprojection.
    """
    features_curr = curr_frame["features"]
    current_hit = curr_frame["hit_any"]
    target_delta = curr_frame["delta_rgb"]
    baseline_rgb = curr_frame["baseline_rgb"]
    target_rgb = curr_frame["target_rgb"]

    delta_reproj = masked_zero_like_hw3(height, width)
    valid_reproj = masked_zero_like_hw(height, width)

    if prev_frame is not None:
        cache = extract_delta_cache(
            delta_rgb=prev_frame["delta_rgb"],
            world_pos=prev_frame["world_pos"],
            valid_mask=prev_frame["hit_any"] > 0.5,
        )

        delta_real, reproj_zbuf, valid_proj = rasterize_reprojected_delta(
            points_world=cache["points_world"],
            delta_rgb=cache["delta_rgb"],
            camera=curr_frame["camera"],
            width=width,
            height=height,
        )

        valid_geom = compute_geom_validity(
            reproj_zbuf=reproj_zbuf,
            current_world_pos=curr_frame["world_pos"],
            current_hit_mask=current_hit > 0.5,
            camera=curr_frame["camera"],
            depth_eps=depth_eps,
        )

        valid_real = (valid_proj & valid_geom).astype(np.float32)
        delta_real *= valid_real[..., None]

        if case_name == "reproj":
            delta_reproj = delta_real
            valid_reproj = valid_real

        elif case_name == "none":
            delta_reproj = masked_zero_like_hw3(height, width)
            valid_reproj = masked_zero_like_hw(height, width)

        elif case_name == "corrupt":
            delta_reproj, valid_reproj = corrupt_reprojected_inputs(
                rng=rng,
                delta_reproj=delta_real,
                valid_reproj=valid_real,
                current_hit=current_hit,
            )

        else:
            raise ValueError(f"Unknown case_name: {case_name}")

    else:
        # First frame in a path has no previous frame by construction
        delta_reproj = masked_zero_like_hw3(height, width)
        valid_reproj = masked_zero_like_hw(height, width)
        case_name = "none"

    preview_rgb = compose_reprojected_frame(
        baseline_rgb=baseline_rgb,
        delta_reproj=delta_reproj,
        valid_mask=valid_reproj > 0.5,
    )

    return {
        "features_curr": features_curr.astype(np.float32),
        "delta_reproj": delta_reproj.astype(np.float32),
        "valid_reproj": valid_reproj.astype(np.float32),
        "target_delta": target_delta.astype(np.float32),
        "baseline_rgb": baseline_rgb.astype(np.float32),
        "target_rgb": target_rgb.astype(np.float32),
        "preview_rgb": preview_rgb.astype(np.float32),
        "curr_hit": current_hit.astype(np.float32),
        "case_id": np.int32(CASE_NAME_TO_ID[case_name]),
    }


def save_temporal_dataset_npz(
    out_path: Path,
    samples: list[dict],
) -> None:
    features_curr = np.stack([s["features_curr"] for s in samples], axis=0)
    delta_reproj = np.stack([s["delta_reproj"] for s in samples], axis=0)
    valid_reproj = np.stack([s["valid_reproj"] for s in samples], axis=0)
    target_delta = np.stack([s["target_delta"] for s in samples], axis=0)
    baseline_rgb = np.stack([s["baseline_rgb"] for s in samples], axis=0)
    target_rgb = np.stack([s["target_rgb"] for s in samples], axis=0)
    preview_rgb = np.stack([s["preview_rgb"] for s in samples], axis=0)
    curr_hit = np.stack([s["curr_hit"] for s in samples], axis=0)
    case_id = np.asarray([s["case_id"] for s in samples], dtype=np.int32)

    np.savez_compressed(
        out_path,
        features_curr=features_curr,
        delta_reproj=delta_reproj,
        valid_reproj=valid_reproj,
        target_delta=target_delta,
        baseline_rgb=baseline_rgb,
        target_rgb=target_rgb,
        preview_rgb=preview_rgb,
        curr_hit=curr_hit,
        case_id=case_id,
    )


def save_temporal_preview_grid(
    outdir: Path,
    samples: list[dict],
    max_items: int = 24,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    n = min(max_items, len(samples))
    if n == 0:
        return

    for i in range(n):
        s = samples[i]
        valid_rgb = np.zeros_like(s["baseline_rgb"])
        valid_rgb[..., 1] = s["valid_reproj"]

        save_triptych(
            outdir / f"sample_{i:04d}_{CASE_ID_TO_NAME[int(s['case_id'])]}.png",
            s["baseline_rgb"],
            s["target_rgb"],
            s["preview_rgb"],
        )
        save_image(outdir / f"sample_{i:04d}_valid.png", valid_rgb)
        # Visualise the reprojected delta around zero
        delta_vis = np.clip(0.5 + 0.5 * s["delta_reproj"], 0.0, 1.0)
        save_image(outdir / f"sample_{i:04d}_delta_reproj.png", delta_vis)


def render_path_preview_gif(
    out_path: Path,
    path_frames: list[dict],
) -> None:
    frames = []
    for s in path_frames:
        case_name = CASE_ID_TO_NAME[int(s["case_id"])]
        valid_rgb = np.zeros_like(s["baseline_rgb"])
        valid_rgb[..., 1] = s["valid_reproj"]
        row = concat_frames_horiz(
            [s["baseline_rgb"], s["target_rgb"], s["preview_rgb"], valid_rgb]
        )
        frames.append(row)
    save_gif(out_path, frames, duration_ms=80, loop=0)


def concat_frames_horiz(frames: list[np.ndarray]) -> list[np.ndarray] | np.ndarray:
    """
    Overload tolerant helper:
    if given a list of per image arrays, return one concatenated frame.
    if existing project helper already exists with a different signature, rename this.
    """
    if len(frames) == 0:
        raise ValueError("No frames to concatenate")
    return np.concatenate(frames, axis=1)

# ------------------------------------------------------------
# Broader camera start distribution plus local path generation
# ------------------------------------------------------------

def sample_base_camera(
    rng: np.random.Generator,
    base_radius: float = 4.0,
    base_height: float = 1.8,
    base_fov_y_deg: float = 45.0,
) -> Camera:
    """
    Sample a broad starting camera.

    Mixture of:
    1. Canonical anchor views with jitter
    2. Broad random views
    3. More extreme but still useful views
    """
    mode = rng.choice(
        ["anchor", "broad", "extreme"],
        p=[0.50, 0.35, 0.15],
    )

    target = np.array(
        [
            rng.uniform(-0.45, 0.45),
            rng.uniform(0.45, 1.05),
            rng.uniform(-0.45, 0.45),
        ],
        dtype=np.float32,
    )

    if mode == "anchor":
        # Canonical azimuth anchors around the object, similar spirit to original training coverage
        anchor_thetas = np.array(
            [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
            dtype=np.float32,
        )
        theta = float(rng.choice(anchor_thetas) + rng.uniform(-18.0, 18.0))
        radius = float(base_radius * rng.uniform(0.82, 1.18))
        height = float(base_height + rng.uniform(-0.55, 0.55))
        fov = float(base_fov_y_deg * rng.uniform(0.90, 1.12))

    elif mode == "broad":
        theta = float(rng.uniform(0.0, 360.0))
        radius = float(base_radius * rng.uniform(0.65, 1.45))
        height = float(base_height + rng.uniform(-1.00, 1.00))
        fov = float(base_fov_y_deg * rng.uniform(0.82, 1.22))

    else:
        # Deliberately include some harder starts
        theta = float(rng.uniform(0.0, 360.0))

        extreme_kind = rng.choice(
            ["close_low", "close_high", "far_low", "far_high", "grazing"],
            p=[0.22, 0.18, 0.20, 0.18, 0.22],
        )

        if extreme_kind == "close_low":
            radius = float(base_radius * rng.uniform(0.55, 0.80))
            height = float(base_height + rng.uniform(-1.10, -0.30))
        elif extreme_kind == "close_high":
            radius = float(base_radius * rng.uniform(0.55, 0.85))
            height = float(base_height + rng.uniform(0.55, 1.35))
        elif extreme_kind == "far_low":
            radius = float(base_radius * rng.uniform(1.20, 1.70))
            height = float(base_height + rng.uniform(-0.95, -0.10))
        elif extreme_kind == "far_high":
            radius = float(base_radius * rng.uniform(1.20, 1.75))
            height = float(base_height + rng.uniform(0.50, 1.55))
        else:
            radius = float(base_radius * rng.uniform(0.70, 1.35))
            height = float(base_height + rng.uniform(-1.20, -0.65))

        fov = float(base_fov_y_deg * rng.uniform(0.85, 1.18))

    cam = orbit_camera_with_target(
        theta_deg=theta,
        radius=radius,
        height=height,
        target=target,
        fov_y_deg=fov,
    )

    # Add a small base orientation perturbation so starting views are not too rigid
    cam = perturb_camera_orientation(
        cam,
        yaw_deg=float(rng.uniform(-10.0, 10.0)),
        pitch_deg=float(rng.uniform(-8.0, 8.0)),
        roll_deg=float(rng.uniform(-4.0, 4.0)),
    )

    return cam


def sample_camera_path_from_base(
    rng: np.random.Generator,
    base_camera: Camera,
    n_frames: int,
) -> list[Camera]:
    """
    Generate a short smooth path starting from a broad base camera.
    """
    mode = rng.choice(
        ["orbit", "dolly", "truck", "pedestal", "pan", "tilt", "mixed"],
        p=[0.20, 0.12, 0.14, 0.10, 0.14, 0.12, 0.18],
    )

    cameras: list[Camera] = [base_camera]
    f0, r0, u0 = camera_basis(base_camera)

    if n_frames <= 1:
        return cameras

    if mode == "orbit":
        # Orbit around the current target, preserving the spirit of a local fly around
        rel = base_camera.eye - base_camera.target
        radius0 = float(np.linalg.norm(rel[[0, 2]]))
        theta0 = math.degrees(math.atan2(rel[2], rel[0]))
        height0 = float(base_camera.eye[1])

        dtheta = rng.uniform(-40.0, 40.0)
        dr = rng.uniform(-0.45, 0.45)
        dh = rng.uniform(-0.45, 0.45)
        dtgt = rng.normal(0.0, 0.05, size=(3,)).astype(np.float32)

        for i in range(1, n_frames):
            t = i / (n_frames - 1)
            cam = orbit_camera_with_target(
                theta_deg=theta0 + dtheta * t,
                radius=max(0.4, radius0 + dr * t),
                height=height0 + dh * t,
                target=base_camera.target + dtgt * t,
                fov_y_deg=base_camera.fov_y_deg,
            )
            cam = perturb_camera_orientation(
                cam,
                yaw_deg=float(rng.uniform(-3.0, 3.0) * t),
                pitch_deg=float(rng.uniform(-3.0, 3.0) * t),
                roll_deg=float(rng.uniform(-1.0, 1.0) * t),
            )
            cameras.append(cam)

    elif mode == "dolly":
        d = rng.uniform(-1.20, 1.20)
        dtarget = rng.normal(0.0, 0.04, size=(3,)).astype(np.float32)

        for i in range(1, n_frames):
            t = i / (n_frames - 1)
            eye = base_camera.eye + f0 * (d * t)
            target = base_camera.target + dtarget * t
            cam = make_camera(
                eye=eye,
                target=target,
                up=base_camera.up,
                fov_y_deg=base_camera.fov_y_deg,
            )
            cameras.append(cam)

    elif mode == "truck":
        d = rng.uniform(-1.20, 1.20)
        d2 = rng.uniform(-0.35, 0.35)

        for i in range(1, n_frames):
            t = i / (n_frames - 1)
            shift = r0 * (d * t) + f0 * (d2 * t)
            cam = make_camera(
                eye=base_camera.eye + shift,
                target=base_camera.target + shift,
                up=base_camera.up,
                fov_y_deg=base_camera.fov_y_deg,
            )
            cameras.append(cam)

    elif mode == "pedestal":
        d = rng.uniform(-1.00, 1.00)

        for i in range(1, n_frames):
            t = i / (n_frames - 1)
            shift = u0 * (d * t)
            cam = make_camera(
                eye=base_camera.eye + shift,
                target=base_camera.target + shift,
                up=base_camera.up,
                fov_y_deg=base_camera.fov_y_deg,
            )
            cameras.append(cam)

    elif mode == "pan":
        yaw = rng.uniform(-28.0, 28.0)
        roll = rng.uniform(-5.0, 5.0)

        for i in range(1, n_frames):
            t = i / (n_frames - 1)
            cam = perturb_camera_orientation(
                base_camera,
                yaw_deg=yaw * t,
                pitch_deg=0.0,
                roll_deg=roll * t,
            )
            cameras.append(cam)

    elif mode == "tilt":
        pitch = rng.uniform(-20.0, 20.0)
        yaw = rng.uniform(-8.0, 8.0)

        for i in range(1, n_frames):
            t = i / (n_frames - 1)
            cam = perturb_camera_orientation(
                base_camera,
                yaw_deg=yaw * t,
                pitch_deg=pitch * t,
                roll_deg=0.0,
            )
            cameras.append(cam)

    else:
        # Mixed motion tends to look most like real camera motion
        rel = base_camera.eye - base_camera.target
        radius0 = float(np.linalg.norm(rel[[0, 2]]))
        theta0 = math.degrees(math.atan2(rel[2], rel[0]))
        height0 = float(base_camera.eye[1])

        dtheta = rng.uniform(-26.0, 26.0)
        dr = rng.uniform(-0.55, 0.55)
        dh = rng.uniform(-0.55, 0.55)
        shift_r = rng.uniform(-0.55, 0.55)
        shift_u = rng.uniform(-0.40, 0.40)
        yaw = rng.uniform(-16.0, 16.0)
        pitch = rng.uniform(-12.0, 12.0)
        dtgt = rng.normal(0.0, 0.05, size=(3,)).astype(np.float32)

        for i in range(1, n_frames):
            t = i / (n_frames - 1)
            cam = orbit_camera_with_target(
                theta_deg=theta0 + dtheta * t,
                radius=max(0.4, radius0 + dr * t),
                height=height0 + dh * t,
                target=base_camera.target + dtgt * t,
                fov_y_deg=base_camera.fov_y_deg,
            )
            _, rr, uu = camera_basis(cam)
            shift = rr * (shift_r * t) + uu * (shift_u * t)
            cam = make_camera(
                eye=cam.eye + shift,
                target=cam.target + shift,
                up=cam.up,
                fov_y_deg=cam.fov_y_deg,
            )
            cam = perturb_camera_orientation(
                cam,
                yaw_deg=yaw * t,
                pitch_deg=pitch * t,
                roll_deg=0.0,
            )
            cameras.append(cam)

    return cameras


def sample_camera_path(
    rng: np.random.Generator,
    n_frames: int,
    base_radius: float = 4.0,
    base_height: float = 1.8,
    base_fov_y_deg: float = 45.0,
) -> list[Camera]:
    """
    Backwards compatible wrapper.
    """
    base_cam = sample_base_camera(
        rng=rng,
        base_radius=base_radius,
        base_height=base_height,
        base_fov_y_deg=base_fov_y_deg,
    )
    return sample_camera_path_from_base(
        rng=rng,
        base_camera=base_cam,
        n_frames=n_frames,
    )

def main_training_gen_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--outdir", type=str, default="hybrid_render_training_gen_test")
    parser.add_argument("--num-paths", type=int, default=120)
    parser.add_argument("--frames-per-path", type=int, default=8)
    parser.add_argument("--preview-items", type=int, default=36)
    parser.add_argument("--preview-path-gifs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-eps", type=float, default=0.05)

    parser.add_argument("--p-reproj", type=float, default=0.60)
    parser.add_argument("--p-none", type=float, default=0.20)
    parser.add_argument("--p-corrupt", type=float, default=0.20)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    width = int(args.width)
    height = int(args.height)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    preview_path_samples: list[list[dict]] = []

    case_counts = {
        "reproj": 0,
        "none": 0,
        "corrupt": 0,
    }

    for path_idx in range(args.num_paths):
        cameras = sample_camera_path(
            rng=rng,
            n_frames=args.frames_per_path,
            base_radius=4.0,
            base_height=1.8,
            base_fov_y_deg=45.0,
        )

        rendered = [render_scene_frame(width=width, height=height, camera=cam) for cam in cameras]

        path_samples: list[dict] = []
        prev_frame = None

        for frame_idx, curr_frame in enumerate(rendered):
            if prev_frame is None:
                case_name = "none"
            else:
                case_name = choose_temporal_case(
                    rng=rng,
                    p_reproj=args.p_reproj,
                    p_none=args.p_none,
                    p_corrupt=args.p_corrupt,
                )

            sample = make_temporal_training_sample(
                rng=rng,
                prev_frame=prev_frame,
                curr_frame=curr_frame,
                width=width,
                height=height,
                depth_eps=float(args.depth_eps),
                case_name=case_name,
            )

            case_counts[CASE_ID_TO_NAME[int(sample["case_id"])]] += 1
            samples.append(sample)
            path_samples.append(sample)

            prev_frame = curr_frame

        preview_path_samples.append(path_samples)

    save_temporal_dataset_npz(outdir / "temporal_training_samples.npz", samples)
    save_temporal_preview_grid(outdir / "preview_samples", samples, max_items=args.preview_items)

    gif_count = min(args.preview_path_gifs, len(preview_path_samples))
    gif_dir = outdir / "preview_path_gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    for i in range(gif_count):
        render_path_preview_gif(
            gif_dir / f"path_{i:03d}.gif",
            preview_path_samples[i],
        )

    num_samples = len(samples)
    print("Saved:", outdir / "temporal_training_samples.npz")
    print("Total samples:", num_samples)
    for k in ["reproj", "none", "corrupt"]:
        frac = case_counts[k] / max(1, num_samples)
        print(f"{k:8s} count={case_counts[k]:6d} frac={frac:.3f}")

    # Basic sanity checks
    valid_means = np.asarray([np.mean(s["valid_reproj"]) for s in samples], dtype=np.float64)
    delta_mags = np.asarray([np.mean(np.abs(s["delta_reproj"])) for s in samples], dtype=np.float64)
    print(f"mean(valid_reproj) = {valid_means.mean():.4f}")
    print(f"mean(abs(delta_reproj)) = {delta_mags.mean():.4f}")


def main_delta_reprojection_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--outdir", type=str, default="hybrid_render_out_test")
    parser.add_argument("--n-frames", type=int, default=64)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--height-base", type=float, default=1.8)
    parser.add_argument("--height-amp", type=float, default=0.6)
    parser.add_argument("--theta-start-deg", type=float, default=0.0)
    parser.add_argument("--theta-end-deg", type=float, default=360.0)
    parser.add_argument("--depth-eps", type=float, default=0.05)
    parser.add_argument("--hole-fill-iters", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (
        baseline_frames,
        target_frames,
        reproj_frames,
        valid_vis_frames,
        coverage_vis_frames,
        reproj_ms_list,
    ) = render_reprojected_delta_sequence(
        width=args.width, height=args.height, n_frames=args.n_frames, radius=args.radius,
        height_base=args.height_base,
        height_amp=args.height_amp,
        theta_start_deg=args.theta_start_deg,
        theta_end_deg=args.theta_end_deg,
        depth_eps=args.depth_eps,
        hole_fill_iters=args.hole_fill_iters,
    )

    # Single frame previews
    save_image(outdir / "frame0_baseline.png", baseline_frames[0])
    save_image(outdir / "frame0_target.png", target_frames[0])
    save_image(outdir / "frame0_reprojected.png", reproj_frames[0])
    save_triptych(
        outdir / "frame0_triptych.png",
        baseline_frames[0],
        target_frames[0],
        reproj_frames[0],
    )

    # GIFs
    save_gif(outdir / "baseline.gif", baseline_frames, duration_ms=60, loop=0)
    save_gif(outdir / "target.gif", target_frames, duration_ms=60, loop=0)
    save_gif(outdir / "reprojected_delta.gif", reproj_frames, duration_ms=60, loop=0)
    save_gif(outdir / "validity.gif", valid_vis_frames, duration_ms=60, loop=0)
    save_gif(outdir / "coverage.gif", coverage_vis_frames, duration_ms=60, loop=0)

    # Side by side comparisons
    save_gif(
        outdir / "baseline_vs_reprojected.gif",
        concat_frames_horiz(baseline_frames, reproj_frames),
        duration_ms=60,
        loop=0,
    )
    save_gif(
        outdir / "target_vs_reprojected.gif",
        concat_frames_horiz(target_frames, reproj_frames),
        duration_ms=60,
        loop=0,
    )

    # Simple coverage metric over the whole sequence
    coverages = []
    mses = []
    for target, reproj in zip(target_frames, reproj_frames):
        # Estimate effective coverage as pixels that differ from baseline composition result
        # is not directly available here, so use a conservative "non baseline difference" proxy.
        # Better quantitative metrics can be added later with explicit masks returned.
        diff = np.mean(np.abs(target - reproj), axis=-1)
        mses.append(float(np.mean((target - reproj) ** 2)))
        coverages.append(float(np.mean(diff > 1e-5)))

    print("Saved outputs to:", outdir)
    print(f"Approx mean frame MSE target vs reprojected: {np.mean(mses):.6f}")
    print(f"Approx mean changed pixel fraction:        {np.mean(coverages):.6f}")
    print_timing_stats("reprojection", reproj_ms_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default="hybrid_render_out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_camera = orbit_camera(theta_deg=40.0)
    scene = build_scene(args.width, args.height, camera=train_camera)

    save_image(outdir / "baseline.png", scene["baseline_rgb"])
    save_image(outdir / "target.png", scene["target_rgb"])

    train_data = make_dataset(
        n_views=192,
        width=args.width,
        height=args.height,
        theta_min_deg=0.0,
        theta_max_deg=180.0,
        seed=args.seed,
    )

    test_data = make_dataset(
        n_views=24,
        width=args.width,
        height=args.height,
        theta_min_deg=0.0,
        theta_max_deg=180.0,
        seed=args.seed + 1,
    )

    model = train_partial_cnn_multiview(
        train_data["features"],
        train_data["baseline_rgb"],
        train_data["target_rgb"],
        train_data["hit_any"],
        steps=args.steps,
        batch_size=8,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        residual_scale=0.5,
    )

    save_dataset_examples(
        model,
        train_data,
        outdir / "train_examples",
        prefix="train",
        device=args.device,
        residual_scale=0.5,
        count=4,
    )

    save_dataset_examples(
        model,
        test_data,
        outdir / "test_examples",
        prefix="test",
        device=args.device,
        residual_scale=0.5,
        count=4,
    )

    baseline_frames, target_frames, pred_frames = render_gif_frames(
        model,
        width=args.width,
        height=args.height,
        n_frames=64,
        radius=4.0,
        height_base=1.8,
        height_amp=0.6,
        theta_start_deg=0.0,
        theta_end_deg=360.0,
        device=args.device,
        residual_scale=0.5,
    )

    save_gif(outdir / "neural.gif", pred_frames, duration_ms=70)
    save_gif(outdir / "baseline_vs_neural.gif", concat_frames_horiz(baseline_frames, pred_frames), duration_ms=70)
    save_gif(outdir / "target_vs_neural.gif", concat_frames_horiz(target_frames, pred_frames))
    
    print(f"Saved outputs to: {outdir.resolve()}")
    print("Columns in triptych images are: baseline | stylised target | neural prediction")


if __name__ == "__main__":
    main_training_gen_test()
