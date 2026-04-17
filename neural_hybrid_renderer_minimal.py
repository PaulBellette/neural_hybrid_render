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

def main_test():
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
    main_test()
