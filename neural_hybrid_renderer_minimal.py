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
        if rng.random() < 0.3:
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

    # model, pred = train_on_single_view(
        # scene["features"],
        # scene["target_rgb"],
        # steps=args.steps,
        # lr=args.lr,
        # device=args.device,
        # seed=args.seed,
    # )

    train_data = make_dataset(
        n_views=96,
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

    # model, pred = train_cnn_on_single_view(
        # scene["features"],
        # scene["baseline_rgb"],
        # scene["target_rgb"],
        # steps=args.steps,
        # lr=args.lr,
        # device=args.device,
        # seed=args.seed,
    # )
# 
    # save_image(outdir / "prediction_train_view.png", pred)
    # save_triptych(
        # outdir / "triptych_train_view.png",
        # scene["baseline_rgb"],
        # scene["target_rgb"],
        # pred,
    # )

    # A couple of held-out viewpoints
    # for angle in [10.0, 80.0, 140.0]:
        # cam = orbit_camera(theta_deg=angle)
        # test_scene = build_scene(args.width, args.height, camera=cam)
        #test_pred = render_with_model(model, test_scene["features"], device=args.device)
        # test_pred = render_with_cnn(model, test_scene["features"], test_scene["baseline_rgb"], device=args.device)
# 
        # save_triptych(
            # outdir / f"triptych_view_{int(angle):03d}.png",
            # test_scene["baseline_rgb"],
            # test_scene["target_rgb"],
            # test_pred,
        # )
# 
    print(f"Saved outputs to: {outdir.resolve()}")
    print("Columns in triptych images are: baseline | stylised target | neural prediction")


if __name__ == "__main__":
    main()
