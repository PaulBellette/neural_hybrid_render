#!/usr/bin/env python3

# use with blender:
# blender -b scene.blend -P export_normals_depth.py -- --camera Camera --outdir .gout --width 512 --height 512 --depth-near 0.1 --depth-far 10.0

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser(
        description="Render Blender scene headlessly and export normal RGB plus grayscale mist depth."
    )
    p.add_argument("--camera", type=str, default=None, help="Camera object name. Defaults to scene camera.")
    p.add_argument("--view-layer", type=str, default=None, help="View layer name. Defaults to active view layer.")
    p.add_argument("--frame", type=int, default=None, help="Frame to render. Defaults to current frame.")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    p.add_argument("--width", type=int, default=None, help="Render width in pixels.")
    p.add_argument("--height", type=int, default=None, help="Render height in pixels.")
    p.add_argument("--depth-near", type=float, required=True, help="Mist start distance.")
    p.add_argument("--depth-far", type=float, required=True, help="Mist end distance.")
    p.add_argument("--invert-depth", action="store_true", help="Make near white and far black.")
    p.add_argument("--debug-api", action="store_true", help="Print compositor debug information.")
    return p.parse_args(argv)


def set_camera(scene, camera_name):
    if camera_name is None:
        if scene.camera is None:
            raise RuntimeError("Scene has no active camera. Use --camera to specify one.")
        return scene.camera

    cam = bpy.data.objects.get(camera_name)
    if cam is None:
        raise RuntimeError(f"Camera '{camera_name}' not found.")
    if cam.type != "CAMERA":
        raise RuntimeError(f"Object '{camera_name}' is not a camera.")
    scene.camera = cam
    return cam


def set_resolution(scene, width, height):
    if width is not None:
        scene.render.resolution_x = width
    if height is not None:
        scene.render.resolution_y = height
    scene.render.resolution_percentage = 100


def get_view_layer(scene, view_layer_name):
    if view_layer_name is None:
        return bpy.context.view_layer
    vl = scene.view_layers.get(view_layer_name)
    if vl is None:
        raise RuntimeError(f"View layer '{view_layer_name}' not found.")
    return vl


def enable_passes(scene, view_layer, mist_start, mist_end):
    view_layer.use_pass_normal = True
    view_layer.use_pass_mist = True

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    mist = world.mist_settings
    mist.use_mist = True
    mist.start = float(mist_start)
    mist.depth = float(mist_end - mist_start)
    if mist.depth <= 0:
        raise RuntimeError("--depth-far must be greater than --depth-near")


def clear_node_tree(tree):
    while tree.nodes:
        tree.nodes.remove(tree.nodes[0])


def socket_names(sockets):
    return [sock.name for sock in sockets]


def find_render_output_name(render_node, candidates):
    output_names = set(render_node.outputs.keys())
    for candidate in candidates:
        if candidate in output_names:
            return candidate
    raise RuntimeError(
        f"Could not find any of {candidates} on Render Layers node. "
        f"Available outputs: {sorted(output_names)}"
    )


def setup_viewer_compositor(scene, view_layer_name, pass_name, debug_api=False):
    tree = getattr(scene, "compositing_node_group", None)
    if tree is None:
        tree = bpy.data.node_groups.new(
            name=f"{scene.name}_Compositor",
            type="CompositorNodeTree",
        )
        scene.compositing_node_group = tree

    clear_node_tree(tree)

    n_render = tree.nodes.new(type="CompositorNodeRLayers")
    n_render.location = (0, 0)

    if hasattr(n_render, "layer"):
        n_render.layer = view_layer_name
    elif hasattr(n_render, "view_layer"):
        n_render.view_layer = view_layer_name

    n_viewer = tree.nodes.new(type="CompositorNodeViewer")
    n_viewer.location = (300, 0)

    output_name = find_render_output_name(n_render, [pass_name])

    tree.links.new(n_render.outputs[output_name], n_viewer.inputs[0])

    if debug_api:
        print(f"Viewer compositor for pass {pass_name}")
        print("Render Layer outputs:", socket_names(n_render.outputs))
        print("Viewer inputs:", socket_names(n_viewer.inputs))

    return tree


def render(scene, frame=None):
    if frame is not None:
        scene.frame_set(frame)
    bpy.ops.render.render(write_still=False)


def load_viewer_pixels():
    img = bpy.data.images.get("Viewer Node")
    if img is None:
        raise RuntimeError("Viewer Node image not found after render.")

    w, h = img.size
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Viewer Node image has invalid size {w}x{h}")

    arr = np.array(img.pixels[:], dtype=np.float32)
    expected = w * h * 4
    if arr.size != expected:
        raise RuntimeError(f"Viewer Node pixel buffer has size {arr.size}, expected {expected}")

    return arr.reshape(h, w, 4)


def make_normal_rgb(normal_xyz):
    return np.clip(0.5 * normal_xyz + 0.5, 0.0, 1.0)


def save_png_rgb(path: Path, rgb01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise RuntimeError(f"Expected rgb01 shape (H, W, 3), got {rgb01.shape}")

    h, w, _ = rgb01.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(rgb01, 0.0, 1.0)

    img = bpy.data.images.new(name=path.stem, width=w, height=h, alpha=True, float_buffer=False)
    try:
        img.filepath_raw = str(path)
        img.file_format = "PNG"
        img.pixels.foreach_set(rgba.reshape(-1))
        img.save()
    finally:
        bpy.data.images.remove(img)


def save_png_gray(path: Path, gray01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    if gray01.ndim != 2:
        raise RuntimeError(f"Expected gray01 shape (H, W), got {gray01.shape}")

    h, w = gray01.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    g = np.clip(gray01, 0.0, 1.0)
    rgba[..., 0] = g
    rgba[..., 1] = g
    rgba[..., 2] = g

    img = bpy.data.images.new(name=path.stem, width=w, height=h, alpha=True, float_buffer=False)
    try:
        img.filepath_raw = str(path)
        img.file_format = "PNG"
        img.pixels.foreach_set(rgba.reshape(-1))
        img.save()
    finally:
        bpy.data.images.remove(img)


def main():
    args = parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    cam = set_camera(scene, args.camera)
    set_resolution(scene, args.width, args.height)

    view_layer = get_view_layer(scene, args.view_layer)
    enable_passes(scene, view_layer, args.depth_near, args.depth_far)

    print(f"Using compositor node tree from scene.compositing_node_group: "
          f"{getattr(scene.compositing_node_group, 'name', '<new tree>')}")

    # Render normal pass to Viewer Node
    setup_viewer_compositor(scene, view_layer.name, "Normal", debug_api=args.debug_api)
    render(scene, frame=args.frame)
    normal_rgba = load_viewer_pixels()
    normal = normal_rgba[..., :3].astype(np.float32)
    normal_rgb = make_normal_rgb(normal)

    # Render mist pass to Viewer Node
    setup_viewer_compositor(scene, view_layer.name, "Mist", debug_api=args.debug_api)
    render(scene, frame=args.frame)
    mist_rgba = load_viewer_pixels()
    depth_vis = mist_rgba[..., 0].astype(np.float32)

    if args.invert_depth:
        depth_vis = 1.0 - depth_vis

    normal_png = outdir / "normal_rgb.png"
    depth_png = outdir / "depth_gray.png"

    save_png_rgb(normal_png, normal_rgb)
    save_png_gray(depth_png, depth_vis)

    np.savez_compressed(
        outdir / "passes.npz",
        normal=normal,
        normal_rgb=normal_rgb.astype(np.float32),
        depth_vis=depth_vis.astype(np.float32),
        camera_name=np.array(cam.name),
        frame=np.int32(scene.frame_current),
        width=np.int32(scene.render.resolution_x),
        height=np.int32(scene.render.resolution_y),
        depth_vis_near=np.float32(args.depth_near),
        depth_vis_far=np.float32(args.depth_far),
        depth_inverted=np.uint8(1 if args.invert_depth else 0),
    )

    print(f"Saved {normal_png}")
    print(f"Saved {depth_png}")
    print(f"Saved {outdir / 'passes.npz'}")
    print(f"Camera: {cam.name}")
    print(f"Frame: {scene.frame_current}")
    print(f"Depth normalization range: [{args.depth_near}, {args.depth_far}]")
    print("passes.npz contains raw normal, remapped normal_rgb, and normalized depth_vis.")


if __name__ == "__main__":
    main()

