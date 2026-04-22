#!/usr/bin/env python3

# use with blender:
# blender -b scene.blend -P export_normals_depth.py -- --camera Camera --outdir .gout --width 512 --height 512

import argparse
import shutil
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
        description="Render Blender scene headlessly and export normal RGB and grayscale depth from the active camera."
    )
    p.add_argument("--camera", type=str, default=None, help="Camera object name. Defaults to scene camera.")
    p.add_argument("--view-layer", type=str, default=None, help="View layer name. Defaults to active view layer.")
    p.add_argument("--frame", type=int, default=None, help="Frame to render. Defaults to current frame.")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    p.add_argument("--width", type=int, default=None, help="Render width in pixels.")
    p.add_argument("--height", type=int, default=None, help="Render height in pixels.")
    p.add_argument("--depth-near", type=float, required=True, help="Near depth for normalization.")
    p.add_argument("--depth-far", type=float, required=True, help="Far depth for normalization.")
    p.add_argument("--invert-depth", action="store_true", help="Make near white and far black.")
    p.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate compositor files.")
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


def enable_passes(view_layer):
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True


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


def find_new_input_socket(node, before_names, after_names):
    new_names = [name for name in after_names if name not in before_names]
    if len(new_names) == 1:
        return node.inputs[new_names[0]]
    if len(node.inputs) > len(before_names):
        return node.inputs[-1]
    raise RuntimeError(
        f"Could not determine new input socket. Before: {before_names}, After: {after_names}"
    )


def setup_file_output_node(tree, location, directory: Path, file_name: str, socket_type: str, item_name: str):
    n_out = tree.nodes.new(type="CompositorNodeOutputFile")
    n_out.location = location
    n_out.directory = str(directory)
    n_out.file_name = file_name
    n_out.format.file_format = "PNG"
    n_out.format.color_depth = "16"
    n_out.format.color_mode = "RGB"

    items = n_out.file_output_items
    before_names = socket_names(n_out.inputs)
    item = items.new(socket_type, item_name)
    after_names = socket_names(n_out.inputs)
    input_socket = find_new_input_socket(n_out, before_names, after_names)
    return n_out, item, input_socket


def setup_compositor(scene, view_layer_name: str, outdir: Path, depth_near: float, depth_far: float, invert_depth: bool, debug_api=False):
    tree = getattr(scene, "compositing_node_group", None)
    if tree is None:
        tree = bpy.data.node_groups.new(
            name=f"{scene.name}_Compositor",
            type="CompositorNodeTree",
        )
        scene.compositing_node_group = tree

    print(f"Using compositor node tree from scene.compositing_node_group: {tree.name}")

    clear_node_tree(tree)

    n_render = tree.nodes.new(type="CompositorNodeRLayers")
    n_render.location = (0, 0)

    if hasattr(n_render, "layer"):
        n_render.layer = view_layer_name
    elif hasattr(n_render, "view_layer"):
        n_render.view_layer = view_layer_name

    normal_output_name = find_render_output_name(n_render, ["Normal", "normal"])
    depth_output_name = find_render_output_name(n_render, ["Depth", "Z", "depth", "z"])

    # Normal remap from [-1, 1] to [0, 1]
    n_sep = tree.nodes.new(type="CompositorNodeSeparateColor")
    n_sep.location = (220, 200)
    if hasattr(n_sep, "mode"):
        n_sep.mode = "RGB"

    n_mul_r = tree.nodes.new(type="CompositorNodeMath")
    n_mul_r.operation = "MULTIPLY"
    n_mul_r.inputs[1].default_value = 0.5
    n_mul_r.location = (420, 320)

    n_add_r = tree.nodes.new(type="CompositorNodeMath")
    n_add_r.operation = "ADD"
    n_add_r.inputs[1].default_value = 0.5
    n_add_r.location = (620, 320)

    n_mul_g = tree.nodes.new(type="CompositorNodeMath")
    n_mul_g.operation = "MULTIPLY"
    n_mul_g.inputs[1].default_value = 0.5
    n_mul_g.location = (420, 200)

    n_add_g = tree.nodes.new(type="CompositorNodeMath")
    n_add_g.operation = "ADD"
    n_add_g.inputs[1].default_value = 0.5
    n_add_g.location = (620, 200)

    n_mul_b = tree.nodes.new(type="CompositorNodeMath")
    n_mul_b.operation = "MULTIPLY"
    n_mul_b.inputs[1].default_value = 0.5
    n_mul_b.location = (420, 80)

    n_add_b = tree.nodes.new(type="CompositorNodeMath")
    n_add_b.operation = "ADD"
    n_add_b.inputs[1].default_value = 0.5
    n_add_b.location = (620, 80)

    n_combine = tree.nodes.new(type="CompositorNodeCombineColor")
    n_combine.location = (840, 200)
    if hasattr(n_combine, "mode"):
        n_combine.mode = "RGB"

    # Depth map range to [0, 1]
    n_map = tree.nodes.new(type="CompositorNodeMapRange")
    n_map.location = (300, -220)
    n_map.inputs[1].default_value = depth_near
    n_map.inputs[2].default_value = depth_far
    if invert_depth:
        n_map.inputs[3].default_value = 1.0
        n_map.inputs[4].default_value = 0.0
    else:
        n_map.inputs[3].default_value = 0.0
        n_map.inputs[4].default_value = 1.0
    if hasattr(n_map, "clamp"):
        n_map.clamp = True
    elif hasattr(n_map, "use_clamp"):
        n_map.use_clamp = True

    n_depth_rgb = tree.nodes.new(type="CompositorNodeCombineColor")
    n_depth_rgb.location = (540, -220)
    if hasattr(n_depth_rgb, "mode"):
        n_depth_rgb.mode = "RGB"

    # Output nodes
    n_normal_out, _, normal_input = setup_file_output_node(
        tree=tree,
        location=(1100, 220),
        directory=outdir,
        file_name="normal_rgb",
        socket_type="RGBA",
        item_name="normal_rgb",
    )

    n_depth_out, _, depth_input = setup_file_output_node(
        tree=tree,
        location=(820, -220),
        directory=outdir,
        file_name="depth_gray",
        socket_type="RGBA",
        item_name="depth_gray",
    )

    # Links
    tree.links.new(n_render.outputs[normal_output_name], n_sep.inputs[0])

    tree.links.new(n_sep.outputs[0], n_mul_r.inputs[0])
    tree.links.new(n_mul_r.outputs[0], n_add_r.inputs[0])

    tree.links.new(n_sep.outputs[1], n_mul_g.inputs[0])
    tree.links.new(n_mul_g.outputs[0], n_add_g.inputs[0])

    tree.links.new(n_sep.outputs[2], n_mul_b.inputs[0])
    tree.links.new(n_mul_b.outputs[0], n_add_b.inputs[0])

    tree.links.new(n_add_r.outputs[0], n_combine.inputs[0])
    tree.links.new(n_add_g.outputs[0], n_combine.inputs[1])
    tree.links.new(n_add_b.outputs[0], n_combine.inputs[2])

    tree.links.new(n_combine.outputs[0], normal_input)

    tree.links.new(n_render.outputs[depth_output_name], n_map.inputs[0])
    tree.links.new(n_map.outputs[0], n_depth_rgb.inputs[0])
    tree.links.new(n_map.outputs[0], n_depth_rgb.inputs[1])
    tree.links.new(n_map.outputs[0], n_depth_rgb.inputs[2])
    tree.links.new(n_depth_rgb.outputs[0], depth_input)

    if debug_api:
        print("Render Layer outputs:", socket_names(n_render.outputs))
        print("Normal File Output inputs:", socket_names(n_normal_out.inputs))
        print("Depth File Output inputs:", socket_names(n_depth_out.inputs))
        print("Normal input chosen:", normal_input.name, getattr(normal_input, "type", None))
        print("Depth input chosen:", depth_input.name, getattr(depth_input, "type", None))

    return tree


def render(scene, frame=None):
    if frame is not None:
        scene.frame_set(frame)
    bpy.ops.render.render(write_still=False)


def list_written_files(root: Path):
    return sorted(str(p) for p in root.glob("**/*") if p.is_file())


def find_output_file(root: Path, prefixes, suffix=".png"):
    for prefix in prefixes:
        matches = sorted(
            p for p in root.glob("**/*")
            if p.is_file() and p.suffix.lower() == suffix and p.name.startswith(prefix)
        )
        if matches:
            return matches[0]

    written = list_written_files(root)
    raise RuntimeError(
        f"Could not find output file with any prefix {prefixes} under {root}.\n"
        f"Found files:\n" + "\n".join(written)
    )


def load_image_pixels(path: Path):
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        w, h = img.size
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        return arr
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
    enable_passes(view_layer)

    setup_compositor(
        scene=scene,
        view_layer_name=view_layer.name,
        outdir=outdir,
        depth_near=args.depth_near,
        depth_far=args.depth_far,
        invert_depth=args.invert_depth,
        debug_api=args.debug_api,
    )

    render(scene, frame=args.frame)

    if args.debug_api:
        print("Written files after render:")
        for f in list_written_files(outdir):
            print(" ", f)

    normal_png = find_output_file(outdir, prefixes=["normal_rgb"])
    depth_png = find_output_file(outdir, prefixes=["depth_gray"])

    normal_rgba = load_image_pixels(normal_png)
    depth_rgba = load_image_pixels(depth_png)

    normal_rgb = normal_rgba[..., :3].astype(np.float32)
    depth_vis = depth_rgba[..., 0].astype(np.float32)

    np.savez_compressed(
        outdir / "passes.npz",
        normal_rgb=normal_rgb,
        depth_vis=depth_vis,
        camera_name=np.array(cam.name),
        frame=np.int32(scene.frame_current),
        width=np.int32(scene.render.resolution_x),
        height=np.int32(scene.render.resolution_y),
        depth_vis_near=np.float32(args.depth_near),
        depth_vis_far=np.float32(args.depth_far),
        depth_inverted=np.uint8(1 if args.invert_depth else 0),
    )

    if not args.keep_intermediate:
        for p in outdir.glob("*.tmp"):
            p.unlink(missing_ok=True)

    print(f"Saved {normal_png}")
    print(f"Saved {depth_png}")
    print(f"Saved {outdir / 'passes.npz'}")
    print(f"Camera: {cam.name}")
    print(f"Frame: {scene.frame_current}")
    print(f"Depth normalization range: [{args.depth_near}, {args.depth_far}]")
    print("Note: passes.npz contains normalized depth_vis, not raw metric depth.")


if __name__ == "__main__":
    main()
