import glob
import os
import numpy as np
import bpy
from mathutils import Matrix
import math

def unproject_depth_map_to_point_map(depth, extrinsics, intrinsics):
    N, H, W = depth.shape
    world_points = np.zeros((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pixels = np.stack([u, v, np.ones((H, W))], axis=-1).reshape(-1, 3)  # HW, 3
        invK = np.linalg.inv(intrinsics[i])
        rays = (invK @ pixels.T).T  # HW, 3
        depths = depth[i].reshape(-1)  # HW
        cam_points = rays * depths[:, np.newaxis]  # HW, 3
        cam_points_hom = np.hstack([cam_points, np.ones((len(depths), 1))])  # HW, 4
        E = np.vstack([extrinsics[i], [0, 0, 0, 1]])  # 4, 4
        cam_to_world = np.linalg.inv(E)
        world_points_hom = (cam_to_world @ cam_points_hom.T).T  # HW, 4
        world_points_i = world_points_hom[:, :3] / world_points_hom[:, 3:4]
        world_points[i] = world_points_i.reshape(H, W, 3)
    return world_points

def run_model(target_dir, model):
    print(f"Processing images from {target_dir}")
    image_paths = sorted(glob.glob(os.path.join(target_dir, "*.[jJpP][pPnN][gG]")))
    if not image_paths:
        raise ValueError("No images found in the target directory.")
    print(f"Found {len(image_paths)} images")
    prediction = model.inference(image_paths)
    predictions = {}
    predictions['images'] = prediction.processed_images.astype(np.float32) / 255.0  # [N, H, W, 3]
    predictions['depth'] = prediction.depth
    predictions['extrinsic'] = prediction.extrinsics
    predictions['intrinsic'] = prediction.intrinsics
    print("Computing world points from depth map...")
    world_points = unproject_depth_map_to_point_map(predictions['depth'], predictions['extrinsic'], predictions['intrinsic'])
    predictions["world_points_from_depth"] = world_points
    return predictions

def import_point_cloud(d):
    points = d["world_points_from_depth"]
    images = d["images"]
    points_batch = points.reshape(-1, 3)
    reordered_points_batch = points_batch.copy()
    reordered_points_batch[:, [0, 1, 2]] = points_batch[:, [0, 2, 1]]
    reordered_points_batch[:, 2] = -reordered_points_batch[:, 2]
    points_batch = reordered_points_batch
    colors_batch = images.reshape(-1, 3)
    colors_batch = np.hstack((colors_batch, np.ones((colors_batch.shape[0], 1))))
    mesh = bpy.data.meshes.new(name="Points")
    vertices = points_batch.tolist()
    mesh.from_pydata(vertices, [], [])
    attribute = mesh.attributes.new(name="point_color", type="FLOAT_COLOR", domain="POINT")
    color_values = colors_batch.flatten().tolist()
    attribute.data.foreach_set("color", color_values)
    obj = bpy.data.objects.new("Points", mesh)
    bpy.context.collection.objects.link(obj)
    mat = bpy.data.materials.new(name="PointMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    attr_node = nodes.new('ShaderNodeAttribute')
    attr_node.attribute_name = "point_color"
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
    output_node_material = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output_node_material.inputs['Surface'])
    geo_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
    node_group = bpy.data.node_groups.new(name="PointCloud", type='GeometryNodeTree')
    node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    geo_mod.node_group = node_group
    input_node = node_group.nodes.new('NodeGroupInput')
    output_node = node_group.nodes.new('NodeGroupOutput')
    mesh_to_points = node_group.nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points.inputs['Radius'].default_value = 0.002
    set_material_node = node_group.nodes.new('GeometryNodeSetMaterial')
    set_material_node.inputs['Material'].default_value = mat
    node_group.links.new(input_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
    node_group.links.new(mesh_to_points.outputs['Points'], set_material_node.inputs['Geometry'])
    node_group.links.new(set_material_node.outputs['Geometry'], output_node.inputs['Geometry'])

def create_cameras(predictions, image_width=None, image_height=None):
    scene = bpy.context.scene
    if image_width is None or image_height is None:
        H, W = predictions['images'].shape[1:3]
        image_width = W
        image_height = H
    K0 = predictions["intrinsic"][0]
    pixel_aspect_y = K0[1,1] / K0[0,0]
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = float(pixel_aspect_y)
    num_cameras = len(predictions["extrinsic"])
    if len(predictions["intrinsic"]) != num_cameras:
        raise ValueError("Extrinsic and intrinsic lists must have the same length")
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    for i in range(num_cameras):
        cam_data = bpy.data.cameras.new(name=f"Camera_{i}")
        K = predictions["intrinsic"][i]
        f_x = K[0,0]
        c_x = K[0,2]
        c_y = K[1,2]
        sensor_width = 36.0
        cam_data.sensor_width = sensor_width
        cam_data.lens = (f_x / image_width) * sensor_width
        cam_data.shift_x = (c_x - image_width / 2.0) / image_width
        cam_data.shift_y = (c_y - image_height / 2.0) / image_height
        cam_obj = bpy.data.objects.new(name=f"Camera_{i}", object_data=cam_data)
        scene.collection.objects.link(cam_obj)
        ext = predictions["extrinsic"][i]
        E = np.vstack((ext, [0, 0, 0, 1]))
        E_inv = np.linalg.inv(E)
        M = np.dot(E_inv, T)
        cam_obj.matrix_world = Matrix(M.tolist())
        R = Matrix.Rotation(math.radians(-90), 4, 'X')
        cam_obj.matrix_world = R @ cam_obj.matrix_world