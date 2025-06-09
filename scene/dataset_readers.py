#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import cv2
import json
import torch
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.SMCReader import SMCReader
import pickle
from xrprimer.data_structure.camera import FisheyeCameraParameter
from scene.ours_undistort import undistort_images
from mmhuman3d.core.conventions.cameras.convert_convention import convert_camera_matrix
from tqdm import tqdm


opengl_to_cv2 = torch.tensor([[1.0, 0.0, 0.0, 0.0],   # inverse y-axis and z-axis, no translation
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

def init_camera_x(nviews=36, camera_sampling='uniform', camera_mode='persp', size=1024, cam_dis=3.0):
    def cartesian_to_spherical(xyz):
        radius = torch.sqrt(torch.sum(xyz ** 2, axis=1))
        xz = torch.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
        elevation = torch.atan2(xyz[:, 1], xz)
        azimuth = torch.atan2(xyz[:, 0], xyz[:, 2])

        return torch.rad2deg(elevation), torch.rad2deg(azimuth), radius

    look_at = torch.zeros((nviews, 3), dtype=torch.float32)
    camera_up_direction = torch.tensor([[0, 1, 0]], dtype=torch.float32).repeat(nviews, 1, )

    if camera_sampling == 'uniform':
        angle = torch.linspace(0, 2 * np.pi, nviews + 1)[:-1]
        camera_position = torch.stack((cam_dis * torch.sin(angle), torch.zeros_like(angle), cam_dis * torch.cos(angle)), dim=1)


    if camera_mode == 'orth':
        camera = kal.render.camera.Camera.from_args(eye=camera_position,
                                                    at=look_at,
                                                    up=camera_up_direction,
                                                    width=size, height=size,
                                                    near=-512, far=512,
                                                    fov_distance=1.0, device='cpu')

    elif camera_mode == 'persp':
        camera = kal.render.camera.Camera.from_args(eye=camera_position,
                                                    at=look_at,
                                                    up=camera_up_direction,
                                                    fov=45 * np.pi / 180,
                                                    width=size, height=size,
                                                    near=0.01, far=10,
                                                    device='cpu')
    else:
        raise NotImplementedError

    cam_pos = camera.extrinsics.cam_pos()
    elevation, azimuth, radius = cartesian_to_spherical(cam_pos)

    camera_dict = {}
    camera_dict['cam_pos'] = cam_pos.cpu().numpy()
    camera_dict['elevation'] = elevation.cpu().numpy()
    camera_dict['azimuth'] = azimuth.cpu().numpy()
    camera_dict['radius'] = radius.cpu().numpy()
    camera_dict['R'] = camera.extrinsics.R.cpu().numpy()
    camera_dict['t'] = camera.extrinsics.t.cpu().numpy()
    camera_dict['intr'] = {
        'cx': camera.intrinsics.cx.cpu().numpy(),
        'cy': camera.intrinsics.cy.cpu().numpy(),
        'fx': camera.intrinsics.focal_x.cpu().numpy(),
        'fy': camera.intrinsics.focal_y.cpu().numpy(),
    }
    return camera, camera_dict

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    corners_2d[:,0] = np.clip(corners_2d[:,0], 0, W)
    corners_2d[:,1] = np.clip(corners_2d[:,1], 0, H)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

class CameraInfo(NamedTuple):
    uid: int
    global_id: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float

class CameraInfo_dna(NamedTuple):
    uid: int
    global_id: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    bkgd_mask: np.array
    bound_mask: np.array
    gray_image: np.array
    big_pose_world_vertex: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        cam_info = CameraInfo(uid=uid, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, fx=focal_length_x, fy=focal_length_y)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('_')[-1]))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    if train_list is not None:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
        print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            print(f"xyz {xyz.shape}")
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


#### dna
def readCamerasdna(path, output_view, white_background, image_scaling=0.5, split='train',
                              novel_view_vis=False, debug_data=False, desire_size=1024, n_views=36):
    frame_id = path.split('/')[-1]
    path = os.path.dirname(path)

    smc_rgb = SMCReader(path+'.smc')
    smc_annot = SMCReader(path.replace('main', 'annotations') + '_annots.smc')


    cam_infos = []
    cam_list = list(range(smc_rgb.Camera_5mp_info['num_device']))
    if split == 'train':
        pose_num = len(cam_list)
        if debug_data:
            pose_num = 10 # debug
    elif split == 'test':
        pose_num = n_views
        if debug_data:
            pose_num = 1 # debug

    ###
    from smplx import SMPLX
    smplx_zoo = {
        'neutral': SMPLX(
                '/ssd2/jhp/Tools/smplx', smpl_type='smplx',
                gender='neutral', use_face_contour=True, flat_hand_mean=False, use_pca=False,
                num_betas=10, num_expression_coeffs=10, ext='npz'),
        'male': SMPLX(
            '/ssd2/jhp/Tools/smplx', smpl_type='smplx',
            gender='male', use_face_contour=True, flat_hand_mean=False, use_pca=False,
            num_betas=10, num_expression_coeffs=10, ext='npz'),
        'female': SMPLX(
            '/ssd2/jhp/Tools/smplx', smpl_type='smplx',
            gender='female', use_face_contour=True, flat_hand_mean=False, use_pca=False,
            num_betas=10, num_expression_coeffs=10, ext='npz'),
    }
    gender = 'neutral'
    # with open('assets/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
    #     idxs_data = pickle.load(f)
    # left_hand = idxs_data['left_hand']
    # right_hand = idxs_data['right_hand']
    # hand_idx = np.concatenate([left_hand, right_hand])
    # face_idxs = np.load('assets/SMPL-X__FLAME_vertex_ids.npy')
    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['global_orient'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['betas'] = np.zeros((1, 10)).astype(np.float32)
    big_pose_smpl_param['body_pose'] = np.zeros((1, 63)).astype(np.float32)
    big_pose_smpl_param['jaw_pose'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['left_hand_pose'] = np.zeros((1, 45)).astype(np.float32)
    big_pose_smpl_param['right_hand_pose'] = np.zeros((1, 45)).astype(np.float32)
    big_pose_smpl_param['leye_pose'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['reye_pose'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['expression'] = np.zeros((1, 10)).astype(np.float32)
    big_pose_smpl_param['transl'] = np.zeros((1, 3)).astype(np.float32)
    big_pose_smpl_param['body_pose'][0, 2] = 45 / 180 * np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 5] = -45 / 180 * np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 20] = -30 / 180 * np.array(np.pi)
    big_pose_smpl_param['body_pose'][0, 23] = 30 / 180 * np.array(np.pi)

    # cameras, camera_dict = init_camera_x(nviews=n_views, size=desire_size)

    idx = 0
    cam_param_dict_all = {}
    for pose_index in tqdm(range(pose_num)):
        if novel_view_vis:
            view_index_look_at = view_index
            view_index = 0

        # Load image, mask, K, D, R, T
        rgb = smc_rgb.get_img('Camera_5mp', str(pose_index), 'color', frame_id)[None]
        mask = smc_annot.get_mask(str(pose_index), frame_id)
        cam_params = smc_annot.get_Calibration(str(pose_index))
        camera_parameter = FisheyeCameraParameter(name=str(pose_index))
        K = cam_params['K']
        D = cam_params['D']  # k1, k2, p1, p2, k3
        RT = cam_params['RT']
        extrinsic = cam_params['RT']
        r_mat_inv = extrinsic[:3, :3]
        r_mat = np.linalg.inv(r_mat_inv)
        t_vec = extrinsic[:3, 3:]
        t_vec = -np.dot(r_mat, t_vec).reshape((3))
        R = r_mat
        T = t_vec
        dist_coeff_k = [D[0], D[1], D[4]]
        dist_coeff_p = D[2:4]
        camera_parameter.set_KRT(K, R, T)
        camera_parameter.set_dist_coeff(dist_coeff_k, dist_coeff_p)
        camera_parameter.inverse_extrinsic()
        camera_parameter.set_resolution(rgb.shape[1], rgb.shape[2])

        corrected_cam, corrected_img = undistort_images(camera_parameter, rgb)
        _, corrected_mask = undistort_images(camera_parameter, mask[None])

        K = np.asarray(corrected_cam.get_intrinsic())
        R = np.asarray(corrected_cam.get_extrinsic_r())
        T = np.asarray(corrected_cam.get_extrinsic_t())

        K_new, R_new, T_new = convert_camera_matrix(
            convention_dst='opengl',
            K=K,
            R=R,
            T=T,
            is_perspective=True,
            convention_src='opencv',
            resolution_src=(2448, 2048),
            in_ndc_src=False,
            in_ndc_dst=False)

        image = np.array(corrected_img[0].astype(np.float32) / 255.)
        # msk = corrected_mask / 255.0
        from scipy.ndimage import label
        labeled_mask, num_features = label(corrected_mask)
        largest_component = np.zeros_like(labeled_mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled_mask.flat)
            largest_component_index = np.argmax(component_sizes[1:]) + 1
            largest_component = (labeled_mask == largest_component_index).astype(np.uint8)
        processed_mask = largest_component[0]
        image = image[...,::-1] * processed_mask[:, :, None]
        gray_image = (0.299 * image[...,0] + 0.587 * image[...,1] + 0.114 * image[...,2])[None]
        msk = processed_mask

        K_new = np.array([
            [K_new[0][0][0], 0, K_new[0][0][2]],
            [0, K_new[0][1][1], K_new[0][1][2]],
            [0, 0, 1]
        ])


        # image_path = os.path.join(path, 'images_lr', cam_list[pose_index], frame_name + '.jpg')
        # image = np.array(imageio.imread(image_path).astype(np.float32) / 255.)
        # msk_path = os.path.join(path, 'fmask_lr', cam_list[pose_index], frame_name + '_fmask.png')
        # msk = cv2.imread(msk_path) / 255.0

        if split == 'train':
            K, R, T = K_new, R_new[0], T_new[0][:,None]
            cam_param_dict_all[pose_index] = {
                'K': K,
                'R': R,
                'T': T
            }
            RT = np.concatenate([R, T], axis=-1)
            w2c = np.concatenate([RT, np.array([[0, 0, 0, 1]])])
            w2c_tensor = torch.from_numpy(w2c).float()
            w2c_tensor = opengl_to_cv2.unsqueeze(0) @ w2c_tensor
            R = w2c_tensor[0][:3, :3]
            T = w2c_tensor[0][:3, 3:]
        # else:
        #     image = image * msk[...,None]
        #     image = cv2.resize(image, (desire_size, desire_size), interpolation=cv2.INTER_AREA)
        #     R = camera_dict['R'][pose_index]
        #     T = camera_dict['t'][pose_index]
        #     RT = np.concatenate([R, T], axis=-1)
        #     w2c = np.concatenate([RT, np.array([[0, 0, 0, 1]])])
        #     w2c_tensor = torch.from_numpy(w2c).float()
        #     w2c_tensor = opengl_to_cv2.unsqueeze(0) @ w2c_tensor
        #     w2c_tensor = w2c_tensor @ opengl_to_cv2.unsqueeze(0)
        #     w2c_renewed = w2c_tensor.numpy()[0]
        #     R = w2c_renewed[:3, :3]
        #     T = w2c_renewed[:3, 3:]
        #     fx, fy, cx, cy = camera_dict['intr']['fx'][0], camera_dict['intr']['fy'][0], camera_dict['intr']['cx'][0], camera_dict['intr']['cy'][0]
        #     K = np.array([
        #         [fx, 0, cx],
        #         [0, fy, cy],
        #         [0, 0, 1]
        #     ])

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3:4] = T
        # get the world-to-camera transform and set R, T
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # Reduce the image resolution by ratio, then remove the back ground
        ratio = image_scaling
        # ratio = 1
        if ratio != 1.:
            H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * ratio

        image1 = Image.fromarray(np.array(image * 255.0, dtype=np.byte), "RGB")

        focalX = K[0, 0]
        focalY = K[1, 1]
        if split == 'train':
            FovX = focal2fov(focalX, image1.size[0])
            FovY = focal2fov(focalY, image1.size[1])
        else:
            FovX = focal2fov(focalX, desire_size)
            FovY = focal2fov(focalY, desire_size)

        gender = smc_rgb.actor_info['gender']
        smplx_dict = smc_annot.get_SMPLx(Frame_id=frame_id)
        betas = torch.from_numpy(smplx_dict["betas"]).unsqueeze(0).float()
        expression = torch.from_numpy(smplx_dict["expression"]).unsqueeze(0).float()
        fullpose = torch.from_numpy(smplx_dict["fullpose"]).unsqueeze(0).float()
        translation = torch.from_numpy(smplx_dict['transl']).unsqueeze(0).float()
        output = smplx_zoo[gender](
            betas=betas,
            expression=expression,
            global_orient=fullpose[:, 0].clone(),
            body_pose=fullpose[:, 1:22].clone(),
            jaw_pose=fullpose[:, 22].clone(),
            leye_pose=fullpose[:, 23].clone(),
            reye_pose=fullpose[:, 24].clone(),
            left_hand_pose=fullpose[:, 25:40].clone(),
            right_hand_pose=fullpose[:, 40:55].clone(),
            transl=translation,
            return_verts=True)
        xyz = output.vertices.detach().cpu().numpy().squeeze()

        def upsample_mesh(verts, faces, target_multiple=3.0, max_iterations=3):
            """
            使用Loop细分对网格进行上采样 (仅需在内存中处理)。

            参数：
                verts           : numpy数组，形状(N, 3)，表示顶点坐标
                faces           : numpy数组，形状(M, 3)，表示三角面索引
                target_multiple : 目标顶点倍数，例如3.0表示约3倍原始顶点数
                max_iterations  : 最大迭代次数，防止顶点数过度增长

            返回：
                upsampled_verts : 上采样后的顶点坐标（numpy数组，形状不定）
                upsampled_faces : 上采样后的三角面索引（numpy数组，形状不定）
            """
            import open3d as o3d
            # 构造Open3D网格
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()  # 可选，若需要法线信息

            # 原始顶点数
            original_vertex_count = len(mesh.vertices)
            print(f"原始顶点数: {original_vertex_count}")

            current_multiple = 1.0
            iteration = 0
            current_mesh = mesh

            # 循环细分，直到达到目标倍数或超过最大迭代次数
            while current_multiple < target_multiple and iteration < max_iterations:
                iteration += 1

                # 对网格进行一次Loop细分
                current_mesh = current_mesh.subdivide_loop(1)

                new_vertex_count = len(current_mesh.vertices)
                current_multiple = new_vertex_count / original_vertex_count
                print(f"第 {iteration} 次细分后: 顶点数 = {new_vertex_count}, "
                      f"约为原始网格的 {current_multiple:.2f} 倍")

            # 转回 numpy 数组
            upsampled_verts = np.asarray(current_mesh.vertices)
            upsampled_faces = np.asarray(current_mesh.triangles)

            return upsampled_verts, upsampled_faces

        # xyz = upsample_mesh(xyz, smplx_zoo[gender].faces)[0]


        # file_path = os.path.join(path, 'smplx/smpl', '{:06d}.json'.format(int(int(frame_name[:4]) / 5 - 1)))
        # with open(file_path, 'r') as f:
        #     smplx_data = json.load(f)
        # if isinstance(smplx_data, dict):
        #     smplx_data = smplx_data['annots']
        # smplx_param = []
        # for data in smplx_data:
        #     for key in ['Rh', 'Th', 'poses', 'handl', 'handr', 'shapes', 'expression', 'keypoints3d']:
        #         if key in data.keys():
        #             data[key] = torch.from_numpy(np.array(data[key], dtype=np.float32))
        #     # for smplx results
        #     smplx_param.append(data)
        # smpl_param = smplx_param[0]
        # body_model_output = smplx_model(return_verts=True, **smpl_param)
        # xyz = body_model_output[0].numpy()

        # from nosmpl.vis.vis_o3d import vis_mesh_o3d
        # vertices = body_model_output.vertices.squeeze()
        # faces = smplx_model.faces.astype(np.int32)
        # vis_mesh_o3d(vertices.detach().cpu().numpy(), faces)
        # vis_mesh_o3d(xyz, faces)
        # ##
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # proj_2d = project(xyz, K, cameras_gt[cam_list[pose_index]]['RT'])
        # plt.scatter(proj_2d[:, 0], proj_2d[:, 1])
        # camera_position = -np.matmul(RT[:3, :3].transpose(0, 1), RT[:3, 3])
        # print(camera_position, np.linalg.norm(camera_position))
        # plt.show()

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)

        # get bounding mask and background mask
        bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image1.size[1], image1.size[0])
        bound_mask = Image.fromarray(np.array(bound_mask * 255.0, dtype=np.byte))

        try:
            bkgd_mask = Image.fromarray(np.array(msk * 255.0, dtype=np.byte))
        except:
            bkgd_mask = Image.fromarray(np.array(msk[:,:,0] * 255.0, dtype=np.byte))

        cam_infos.append(CameraInfo_dna(
                                    uid=1,
                                    global_id=pose_index,
                                    R=R, T=T,
                                    fx=focalX, fy=focalY,
                                    FovY=FovY, FovX=FovX,
                                    image_path='',
                                    image_name=str(pose_index),
                                    image=image.transpose(2,0,1),
                                    gray_image=gray_image,
                                    bkgd_mask=bkgd_mask,
                                    bound_mask=bound_mask,
                                    width=image1.size[0],
                                    height=image1.size[1],
                                    # smpl_param=big_pose_smpl_param,
                                    # world_vertex=xyz,
                                    # world_bound=world_bound,
                                    # big_pose_smpl_param=big_pose_smpl_param,
                                    big_pose_world_vertex=xyz,
                                    # big_pose_world_bound=world_bound,
                                    # face_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                    # lhand_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                    # rhand_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                    # face_render_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                    # lhand_render_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                    # rhand_render_mask=Image.fromarray(np.array(bkgd_mask, dtype=np.byte)),
                                    # kp2d=None,
                                    # depth=None,
                                    ))

        idx += 1
    return cam_infos



def readdnaInfo(path, images, eval, llffhold=8):
    train_view = [0]
    test_view = [0]

    white_background = False
    image_scaling = 1.0
    debug_data = False

    print("Reading Training Transforms")
    train_cam_infos = readCamerasdna(path, train_view, white_background, split='train', image_scaling=image_scaling, debug_data=debug_data)

    print("Reading Test Transforms")
    # test_cam_infos = readCamerasdna(path, test_view, white_background, split='test', novel_view_vis=False, image_scaling=image_scaling, debug_data=debug_data)
    test_cam_infos = []

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    # ply_path = os.path.join('/scratch/10102/hh29499/GS_fitting', output_path, "points3d.ply") # cluster
    os.makedirs('output', exist_ok=True)
    ply_path = os.path.join('output', "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        # num_pts = 10475  # 100_000
        num_pts = train_cam_infos[0].big_pose_world_vertex.shape[0]
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DNA": readdnaInfo,
}