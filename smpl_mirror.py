import os
import cv2
import smplx
import torch
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation


def load_smplx_data(data_folder):
    # SMPLX param
    t = np.load(os.path.join(data_folder, 'transl.npy'))
    r = np.load(os.path.join(data_folder, 'orient.npy'))
    p = np.load(os.path.join(data_folder, 'pose.npy'))
    betas = np.load(os.path.join(data_folder, 'betas.npy'))
    return t, r, p, betas


def create_color_array(vlen, color):
    color_arr = np.zeros((vlen, 3), dtype=np.uint8)
    color_arr[:, 0:3] = color
    return color_arr


def creat_smplx_model():
    smplx_model = smplx.create(
        "models_smplx_v1_1_2/models", model_type='smplx',
        gender='male', ext='npz',
        num_pca_comps=10,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=1,
    ).to(device='cpu')
    return smplx_model


def get_smplx_mesh(smplx_model, param, color):
    output = smplx_model(return_verts=True, **param)
    vertices = output.vertices.detach().squeeze(0).numpy()
    faces = smplx_model.faces
    smplx_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=create_color_array(len(vertices), color))
    return smplx_mesh


def visualize_meshes(meshes, save_name):
    S = trimesh.Scene(meshes)
    S.add_geometry(trimesh.creation.axis())
    ground_size = 4.0
    ground = trimesh.primitives.Box(extents=[ground_size, -0.05, ground_size])
    ground.visual.face_colors = [255,255,255,255]
    # ground.apply_translation([-0.167, -0.01, -0.84])
    S.add_geometry(ground, transform=trimesh.transformations.translation_matrix([0, 0, 0]))

    S.set_camera((-np.pi/8, np.pi/2, 0.0), 4.5, [0, 0, 0])
    png = S.save_image(resolution=[1920, 1200])
    if save_name is not None:
        with open(save_name, 'wb') as f:
            f.write(png)

    # S.show()


def swap_left_right(t, r, pose, betas):
    back_bone_chain = [2,5,8,11,14]
    left_chain = [0,3,6,9,12,15,17,19]
    right_chain = [1,4,7,10,13,16,18,20]
    left_hand_chain = [21, 22, 23, 33, 34, 35, 24, 25, 26, 30, 31, 32, 27, 28, 29]
    right_hand_chain = [42, 43, 44, 45, 46, 47, 39, 40, 41, 36, 37, 38, 48, 49, 50]

    m_r = r.copy()
    m_r[:, 1] *= -1
    # m_r[:, 1] = (m_r[:, 1] + 2 * np.pi) % (2 * np.pi) - np.pi
    m_r[:, 2] *= -1

    R_cw = np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                    ])
    for index in range(m_r.shape[0]):
        r = m_r[index]
        R_c = Rotation.from_rotvec(r).as_matrix()
        R_w = np.dot(R_cw, R_c)
        m_r[index] = Rotation.from_matrix(R_w).as_rotvec()


    m_pose = pose.copy().reshape(pose.shape[0], -1, 3)
    m_pose[:, back_bone_chain, 1] *= -1
    m_pose[:, back_bone_chain, 2] *= -1

    tmp = m_pose[:, left_chain]
    m_pose[:, left_chain] = m_pose[:, right_chain]
    m_pose[:, right_chain] = tmp

    m_pose[:, left_chain, 1] *= -1
    m_pose[:, left_chain, 2] *= -1
    m_pose[:, right_chain, 1] *= -1
    m_pose[:, right_chain, 2] *= -1

    # if fingers
    if t.shape[1] > 24:
        tmp = m_pose[:, left_hand_chain]
        m_pose[:, left_hand_chain] = m_pose[:, right_hand_chain]
        m_pose[:, right_hand_chain] = tmp

    m_t = t.copy()
    m_t[:, 2] = -t[:, 2]
    m_betas = betas.copy()
    return m_t, m_r, m_pose, m_betas


def image_to_video(image_dir, video_path, fps):
    # img2video opencv edition
    images = sorted(
        [img for img in os.listdir(image_dir)],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    if not images:
        raise ValueError("No images found in the directory.")

    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_dir, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()


if __name__ == '__main__':
    smplx_model = creat_smplx_model()
    data_dir = './data'

    for motion_id in os.listdir(data_dir):
        t, r, pose, betas = load_smplx_data(os.path.join(data_dir, motion_id))
        m_t, m_r, m_pose, m_betas = swap_left_right(t, r, pose, betas)

        # T_pose = np.zeros(63, dtype=np.float32).reshape(-1, 3)
        # T_pose[14][0] = np.pi/4

        # visualize
        for index in range(0, t.shape[0], 10):
            param = {
                'transl': torch.tensor(t[index]).reshape(1, -1),
                'global_orient': torch.tensor(r[index]).reshape(1, -1),
                'body_pose': torch.tensor(pose[index]).reshape(1, -1),
                # 'body_pose': torch.tensor(T_pose).reshape(1, -1),
                'betas': torch.tensor(betas[index]).reshape(1, -1)
            }
            pose_mesh = get_smplx_mesh(smplx_model, param, [0, 255, 0])

            m_param = {
                'transl': torch.tensor(m_t[index]).reshape(1, -1),
                'global_orient': torch.tensor(m_r[index]).reshape(1, -1),
                'body_pose': torch.tensor(m_pose[index]).reshape(1, -1),
                'betas': torch.tensor(m_betas[index]).reshape(1, -1)
            }
            mirror_pose_mesh = get_smplx_mesh(smplx_model, m_param, [255, 0, 0])

            meshes = [pose_mesh, mirror_pose_mesh]
            image_dir = f'video/{motion_id}'
            if not os.path.exists(f'video/{motion_id}'):
                os.makedirs(f'video/{motion_id}')

            visualize_meshes(meshes, f'video/{motion_id}/{index}.png')

        image_to_video(f'video/{motion_id}', f'video/{motion_id}.mp4', fps=3)






