import argparse
import glob
import json
import os
import os.path as osp

import cv2
import ipdb
import numpy as np
import torch
import trimesh
from ezc3d import c3d as ezc3d
from tqdm import tqdm

from moyo.utils.constants import frame_select_dict_combined as frame_select_dict
from moyo.utils.misc import colors, copy2cpu as c2c

from xrprimer.data_structure import Keypoints
from xrprimer.data_structure.camera import FisheyeCameraParameter

def main(img_folder, c3d_folder, model_folder, output_dir, cam_folders, frame_offset, split, downsample_factor):
    # presented poses
    c3d_folder = os.path.join(c3d_folder, split, 'c3d')
    c3d_names = os.listdir(c3d_folder)
    c3d_names = [os.path.splitext(x)[0] for x in c3d_names if x[0] != '.' and '.c3d' in x]
    c3d_names = sorted(c3d_names)

    img_folder = os.path.join(img_folder, split)
    # list all folders in img_folder
    img_pose_folders = os.listdir(img_folder)
    img_pose_folders = [item for item in img_pose_folders if os.path.isdir(os.path.join(img_folder, item))]

    for i, c3d_name in enumerate(tqdm(c3d_names)):
        print(f'Processing {i}: {c3d_name}')
        # load images, loop through and read smplx and keypoints
        pose_name = '-'.join('_'.join(c3d_name.split('_')[5:]).split('-')[:-1])
        if pose_name == "Happy_Baby_Pose":
            pose_name = "Happy_Baby_Pose_or_Ananda_Balasana_"  # this name was changed during mocap

        # get soma fit
        try:
            c3d_path = glob.glob(osp.join(c3d_folder, f'{c3d_name}.c3d'))[0]
        except:
            print(f'{c3d_folder}/{c3d_name}_stageii.pkl does not exist. SKIPPING!!!')
            continue

        c3d = ezc3d(c3d_path)

        markers3d = c3d['data']['points'].transpose(2, 1,
                                                    0)  # Frames x NumPoints x 4 (x,y,z,1) in homogenous coordinates

        try:
            c3d_var = '_'.join(c3d_name.split('_')[5:])
            selected_frame = frame_select_dict[c3d_var]
        except:
            print(f'{c3d_var} does not exist in frame_selection_dict. SKIPPING!!!')
            continue

        j3d = markers3d[selected_frame]
        
        #markers are for higher fps
        mymarkers = markers3d[2:,:,:]

        # find out, which indices we need?
        # maybe do this in config

        kps_shape = np.zeros(4)
        #campus convention?
        kps_shape[1]=4
        kps_shape[[0,2,3]]=mymarkers.shape
        kps = np.zeros[kps_shape]
        mask = np.zeros(kps_shape[:-1])
        mask[:,1,:]=1


        pass
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True,
                        help='folder_containing_ioi_data')
    parser.add_argument('--c3d_folder', required=True,
                        help='folder containing raw c3d mocap files')
    parser.add_argument('--cam_folder_first', required=True,
                        help='folder containing camera matrix for first session 210727')
    parser.add_argument('--cam_folder_second', required=True,
                        help='folder containing camera matrix for second session 211117')
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'],
                        help='split to process')
    parser.add_argument('--model_folder', required=False, default='/ps/project/common/smplifyx/models/',
                        help='path to SMPL/SMPL-X model folder')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--downsample_factor', type=float, default=0.5,
                        help='Downsample factor for the images. Release images are downsampled by 0.5')
    parser.add_argument('--frame_offset', type=int,
                        help='To get the frame offset, divide the vicon fnum by 2 and add this offset')

    args = parser.parse_args()

    cam_folders = {'220923': args.cam_folder_first, '220926': args.cam_folder_second}

    main(args.img_folder, args.c3d_folder, args.model_folder, args.output_dir, cam_folders, args.frame_offset,
         args.split, args.downsample_factor)
