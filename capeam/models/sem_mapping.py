import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase
import envs.utils.depth_utils as du

import cv2
import time



class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3 #TODO: add argument
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.print_time = args.print_time
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.no_straight_obs = args.no_straight_obs
        self.view_angles = [0.0]*args.num_processes

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height*100.
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi/2.0]
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(args.num_processes, 1 + self.num_sem_categories, vr, vr,
                                self.max_height - self.min_height).float().to(self.device)
        self.feat = torch.ones(args.num_processes, 1 + self.num_sem_categories,
                          self.screen_h//self.du_scale * self.screen_w//self.du_scale
                         ).float().to(self.device)
        
    def set_view_angles(self, view_angles):
        self.view_angles = [-view_angle for view_angle in view_angles]


    def forward(self, obs, pose_obs, maps_last, poses_last, build_maps=True, no_update = False):
        bs, c, h, w = obs.size()
        depth = obs[:,3,:,:]
        # ###############################################################################################################
        # import matplotlib.pyplot as plt
        # rgb = obs[0,:3].clone().detach().cpu().permute(1,2,0).numpy().astype(np.uint8)
        # dd = obs[0,3].clone().detach().cpu().numpy()
        # plt.subplot(121); plt.imshow(rgb)
        # plt.subplot(122); plt.imshow(dd)
        # plt.show()
        # ###############################################################################################################

        ###############################################################################################################
        # Point Cloud Projection

        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        ###############################################################################################################
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud_t.reshape(-1, 3).detach().cpu().numpy())
        # pcd.estimate_normals()
        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(100)
        # o3d.visualization.draw_geometries([pcd, coord])
        ###############################################################################################################
        
        #Multiprocessing

        agent_view_t = du.transform_camera_view_t_multiple(point_cloud_t, self.agent_height, self.view_angles, self.device)
        ###############################################################################################################
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(agent_view_t.reshape(-1, 3).detach().cpu().numpy())
        # pcd.estimate_normals()
        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(100)
        # o3d.visualization.draw_geometries([pcd, coord])
        ###############################################################################################################

        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)
        ###############################################################################################################
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(agent_view_centered_t.reshape(-1, 3).detach().cpu().numpy())
        # pcd.estimate_normals()
        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(100)
        # o3d.visualization.draw_geometries([pcd, coord])
        ###############################################################################################################

        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[...,:2] / self.resolution - self.vision_range // 2.) / self.vision_range * 2.
        XYZ_cm_std[...,  2] = (XYZ_cm_std[...,2] / self.z_resolution - (self.max_height + self.min_height) // 2.) / (self.max_height - self.min_height) * 2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(obs[:,4:,:,:]).view(
            bs, c - 4, h // self.du_scale * w // self.du_scale
        )
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0], XYZ_cm_std.shape[1], XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])
        ###############################################################################################################

        ###############################################################################################################
        # Voxelization
        voxels = du.splat_feat_nd(self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)
        # import matplotlib.pyplot as plt
        # _voxels = voxels[0].sum(dim=0).cpu().numpy()
        # for z in range(80):
        #     plt.imshow(_voxels[:,:,z])
        #     plt.savefig('visualize_voxels/{:03d}.png'.format(z))
        #     plt.clf()
        ###############################################################################################################

        min_z = int(5 / self.z_resolution - self.min_height)
        max_z = int((self.agent_height + 1 + 50) / self.z_resolution - self.min_height)

        fp_map_pred = voxels[...,min_z:max_z]
        fp_map_pred = fp_map_pred.sum(4)[:, :1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = voxels.sum(4)[:, :1, :, :]
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        if self.no_straight_obs:
            for vi, va in enumerate(self.view_angles):
                if abs(va - 0) <= 5:
                    fp_map_pred[vi, :, :, :] = 0.0

        agent_view = torch.zeros(
            bs, c, self.map_size_cm // self.resolution, self.map_size_cm // self.resolution
        ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4: , y1:y2, x1:x2] = torch.clamp(
            voxels[..., min_z:max_z].sum(4)[:, 1:, :, :] / self.cat_pred_threshold,
        min=0.0, max=1.0)

        if self.cat_pred_threshold > 5.0:
            agent_view[:, 4:, y1:y2, x1:x2][np.where(agent_view[:, 4:, y1:y2, x1:x2].cpu().detach().numpy() < 0.5)] = 0.0
        
        if no_update:
            agent_view = torch.zeros_like(agent_view) #torch.zeros(agent_view.shape).to(self.device)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):
            pose[:,1] += rel_pose_change[:,0] * torch.sin(pose[:,2]/57.29577951308232) + \
                         rel_pose_change[:,1] * torch.cos(pose[:,2]/57.29577951308232)
            pose[:,0] += rel_pose_change[:,0] * torch.cos(pose[:,2]/57.29577951308232) - \
                         rel_pose_change[:,1] * torch.sin(pose[:,2]/57.29577951308232)
            pose[:,2] += rel_pose_change[:,2] * 57.29577951308232

            pose[:,2] = torch.fmod(pose[:,2] - 180.0, 360.0) + 180.0
            pose[:,2] = torch.fmod(pose[:,2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution - self.map_size_cm // (self.resolution * 2)) / \
                         (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - st_pose[:, 2]
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        map_pred, _ = torch.max(torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1), 1)


        pose_pred = poses_last

        return fp_map_pred, map_pred, pose_pred, current_poses, translated
