# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
print("updated.")

from dataset.JointsDataset import JointsDataset
from utils.transforms import projectPoints

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    #'160906_band3',
]
VAL_LIST = ['160906_pizza1']#, '160422_haggling1', '160906_ian5', '160906_band4']

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]

colors = ['r', 'g', 'b', 'y', 'orange', 'brown', 'purple']

class Panoptic(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)
        self.ap_thresholds = np.arange(5, 65, 10)

        if self.image_set == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 3
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][:self.num_views]
            # self.cam_list = list(set([(0, n) for n in range(0, 31)]) - {(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)})
            # self.cam_list.sort()
            self.num_views = len(self.cam_list)
        elif self.image_set == 'validation':
            self.sequence_list = VAL_LIST
            self._interval = 12
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][:self.num_views]
            self.num_views = len(self.cam_list)

        self.db_file = 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            #assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            assert info['cam_list'] == self.cam_list
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'cam_list': self.cam_list,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        # self.db = self._get_db()
        self.db_size = len(self.db)

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        for seq in self.sequence_list:

            cameras = self._get_cam(seq)

            curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

            for i, file in enumerate(anno_files):
                if i % self._interval == 0:
                    with open(file) as dfile:
                        bodies = json.load(dfile)['bodies']
                    if len(bodies) == 0:
                        continue

                    for k, v in cameras.items():
                        postfix = osp.basename(file).replace('body3DScene', '')
                        prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                        image = osp.join(seq, 'hdImgs', prefix,
                                         prefix + postfix)
                        image = image.replace('json', 'jpg')

                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        for body in bodies:
                            pose3d = np.array(body['joints19']).reshape((-1, 4))
                            pose3d = pose3d[:self.num_joints]

                            joints_vis = pose3d[:, -1] > 0.1

                            if not joints_vis[self.root_id]:
                                continue

                            # Coordinate transformation
                            M = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                            all_poses_3d.append(pose3d[:, 0:3] * 10.0)
                            all_poses_vis_3d.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                            pose2d = np.zeros((pose3d.shape[0], 2))
                            pose2d[:, :2] = projectPoints(
                                pose3d[:, 0:3].transpose(), v['K'], v['R'],
                                v['t'], v['distCoef']).transpose()[:, :2]
                            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= width - 1)
                            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                     pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0

                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 2, axis=1))

                        if len(all_poses_3d) > 0:
                            our_cam = {}
                            our_cam['R'] = v['R']
                            our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
                            our_cam['fx'] = np.array(v['K'][0, 0])
                            our_cam['fy'] = np.array(v['K'][1, 1])
                            our_cam['cx'] = np.array(v['K'][0, 2])
                            our_cam['cy'] = np.array(v['K'][1, 2])
                            our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                            our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)

                            db.append({
                                'key': "{}_{}{}".format(seq, prefix, postfix.split('.')[0]),
                                'image': osp.join(self.dataset_root, image),
                                'joints_3d': all_poses_3d,
                                'joints_3d_vis': all_poses_vis_3d,
                                'joints_2d': all_poses,
                                'joints_2d_vis': all_poses_vis,
                                'camera': our_cam
                            })
        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        # if self.image_set == 'train':
        #     # camera_num = np.random.choice([5], size=1)
        #     select_cam = np.random.choice(self.num_views, size=5, replace=False)
        # elif self.image_set == 'validation':
        #     select_cam = list(range(self.num_views))

        for k in range(self.num_views):
            i, t, w, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds):
        eval_list = []
        log_data = []
        gt_num = self.db_size // self.num_views
        #print("GT NUM: %d" % gt_num)
        #print("PRED SIZE %d" % len(preds))
        #assert len(preds) == gt_num, 'number mismatch'
        #print("len preds: %d " % len(preds))
        total_gt = 0

        for i in range(len(preds)):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']
            # Try to plot the images from the DB.
            image_paths = [db_entry['image'] for db_entry in self.db[index:index+self.num_views]]
            #print(image_paths[0])
            
            # fig = plt.figure(figsize=(self.num_views * 3, self.num_views * 2))
            # gs = gridspec.GridSpec(3, self.num_views, wspace=0, hspace=0)

            # axes = []
            # for j in range(self.num_views):
            #     axes.append(fig.add_subplot(gs[0, j]))
            # axes.append(fig.add_subplot(gs[1:, :], projection='3d'))

            # axes[-1].set_zlim(0, 1500)
            # axes[-1].set_ylim(-1250, 1000)
            # axes[-1].set_xlim(-1000, 1500)

            # # Plot the relevant images.
            # for cam_idx in range(self.num_views):
            #     #print(cam_idx)
            #     image = cv2.imread(image_paths[cam_idx])
            #     axes[cam_idx].imshow(image[:, :, ::-1])
            #     axes[cam_idx].axis("off")
            #     axes[cam_idx].margins(x=0, y=0)

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            gts = []
            all_mpjpes = []
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)

                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                gts.append(joints_3d[min_gt])
                all_mpjpes.append(min_mpjpe)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })
                # Compute plot
            #     for j1, j2 in LIMBS:
            #         x_gt = [float(joints_3d[min_gt][j1, 0]), float(joints_3d[min_gt][j2, 0])]
            #         y_gt = [float(joints_3d[min_gt][j1, 1]), float(joints_3d[min_gt][j2, 1])]
            #         z_gt = [float(joints_3d[min_gt][j1, 2]), float(joints_3d[min_gt][j2, 2])]
            #         axes[-1].plot(x_gt, y_gt, z_gt, c=colors[min_gt], ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
            #                 markeredgewidth=1)
            #         x = [float(pose[j1, 0]), float(pose[j2, 0])]
            #         y = [float(pose[j1, 1]), float(pose[j2, 1])]
            #         z = [float(pose[j1, 2]), float(pose[j2, 2])]
            #         axes[-1].plot(x, y, z, lw=1.5, marker='o', c=colors[min_gt], markerfacecolor='w', markersize=2,
            #                 markeredgewidth=1)
            log_data_entry = {}
            log_data_entry['frame_idx'] = i
            log_data_entry['image_paths'] = image_paths
            log_data_entry['pred'] = pred.tolist()
            log_data_entry['gt'] = gts
            log_data_entry['mpjpes'] = all_mpjpes
            log_data.append(log_data_entry)
            # plt.savefig(osp.join("video_viz_panoptic", "image_%d.png" % i))
            # plt.close(fig)
            # total_gt += len(joints_3d)
        logging.info("Finished sequence evaluation, saving to JSON.")
        with open("test_dump_panoptics.json", 'w') as f:
            json.dump(log_data, f)
        logging.info("Succesffuly saved to JSON.")
        aps = []
        recs = []
        for t in self.ap_thresholds:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return aps, recs, self._eval_list_to_mpjpe(eval_list, use_threshold=False), self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500, use_threshold=True):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            threshold_bool = item["mpjpe"] < threshold if use_threshold else True
            if threshold_bool and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt




