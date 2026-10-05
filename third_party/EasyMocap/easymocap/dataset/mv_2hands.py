'''
  @ Date: 2021-01-12 17:12:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-27 20:25:24
  @ FilePath: /EasyMocap/easymocap/dataset/mv1pmf.py
'''
from ..mytools.file_utils import get_bbox_from_pose
from os.path import join
import numpy as np
from os.path import join
from .base_2hands import MVBase_Hand

class MV1P2H(MVBase_Hand):
    # stands for 1 person with 2 hands
    def __init__(self, root, cams=[], pid=0, out=None, config={}, 
        image_root='images', annot_root='annots', annot_file='annot_hand.pickle', kpts_type='body15',
        undis=True, no_img=False,  start_frame="00000.jpg", end_frame="last", verbose=False, cam_as_world=None) -> None:
        super().__init__(root=root, cams=cams, out=out, config=config, 
            image_root=image_root, annot_root=annot_root, annot_file=annot_file, 
            kpts_type=kpts_type, undis=undis, no_img=no_img, start_frame=start_frame, end_frame=end_frame, cam_as_world=cam_as_world)
        self.pid = pid
        self.verbose = verbose

    def write_keypoints3d(self, keypoints3d, nf):
        results = [{'id': self.pid, 'keypoints3d': keypoints3d}]
        super().write_keypoints3d(results, nf)

    def write_smpl(self, params, nf, mode='smpl'):
        result = {'id': 0}
        result.update(params)
        super().write_smpl([result], nf, mode)

    def vis_smpl(self, vertices, faces, images, nf, sub_vis=[], 
        mode='smpl', extra_data=[], add_back=True):
        outname = join(self.out, 'smpl', '{:06d}.jpg'.format(nf))
        render_data = {}
        assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
        pid = self.pid
        render_data[pid] = {'vertices': vertices, 'faces': faces, 
            'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        cameras = {'K': [], 'R':[], 'T':[]}
        if len(sub_vis) == 0:
            sub_vis = self.cams
        for key in cameras.keys():
            cameras[key] = np.stack([self.cameras[cam][key] for cam in sub_vis])
        images = [images[self.cams.index(cam)] for cam in sub_vis]
        self.writer.vis_smpl(render_data, images, cameras, outname, add_back=add_back)
    
    def vis_detections(self, images, annots, nf, to_img=True, sub_vis=[], onehand=False, valid_note=None):
        lDetections = []
        for nv in range(len(images)):
            det = {
                'id': self.pid,
                'bbox': annots['bbox'][nv],
                'keypoints2d': annots['keypoints'][nv]
            }
            lDetections.append([det])
        return super().vis_detections(images, lDetections, nf, sub_vis=sub_vis, onehand=onehand, valid_note=valid_note)

    def vis_repro(self, images, kpts_repro, nf, to_img=True, sub_vis=[], mode='repro', onehand=False, kp_3d=None, valid_note=None, vis_3d_only=False, no_grid=False, transparent=False):
        lDetections = []
        valid_idx=0
        for nv in range(len(images)):
            if valid_note[nv] and kpts_repro is not None:
                det = {
                    'id': -1,
                    'keypoints2d': kpts_repro[valid_idx],
                    'bbox': get_bbox_from_pose(kpts_repro[valid_idx], images[nv])
                }
                valid_idx += 1
            else:
                det = {
                    'id': -1,
                    'keypoints2d': None,
                    'bbox': None
                }
            lDetections.append([det])
        return super().vis_detections(images, lDetections, nf, mode=mode, sub_vis=sub_vis, onehand=onehand, kp_3d=kp_3d, valid_note=valid_note, vis_3d_only=vis_3d_only, no_grid=no_grid, transparent=transparent)

    def __getitem__(self, index: int):
        images, annots_all, img_idx = super().__getitem__(index)
        annots_left, valid_left = self.select_hand(annots_all, index, "Left")
        annots_right, valid_right = self.select_hand(annots_all, index, "Right")   # due to unknown reason, mediapipe recognize right hand and left hand label wrongly.

        return images, img_idx, annots_left, annots_right, valid_left, valid_right   # annots_left: [num_cams x num_kps x 2]



class MV1P1H(MVBase_Hand):
    # stands for 1 person with 2 hands
    def __init__(self, root, cams=[], pid=0, out=None, config={}, 
        image_root='images', annot_root='annots', annot_file='handpose.pkl', kpts_type='body15',
        undis=True, no_img=False,  start_frame="00000.jpg", end_frame="last", handness='left', verbose=False) -> None:
        super().__init__(root=root, cams=cams, out=out, config=config, 
            image_root=image_root, annot_root=annot_root, annot_file=annot_file, 
            kpts_type=kpts_type, undis=undis, no_img=no_img, start_frame=start_frame, end_frame=end_frame)
        self.pid = pid
        self.verbose = verbose
        assert handness in ['left', 'right']
        self.handness = handness

    def write_keypoints3d(self, keypoints3d, nf):
        results = [{'id': self.pid, 'keypoints3d': keypoints3d}]
        super().write_keypoints3d(results, nf)

    def write_smpl(self, params, nf, mode='smpl'):
        result = {'id': 0}
        result.update(params)
        super().write_smpl([result], nf, mode)

    def vis_smpl(self, vertices, faces, images, nf, sub_vis=[], 
        mode='smpl', extra_data=[], add_back=True):
        outname = join(self.out, 'smpl', '{:06d}.jpg'.format(nf))
        render_data = {}
        assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
        pid = self.pid
        render_data[pid] = {'vertices': vertices, 'faces': faces, 
            'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        cameras = {'K': [], 'R':[], 'T':[]}
        if len(sub_vis) == 0:
            sub_vis = self.cams
        for key in cameras.keys():
            cameras[key] = np.stack([self.cameras[cam][key] for cam in sub_vis])
        images = [images[self.cams.index(cam)] for cam in sub_vis]
        self.writer.vis_smpl(render_data, images, cameras, outname, add_back=add_back)
    
    def vis_detections(self, images, annots, nf, to_img=True, sub_vis=[], onehand=False):
        lDetections = []
        for nv in range(len(images)):
            det = {
                'id': self.pid,
                'bbox': annots['bbox'][nv],
                'keypoints2d': annots['keypoints'][nv]
            }
            lDetections.append([det])
        return super().vis_detections(images, lDetections, nf, sub_vis=sub_vis, onehand=onehand)

    def vis_repro(self, images, kpts_repro, nf, to_img=True, sub_vis=[], mode='repro', onehand=False, kp_3d=None):
        lDetections = []
        for nv in range(len(images)):
            det = {
                'id': -1,
                'keypoints2d': kpts_repro[nv],
                'bbox': get_bbox_from_pose(kpts_repro[nv], images[nv])
            }
            lDetections.append([det])
        return super().vis_detections(images, lDetections, nf, mode=mode, sub_vis=sub_vis, onehand=onehand, kp_3d=kp_3d)

    def __getitem__(self, index: int):
        images, annots_all = super().__getitem__(index)
        query = 'Right' if self.handness=='left' else 'Left'
        annots = self.select_hand(annots_all, index, query)
        return images, annots

if __name__ == "__main__":
    root = '/home/qian/zjurv2/mnt/data/ftp/Human/vis/lightstage/CoreView_302_sync/'
    dataset = MV1P2H(root)
    images, annots = dataset[0]