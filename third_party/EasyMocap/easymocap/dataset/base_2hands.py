'''
  @ Date: 2021-01-13 16:53:55
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 15:53:12
  @ FilePath: /EasyMocapRelease/easymocap/dataset/base.py
  # Modify the original base.py to suit for two hand reconstruction for the surgical task
'''
import os
from os.path import join
from glob import glob
import cv2
import os, sys
import numpy as np
import pickle
from ..mytools.camera_utils import read_camera, get_fundamental_matrix, Undistort, transform_cameras_to_reference
from ..mytools import FileWriter, read_annot, getFileList, save_json
from ..mytools.reader import read_keypoints3d, read_json, read_smpl
from ..mytools.file_utils import merge_params, select_nf, save_annot

def crop_image(img, annot, vis_2d=False, config={}, crop_square=True):
    for det in annot:
        bbox = det['bbox']
        l, t, r, b = det['bbox'][:4]
        if crop_square:
            if b - t > r - l:
                diff = (b - t) - (r - l)
                l -= diff//2
                r += diff//2
            else:
                diff = (r - l) - (b - t)
                t -= diff//2
                b += diff//2
        l = max(0, int(l+0.5))
        t = max(0, int(t+0.5))
        r = min(img.shape[1], int(r+0.5))
        b = min(img.shape[0], int(b+0.5))
        det['bbox'][:4] = [l, t, r, b]
        if vis_2d:
            crop_img = img.copy()
            from easymocap.mytools import plot_keypoints
            plot_keypoints(crop_img, det['keypoints'], pid=det['id'], 
                config=config, use_limb_color=True, lw=2)
        else:
            crop_img = img
        crop_img = crop_img[t:b, l:r, :]
        if crop_square:
            crop_img = cv2.resize(crop_img, (256, 256))
        else:
            crop_img = cv2.resize(crop_img, (128, 256))
        det['crop'] = crop_img
        det['img'] = img
    return 0

class ImageFolder:
    """Dataset for image folders"""
    def __init__(self, root, subs=[], out=None, image_root='images', annot_root='annots', 
        kpts_type='body15', config={}, no_img=False) -> None:
        self.root = root
        self.image_root = join(root, image_root)
        self.annot_root = join(root, annot_root)
        self.kpts_type = kpts_type
        self.no_img = no_img
        if len(subs) == 0:
            self.imagelist = getFileList(self.image_root, '.jpg')
            self.annotlist = getFileList(self.annot_root, '.json')
        else:
            self.imagelist, self.annotlist = [], []
            for sub in subs:
                images = sorted([join(sub, i) for i in os.listdir(join(self.image_root, sub))])
                annots = sorted([join(sub, i) for i in os.listdir(join(self.annot_root, sub))])
                if len(annots) < len(images):
                    print('[WARN] length of annots != lenght of images')
                    images = images[:len(annots)]
                self.imagelist.extend(images)
                self.annotlist.extend(annots)
        self.out = out
        self.writer = FileWriter(self.out, config=config)
        self.gtK, self.gtRT = False, False

    def load_gt_cameras(self):
        cameras = load_cameras(self.root)
        gtCameras = []
        for i, name in enumerate(self.annotlist):
            cam = os.path.dirname(name)
            gtcams = {key:cameras[cam][key].copy() for key in ['K', 'R', 'T', 'dist']}
            gtCameras.append(gtcams)
        self.gtCameras = gtCameras

    def __len__(self) -> int:
        return len(self.imagelist)
        
    def __getitem__(self, index: int):
        imgname = join(self.image_root, self.imagelist[index])
        annname = join(self.annot_root, self.annotlist[index])
        assert os.path.exists(imgname) and os.path.exists(annname), (imgname, annname)
        assert os.path.basename(imgname).split('.')[0] == os.path.basename(annname).split('.')[0], '{}, {}'.format(imgname, annname)
        if not self.no_img:
            img = cv2.imread(imgname)
        else:
            img = None
        annot = read_annot(annname, self.kpts_type)
        return img, annot
    
    def camera(self, index=0, annname=None):
        if annname is None:
            annname = join(self.annot_root, self.annotlist[index])
        data = read_json(annname)
        if 'K' not in data.keys():
            height, width = data['height'], data['width']
            # focal = 1.2*max(height, width) # as colmap
            focal = 1.2*min(height, width) # as colmap
            K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
        else:
            K = np.array(data['K']).reshape(3, 3)
        camera = {'K':K ,'R': np.eye(3), 'T': np.zeros((3, 1)), 'dist': np.zeros((1, 5))}
        if self.gtK:
            camera['K'] = self.gtCameras[index]['K']
        if self.gtRT:
            camera['R'] = self.gtCameras[index]['R']
            camera['T'] = self.gtCameras[index]['T']
        # camera['T'][2, 0] = 5. # guess to 5 meters
        camera['RT'] = np.hstack((camera['R'], camera['T']))
        camera['P'] = camera['K'] @ np.hstack((camera['R'], camera['T']))
        return camera
    
    def basename(self, nf):
        return self.annotlist[nf].replace('.json', '')

    def write_keypoints3d(self, results, nf):
        outname = join(self.out, 'keypoints3d', '{}.json'.format(self.basename(nf)))
        self.writer.write_keypoints3d(results, outname)
    
    def write_vertices(self, results, nf):
        outname = join(self.out, 'vertices', '{}.json'.format(self.basename(nf)))
        self.writer.write_vertices(results, outname)
        
    def write_smpl(self, results, nf):
        outname = join(self.out, 'smpl', '{}.json'.format(self.basename(nf)))
        self.writer.write_smpl(results, outname)

    def vis_smpl(self, render_data, image, camera, nf):
        outname = join(self.out, 'smpl', '{}.jpg'.format(self.basename(nf)))
        images = [image]
        for key in camera.keys():
            camera[key] = camera[key][None, :, :]
        self.writer.vis_smpl(render_data, images, camera, outname, add_back=True)

# class VideoFolder(ImageFolder):
#     "一段视频的图片的文件夹"
#     def __init__(self, root, name, out=None, 
#         image_root='images', annot_root='annots', 
#         kpts_type='body15', config={}, no_img=False) -> None:
#         self.root = root
#         self.image_root = join(root, image_root, name)
#         self.annot_root = join(root, annot_root, name)
#         self.name = name
#         self.kpts_type = kpts_type
#         self.no_img = no_img
#         self.imagelist = sorted(os.listdir(self.image_root))
#         self.annotlist = sorted(os.listdir(self.annot_root))
#         self.ret_crop = False
#         self.gtK, self.gtRT = False, False

    def load_annot_all(self, path):
        # 这个不使用personID，只是单纯的罗列一下
        assert os.path.exists(path), '{} not exists!'.format(path)
        results = []
        annnames = sorted(glob(join(path, '*.json')))
        for annname in annnames:
            datas = read_annot(annname, self.kpts_type)
            if self.ret_crop:
                # TODO:修改imgname
                basename = os.path.basename(annname)
                imgname = annname\
                    .replace('annots-cpn', 'images')\
                    .replace('annots', 'images')\
                    .replace('.json', '.jpg')
                assert os.path.exists(imgname), imgname
                img = cv2.imread(imgname)
                crop_image(img, datas)
            results.append(datas)
        return results

    def load_annot(self, path, pids=[]):
        # 这个根据人的ID预先存一下
        assert os.path.exists(path), '{} not exists!'.format(path)
        results = {}
        annnames = sorted(glob(join(path, '*.json')))
        for annname in annnames:
            nf = int(os.path.basename(annname).replace('.json', ''))
            datas = read_annot(annname, self.kpts_type)
            for data in datas:
                pid = data['id']
                if len(pids) > 0 and pid not in pids:
                    continue
                # 注意 这里没有考虑从哪开始的
                if pid not in results.keys():
                    results[pid] = {'bboxes': [], 'keypoints2d': []}
                results[pid]['bboxes'].append(data['bbox'])
                results[pid]['keypoints2d'].append(data['keypoints'])
        for pid, val in results.items():
            for key in val.keys():
                val[key] = np.stack(val[key])
        return results
        
    def load_smpl(self, path, pids=[]):
        """ load SMPL parameters from files

        Args:
            path (str): root path of smpl
            pids (list, optional): used person ids. Defaults to [], loading all person.
        """
        assert os.path.exists(path), '{} not exists!'.format(path)
        results = {}
        smplnames = sorted(glob(join(path, '*.json')))
        for smplname in smplnames:
            nf = int(os.path.basename(smplname).replace('.json', ''))
            datas = read_smpl(smplname)
            for data in datas:
                pid = data['id']
                if len(pids) > 0 and pid not in pids:
                    continue
                # 注意 这里没有考虑从哪开始的
                if pid not in results.keys():
                    results[pid] = {'body_params': [], 'frames': []}
                results[pid]['body_params'].append(data)
                results[pid]['frames'].append(nf)
        for pid, val in results.items():
            val['body_params'] = merge_params(val['body_params'])
        return results

class _VideoBase:
    """Dataset for single sequence data
    """
    def __init__(self, image_root, annot_root, out=None, config={}, kpts_type='body15', no_img=False) -> None:
        self.image_root = image_root
        self.annot_root = annot_root
        self.kpts_type = kpts_type
        self.no_img = no_img
        self.config = config
        assert out is not None
        self.out = out
        self.writer = FileWriter(self.out, config=config)
        imgnames = sorted(os.listdir(self.image_root))
        self.imagelist = imgnames
        self.annotlist = sorted(os.listdir(self.annot_root))
        self.nFrames = len(self.imagelist)
        self.undis = False
        self.read_camera()
    
    def read_camera(self):
        # 读入相机参数
        annname = join(self.annot_root, self.annotlist[0])
        data = read_json(annname)
        if 'K' not in data.keys():
            height, width = data['height'], data['width']
            focal = 1.2*max(height, width)
            K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
        else:
            K = np.array(data['K']).reshape(3, 3)
        self.camera = {'K':K ,'R': np.eye(3), 'T': np.zeros((3, 1))}

    def __getitem__(self, index: int):
        imgname = join(self.image_root, self.imagelist[index])
        annname = join(self.annot_root, self.annotlist[index])
        assert os.path.exists(imgname) and os.path.exists(annname)
        assert os.path.basename(imgname).split('.')[0] == os.path.basename(annname).split('.')[0], '{}, {}'.format(imgname, annname)
        if not self.no_img:
            img = cv2.imread(imgname)
        else:
            img = None
        annot = read_annot(annname, self.kpts_type)
        return img, annot
    
    def __len__(self) -> int:
        return self.nFrames
    
    def write_smpl(self, peopleDict, nf):
        results = []
        for pid, people in peopleDict.items():
            result = {'id': pid}
            result.update(people.body_params)
            results.append(result)
        self.writer.write_smpl(results, nf)
    
    def vis_detections(self, image, detections, nf, to_img=True):
        return self.writer.vis_detections([image], [detections], nf, 
            key='keypoints', to_img=to_img, vis_id=True)
    
    def vis_repro(self, peopleDict, image, annots, nf):
        # 可视化重投影的关键点与输入的关键点
        detections = []
        for pid, data in peopleDict.items():
            keypoints3d = (data.keypoints3d @ self.camera['R'].T + self.camera['T'].T) @ self.camera['K'].T
            keypoints3d[:, :2] /= keypoints3d[:, 2:]
            keypoints3d = np.hstack([keypoints3d, data.keypoints3d[:, -1:]])
            det = {
                'id': pid,
                'repro': keypoints3d
            }
            detections.append(det)
        return self.writer.vis_detections([image], [detections], nf, key='repro',
            to_img=True, vis_id=False)

    def vis_smpl(self, peopleDict, faces, image, nf, sub_vis=[], 
        mode='smpl', extra_data=[], add_back=True,
        axis=np.array([1., 0., 0.]), degree=0., fix_center=None):
        # 为了统一接口，旋转视角的在此处实现，只在单视角的数据中使用
        # 通过修改相机参数实现
        # 相机参数的修正可以通过计算点的中心来获得
        # render the smpl to each view
        render_data = {}
        for pid, data in peopleDict.items():
            render_data[pid] = {
                'vertices': data.vertices, 'faces': faces, 
                'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        for iid, extra in enumerate(extra_data):
            render_data[10000+iid] = {
                'vertices': extra['vertices'],
                'faces': extra['faces'],
                'colors': extra['colors'],
                'name': extra['name']
            }
        camera = {}
        for key in self.camera.keys():
            camera[key] = self.camera[key][None, :, :]
        # render another view point
        if np.abs(degree) > 1e-3:
            vertices_all = np.vstack([data.vertices for data in peopleDict.values()])
            if fix_center is None:
                center = np.mean(vertices_all, axis=0, keepdims=True)
                new_center = center.copy()
                new_center[:, 0:2] = 0
            else:
                center = fix_center.copy()
                new_center = fix_center.copy()
                new_center[:, 2] *= 1.5
            direc = np.array(axis)
            rot, _ = cv2.Rodrigues(direc*degree/90*np.pi/2)
            # If we rorate the data, it is like:
            # V = Rnew @ (V0 - center) + new_center
            #   = Rnew @ V0 - Rnew @ center + new_center
            # combine with the camera
            # VV = Rc(Rnew @ V0 - Rnew @ center + new_center) + Tc
            #    = Rc@Rnew @ V0 + Rc @ (new_center - Rnew@center) + Tc
            blank = np.zeros_like(image, dtype=np.uint8) + 255
            images = [image, blank]
            Rnew = camera['R'][0] @ rot
            Tnew = camera['R'][0] @ (new_center.T - rot @ center.T) + camera['T'][0]
            camera['K'] = np.vstack([camera['K'], camera['K']])
            camera['R'] = np.vstack([camera['R'], Rnew[None, :, :]])
            camera['T'] = np.vstack([camera['T'], Tnew[None, :, :]])
        else:
            images = [image]
        self.writer.vis_smpl(render_data, nf, images, camera, mode, add_back=add_back)

def load_cameras(path):
    # 读入相机参数
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    if os.path.exists(intri_name) and os.path.exists(extri_name):
        cameras = read_camera(intri_name, extri_name)
        cams = cameras.pop('basenames')
    else:
        print('\n\n!!!there is no camera parameters, maybe bug: \n', intri_name, extri_name, '\n')
        cameras = None
    return cameras

def numpy_to_list(array, precision=3):
    return np.round(array, precision).tolist()

class MVBase_Hand:
    """ Dataset for multiview data
    """
    def __init__(self, root, cams=[], out=None, config={}, 
        image_root='images', annot_root='annot', annot_file='handpose.pkl', 
        kpts_type='body15',
        undis=True, no_img=False, filter2d=None, start_frame="00000.jpg", end_frame="last", cam_as_world=None) -> None:
        # cam_as_world: the camera to use as the world coordinate system
        self.root = root
        self.image_root = join(root, image_root)
        self.annt_root = join(root, annot_root)
        self.kpts_type = kpts_type
        self.undis = undis
        self.no_img = no_img
        # use when debug
        self.ret_crop = False
        self.config = config
        # results path
        # the results store keypoints3d
        self.skel_path = None
        self.out = out
        self.writer = FileWriter(self.out, config=config)
        self.cam_as_world = cam_as_world    
        self.cams = cams
        self.imagelist = {}
        self.annotlist = {}
        self.annot_imglist = {}
        self.start_frame = int(os.path.splitext(start_frame)[0]) if type(start_frame)==str else start_frame
        last_frame = sorted(glob(os.path.join(self.image_root, cams[0], "*.jpg")))[-1]
        last_frame = int(os.path.splitext(os.path.basename(last_frame))[0])
        end_frame = min(end_frame, last_frame) if end_frame>0 else last_frame
        self.end_frame = end_frame
        print("Start frame: {}, End frame: {}".format(self.start_frame, self.end_frame))

        annot_file = annot_file.split('.')[0]        
        for cam in cams: #TODO: 增加start,end
            # ATTN: when image name's frame number is not continuous,
            imgnames = sorted(os.listdir(join(self.image_root, cam)))
            imgnames = [imgname for imgname in imgnames if int(os.path.splitext(imgname)[0])>=self.start_frame and int(os.path.splitext(imgname)[0])<=self.end_frame]
            self.imagelist[cam] = imgnames
            
            if cam=="Cam1":
                annt_file_cam = annot_file.split('_')[0] + f'_{cam}_' + 'track.pkl' if len(annot_file.split('_'))==2 else annot_file.split('_')[0] + f'_{cam}.pkl'
            else:
                if len(annot_file.split('_'))==2:
                    annt_file_cam = annot_file.split('_')[0] + f'_{cam}_' + annot_file.split('_')[1] + '.pkl'
                else:
                    annt_file_cam = annot_file.split('_')[0] + f'_{cam}.pkl'
            with open(os.path.join(self.annt_root, annt_file_cam), 'rb') as file:  
                annot_cam = pickle.load(file)
            self.annot_imglist[cam] = [imgname for imgname in sorted(annot_cam.keys()) if int(os.path.splitext(imgname)[0])>=self.start_frame and int(os.path.splitext(imgname)[0])<=self.end_frame]
            self.annotlist[cam] = [annot_cam[imgname] for imgname in self.annot_imglist[cam]]
            self.has2d = True
        
        
        nFrames = len(self.imagelist[cams[0]])
        self.nFrames = nFrames
        self.nViews = len(cams)
        self.read_camera(join(self.root, 'Calibration'))
        self.filter2d = filter2d
        if filter2d is not None:
            from .filter import make_filter
            self.filter2d = make_filter(filter2d)

    def read_camera(self, path):
        # 读入相机参数
        intri_name = join(path, 'intri.yml')
        extri_name = join(path, 'extri.yml')
        if os.path.exists(intri_name) and os.path.exists(extri_name):
            self.cameras = read_camera(intri_name, extri_name, cam_names=self.cams)
            if self.cam_as_world is not None:
                self.cameras = transform_cameras_to_reference(self.cameras, self.cam_as_world)
            self.cameras.pop('basenames')
            # 注意：这里的相机参数一定要用定义的，不然只用一部分相机的时候会出错
            self.cameras_for_affinity = [[cam['invK'], cam['R'], cam['T']] for name,cam in self.cameras.items()]
            self.Pall = np.stack([self.cameras[cam]['P'] for cam in self.cams])
            self.Fall = get_fundamental_matrix(self.cameras, self.cams)
        else:
            print('\n!!!\n!!!there is no camera parameters, maybe bug: \n', intri_name, extri_name, '\n')
            self.cameras = None

    def undistort(self, images):
        if self.cameras is not None and len(images) > 0:
            images_ = []
            for nv in range(self.nViews):
                mtx = self.cameras[self.cams[nv]]['K']
                dist = self.cameras[self.cams[nv]]['dist']
                if images[nv] is not None:
                    frame = cv2.undistort(images[nv], mtx, dist, None)
                else:
                    frame = None
                images_.append(frame)
        else:
            images_ = images
        return images_
    
    def undis_det(self, lDetections):
        for nv in range(len(lDetections)):
            camera = self.cameras[self.cams[nv]]
            for det in lDetections[nv]:
                if 'bbox' in det.keys():
                    det['bbox'] = Undistort.bbox(det['bbox'], K=camera['K'], dist=camera['dist'])
                #keypoints = det['keypoints']
                keypoints = det['pose']
                #det['keypoints'] = Undistort.points(keypoints=keypoints, K=camera['K'], dist=camera['dist'])
                det['pose'] = Undistort.points(keypoints=keypoints, K=camera['K'], dist=camera['dist'])
        return lDetections

    def select_hand(self, annots_all, index, hand_lr):
        # get keypoints and box for the corresponding hand
        annots = {'bbox': [], 'keypoints': []}
        valid = [] # whether each camera has valid detection or not
        for nv, cam in enumerate(self.cams):
            #data = [d for d in annots_all[nv] if d['side'] == hand_lr]
            data = [d for d in annots_all[nv] if d['handness'] == hand_lr]
            if len(data) == 1 and not (np.sum(data[0]['keypoints'][:,0]<0)>=data[0]['keypoints'].shape[0]//3 or np.sum(data[0]['keypoints'][:,1]<0)>=data[0]['keypoints'].shape[0]//3):
                data = data[0]
                bbox = data['bbox'] if 'bbox' in data else np.array([0, 0, 100., 100., 0., 0.0, 0.0, 0.0, 0.0, -1.0])
                keypoints = data['keypoints']
                if keypoints.shape[-1]==2:
                    # add confidence as 1
                    keypoints = np.concatenate((keypoints, np.ones((keypoints.shape[0],1))), axis=1)
                valid.append(True)
            else:
                if self.verbose:print('not found hand {} in frame {}, view {}'.format(hand_lr, index, nv))
                keypoints = np.zeros((self.config['nJoints'], 3))
                bbox = np.array([0, 0, 100., 100., 0., 0.0, 0.0, 0.0, 0.0, -1.0])
                valid.append(False)
            annots['bbox'].append(bbox)
            annots['keypoints'].append(keypoints)
        for key in ['bbox', 'keypoints']:
            annots[key] = np.stack(annots[key])
        valid = np.array(valid).astype(bool)
        
        return annots, valid

    def __getitem__(self, index: int):
        images, annots = [], []
        for cam in self.cams:
            imgname = join(self.image_root, cam, self.imagelist[cam][index])
            img_idx = int(os.path.splitext(os.path.basename(imgname))[0])
            assert os.path.exists(imgname), imgname
            if self.has2d:
                #assert os.path.exists(annname), annname
                assert self.imagelist[cam][index].split('.')[0] == self.annot_imglist[cam][index].split('.')[0], (cam, self.imagelist[cam][index], self.annot_imglist[cam][index])
                annot = self.annotlist[cam][index]
            if not self.no_img:
                img = cv2.imread(imgname)
                images.append(img)
            else:
                img = None
                images.append(None)
            
            annots.append(annot)
        if self.undis:
            images = self.undistort(images)
            annots = self.undis_det(annots)
        return images, annots, img_idx
    
    def __len__(self) -> int:
        return self.nFrames
    
    def vis_detections(self, images, lDetections, nf, mode='detec', to_img=True, sub_vis=[], onehand=False, vis_id=True, kp_3d=None, valid_note=None, vis_3d_only=False, no_grid=False, transparent=False):
        ext = '.png' if transparent else '.jpg'
        outname = join(self.out, mode, '{:06d}{}'.format(nf, ext))
        images_plt = []
        if len(sub_vis) != 0:
            vis_idx = [self.cams.index(i) for i in sub_vis]
            images_plt = [images[i] for i in vis_idx]
            lDetections = [lDetections[i] for i in vis_idx]
            novis_idx = [i for i in range(len(self.cams)) if i not in vis_idx]
            valid_note = [valid_note[i] for i in vis_idx]
        if onehand:
            images_plt = self.writer.vis_keypoints2d_1hand(images_plt, lDetections, outname=outname, vis_id=vis_id, valid_note=valid_note)
            for idx in novis_idx:
                images_plt.insert(idx, images[idx])
            return images_plt
        else:
            # plot 3d 
            return self.writer.vis_keypoints2d_mv_hand(images_plt, lDetections, outname=outname, vis_id=vis_id, kp_3d=kp_3d, valid_note=valid_note, vis_3d_only=vis_3d_only, no_grid=no_grid, transparent=transparent)

    def basename(self, nf):
        return '{:06d}'.format(nf)

    def write_keypoints2d(self, lDetections, nf):
        for nv in range(len(lDetections)):
            cam = self.cams[nv]
            annname = join(self.annot_root, cam, self.annotlist[cam][nf])
            outname = join(self.out, 'keypoints2d', cam, self.annotlist[cam][nf])
            annot_origin = read_json(annname)
            annots = lDetections[nv]
            results = []
            for annot in annots:
                results.append({
                    'personID': annot['id'],
                    'bbox': numpy_to_list(annot['bbox'], 2),
                    'keypoints': numpy_to_list(annot['keypoints'], 2)
                })
            annot_origin['annots'] = results
            save_annot(outname, annot_origin)

    def write_keypoints3d(self, results, nf):
        outname = join(self.out, 'keypoints3d', self.basename(nf)+'.json')
        self.writer.write_keypoints3d(results, outname)

    def write_vertices(self, results, nf):
        outname = join(self.out, 'vertices', '{}.json'.format(self.basename(nf)))
        self.writer.write_vertices(results, outname)

    def write_smpl(self, results, nf, mode='smpl'):
        outname = join(self.out, mode, self.basename(nf)+'.json')
        self.writer.write_smpl(results, outname)

    def vis_smpl(self, peopleDict, faces, images, nf, sub_vis=[], 
        mode='smpl', extra_data=[], extra_mesh=[], 
        add_back=True, camera_scale=1, cameras=None):
        # render the smpl to each view
        render_data = {}
        for pid, data in peopleDict.items():
            render_data[pid] = {
                'vertices': data.vertices, 'faces': faces, 
                'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        for iid, extra in enumerate(extra_data):
            render_data[10000+iid] = {
                'vertices': extra['vertices'],
                'faces': extra['faces'],
                'name': extra['name']
            }
            if 'colors' in extra.keys():
                render_data[10000+iid]['colors'] = extra['colors']
            elif 'vid' in extra.keys():
                render_data[10000+iid]['vid'] = extra['vid']
            
        if len(sub_vis) == 0:
            sub_vis = self.cams

        images = [images[self.cams.index(cam)] for cam in sub_vis]
        if cameras is None:
            cameras = {'K': [], 'R':[], 'T':[]}
            for key in cameras.keys():
                cameras[key] = [self.cameras[cam][key] for cam in sub_vis]
        for key in cameras.keys():
            cameras[key] = np.stack([self.cameras[cam][key] for cam in sub_vis])
        # 根据camera_back参数，控制相机向后退的距离
        # 相机的光心的位置: -R.T @ T
        if False:
            R = cameras['R']
            T = cameras['T']
            cam_center = np.einsum('bij,bjk->bik', -R.transpose(0, 2, 1), T)
            # 相机的朝向: R @ [0, 0, 1]
            zdir = np.array([0., 0., 1.]).reshape(-1, 3, 1)
            direction = np.einsum('bij,bjk->bik', R, zdir)
            cam_center = cam_center - direction * 1
            # 更新过后的相机的T: - R @ C
            Tnew = - np.einsum('bij,bjk->bik', R, cam_center)
            cameras['T'] = Tnew
        else:
            cameras['K'][:, 0, 0] /= camera_scale
            cameras['K'][:, 1, 1] /= camera_scale
        return self.writer.vis_smpl(render_data, nf, images, cameras, mode, add_back=add_back, extra_mesh=extra_mesh)

    def read_skeleton(self, start, end):
        keypoints3ds = []
        for nf in range(start, end):
            skelname = join(self.out, 'keypoints3d', '{:06d}.json'.format(nf))
            skeletons = read_keypoints3d(skelname)
            skeleton = [i for i in skeletons if i['id'] == self.pid]
            assert len(skeleton) == 1, 'There must be only 1 keypoints3d, id = {} in {}'.format(self.pid, skelname)
            keypoints3ds.append(skeleton[0]['keypoints3d'])
        keypoints3ds = np.stack(keypoints3ds)
        return keypoints3ds

    def read_skel(self, nf, path=None, mode='none'):
        if path is None:
            path = self.skel_path
            assert path is not None, 'please set the skeleton path'
        if mode == 'a4d':
            outname = join(path, '{}.txt'.format(nf))
            assert os.path.exists(outname), outname
            skels = readReasultsTxt(outname)
        elif mode == 'none':
            outname = join(path, '{:06d}.json'.format(nf))
            assert os.path.exists(outname), outname
            skels = readResultsJson(outname)
        else:
            import ipdb; ipdb.set_trace()
        return skels
    
    def read_smpl(self, nf, path=None):
        if path is None:
            path = self.skel_path
            assert path is not None, 'please set the skeleton path'
        outname = join(path, '{:06d}.json'.format(nf))
        assert os.path.exists(outname), outname
        datas = read_json(outname)
        outputs = []
        for data in datas:
            for key in ['Rh', 'Th', 'poses', 'shapes']:
                data[key] = np.array(data[key])
            outputs.append(data)
        return outputs
