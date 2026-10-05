'''
  @ Date: 2021-04-13 19:46:51
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-13 17:56:25
  @ FilePath: /EasyMocap/apps/demo/mv1p.py
'''
import os,sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tqdm import tqdm
from easymocap.smplmodel import check_keypoints, load_model, select_nf
from easymocap.mytools import simple_recon_person, Timer, projectN3
from easymocap.pipeline import smpl_from_keypoints3d2d
from os.path import join
import numpy as np
import pickle
import traceback

def check_repro_error(keypoints3d, kpts_repro, keypoints2d, P, MAX_REPRO_ERROR):
    square_diff = (keypoints2d[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = keypoints3d[None, :, -1:]
    conf = (keypoints3d[None, :, -1:] > 0) * (keypoints2d[:, :, -1:] > 0)
    dist = np.sqrt((((kpts_repro[..., :2] - keypoints2d[..., :2])*conf)**2).sum(axis=-1))
    vv, jj = np.where(dist > MAX_REPRO_ERROR)
    if vv.shape[0] > 0:
        keypoints2d[vv, jj, -1] = 0.
        keypoints3d, kpts_repro = simple_recon_person(keypoints2d, P)
    return keypoints3d, kpts_repro

def mv1pmf_skel(dataset, check_repro=True, args=None):
    MIN_CONF_THRES = args.thres2d
    no_img = not (args.vis_det or args.vis_repro)
    dataset.no_img = no_img
    kp3ds_l, kp3ds_r = [],[]
    kp3d_valid = {0:[], 1:[]}
    
    start = args.start
    imagename_list = []
    for idx in tqdm(range(len(dataset)), desc='triangulation'):
        images, img_idx, annots_left, annots_right, valid_left, valid_right = dataset[idx] # images and annots shape [bs x num_views x ...]
        nf = img_idx
        check_keypoints(annots_left['keypoints'], WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)
        check_keypoints(annots_right['keypoints'], WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)
        kp_hands = []
        kpts_repro_hands = {0:None, 1:None}     # 0 for left and 1 for right 
        
        
        for i, annots in enumerate([annots_left, annots_right]):
            valid_note = valid_left if i==0 else valid_right
            kp3ds = kp3ds_r if i==1 else kp3ds_l
            kp3d_valid_this = kp3d_valid[i]
            
            if np.sum(valid_note)>1:
                    
                if np.sum(valid_note[1:])>2:
                    if i==0:
                        valid_note[3] = False  # don't consider camera 4 for left hand if cam2 and cam3 are valid, as cam 4 has bad quality for left hand pose estimation
                    if i==1:
                        valid_note[1] = False
                    
                if np.sum(valid_note[1:])>1:  # don't consider camera 1 if there are at least 2 other cameras valid
                    valid_note[0] = False        
                 
                kp_valid =  annots['keypoints'][valid_note, ...]
                P_valid = dataset.Pall[valid_note, ...]
                keypoints3d, kpts_repro = simple_recon_person(kp_valid, P_valid)
                if check_repro:
                    keypoints3d, kpts_repro = check_repro_error(keypoints3d, kpts_repro, kp_valid, P=P_valid, MAX_REPRO_ERROR=args.MAX_REPRO_ERROR)
                kpts_repro_hands[i] = kpts_repro
                num_joints = keypoints3d.shape[0]
                if np.sum(keypoints3d[:, -1]>0) < num_joints//2:
                    keypoints3d = np.zeros((21,4))
                    kpts_repro_hands[i] = None
                    kp3d_valid_this.append(False)
                else:
                    kp3d_valid_this.append(True)
            else:
                keypoints3d=np.zeros((21,4))
                kpts_repro_hands[i] = None
                kp3d_valid_this.append(False)
            
            # Added by Cursor (gpt5): Temporal speed sanity check: invalidate current frame if center jump is too large
            if len(kp3ds) > 0 and kp3d_valid_this[-2] and kp3d_valid_this[-1]:
                prev_center = np.mean(kp3ds[-1][:, :3], axis=0)
                curr_center = np.mean(keypoints3d[:, :3], axis=0)
                if np.linalg.norm(curr_center - prev_center) > args.MAX_SPEED_ERROR:
                    keypoints3d = np.zeros((21,4)) # use the previous frame's keypoints
                    kp3d_valid_this[-1] = False
                    kpts_repro_hands[i] = None
            
            #print(keypoints3d)  
        # keypoints3d, kpts_repro = robust_triangulate(annots['keypoints'], dataset.Pall, config=config, ret_repro=True)
            kp3ds.append(keypoints3d)
            kp_hands.append(keypoints3d)
        imagename_list.append('{:06d}.jpg'.format(nf))
        if args.vis_det:
            img_visl = dataset.vis_detections(images, annots_left, nf, sub_vis=args.sub_vis, onehand=True)
            dataset.vis_detections(img_visl, annots_right, nf, sub_vis=args.sub_vis, onehand=False)
        if args.vis_repro or args.vis_3d_only:
            try:
                if args.vis_3d_only:
                    dataset.vis_repro(images, None, nf=nf, sub_vis=args.sub_vis, onehand=False, kp_3d=kp_hands, valid_note=valid_right, vis_3d_only=True, no_grid=args.no_grid, transparent=args.transparent)
                else:
                    img_reproj_l = dataset.vis_repro(images, kpts_repro_hands[0], nf=nf, sub_vis=args.sub_vis, onehand=True, valid_note=valid_left) 
                    dataset.vis_repro(img_reproj_l, kpts_repro_hands[1], nf=nf, sub_vis=args.sub_vis, onehand=False, kp_3d=kp_hands, valid_note=valid_right)
            except Exception:
                print(traceback.format_exc())
        
    # smooth the skeleton
    if args.smooth3d > 0:
        kp3ds_l = smooth_skeleton(kp3ds_l, args.smooth3d)
        kp3ds_r = smooth_skeleton(kp3ds_r, args.smooth3d)
    
    kp_3d = {'left': kp3ds_l, 'right': kp3ds_r, 'image':imagename_list, 'left_valid':kp3d_valid[0], 'right_valid':kp3d_valid[1]}
    with open(join(args.path, args.annotfolder, 'keypoints_3d.pickle'), 'wb') as file:
        pickle.dump(kp_3d, file)

    
    
def mv1pmf_smpl(dataset, args, weight_pose=None, weight_shape=None):
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    keypoints2d, bboxes = [], []
    dataset.no_img = True
    for nf in tqdm(range(start, end), desc='loading'):
        images, annots = dataset[nf]
        keypoints2d.append(annots['keypoints'])
        bboxes.append(annots['bbox'])
    kp3ds = dataset.read_skeleton(start, end)
    keypoints2d = np.stack(keypoints2d)
    bboxes = np.stack(bboxes)
    kp3ds = check_keypoints(kp3ds, 1)
    # optimize the human shape
    with Timer('Loading {}, {}'.format(args.model, args.gender), not args.verbose):
        body_model = load_model(gender=args.gender, model_type=args.model)
    params = smpl_from_keypoints3d2d(body_model, kp3ds, keypoints2d, bboxes, 
        dataset.Pall, config=dataset.config, args=args,
        weight_shape=weight_shape, weight_pose=weight_pose)
    # write out the results
    dataset.no_img = not (args.vis_smpl or args.vis_repro)
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        param = select_nf(params, nf-start)
        dataset.write_smpl(param, nf)
        if args.write_smpl_full:
            param_full = param.copy()
            param_full['poses'] = body_model.full_poses(param['poses'])
            dataset.write_smpl(param_full, nf, mode='smpl_full')
        if args.write_vertices:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            write_data = [{'id': 0, 'vertices': vertices[0]}]
            dataset.write_vertices(write_data, nf)
        if args.vis_smpl:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            dataset.vis_smpl(vertices=vertices[0], faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis, add_back=True)
        if args.vis_repro:
            keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
            kpts_repro = projectN3(keypoints, dataset.Pall)
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis, mode='repro_smpl')

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1P2H
    import argparse
    parser = argparse.ArgumentParser('EasyMocap commond line tools')
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--imagefolder', type=str, default='1st')
    parser.add_argument('--annotfolder', type=str, default='handpose', help="annotation file storing all view estimated keypoints")
    parser.add_argument('--annot_file', default='handpose.pkl')
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the camera folder lists when in video mode')
    parser.add_argument('--start', type=int, default=0,
        help='frame start')
    parser.add_argument('--end', type=int, default=-1,
        help='frame end')    
    parser.add_argument('--step', type=int, default=1,
        help='frame step')
    parser.add_argument('--from_file', type=str, default=None)
    # 
    # keypoints and body model
    # 
    parser.add_argument('--cfg_model', type=str, default=None)
    parser.add_argument('--body', type=str, default='handl', choices=['body15', 'body25', 'h36m', 'bodyhand', 'bodyhandface', 'handl', 'handr', 'total'])
    parser.add_argument('--model', type=str, default='smpl', choices=['smpl', 'smplh', 'smplx', 'manol', 'manor'])
    parser.add_argument('--gender', type=str, default='neutral', 
        choices=['neutral', 'male', 'female'])
    # Input control
    detec = parser.add_argument_group('Detection control')
    detec.add_argument("--thres2d", type=float, default=0.3, 
        help="The threshold for suppress noisy kpts")
    # 
    # Optimization control
    # 
    recon = parser.add_argument_group('Reconstruction control')
    recon.add_argument('--smooth3d', type=int,
        help='the size of window to smooth keypoints3d', default=0)
    recon.add_argument('--MAX_REPRO_ERROR', type=int,
        help='The threshold of reprojection error', default=50)
    recon.add_argument('--MAX_SPEED_ERROR', type=int,
        help='The threshold of reprojection error', default=50)
    recon.add_argument('--robust3d', action='store_true')
    # 
    # visualization part
    # 
    output = parser.add_argument_group('Output control')
    output.add_argument('--vis_det', action='store_true')
    output.add_argument('--vis_repro', action='store_true')
    output.add_argument('--vis_3d_only', action='store_true')
    output.add_argument('--no_grid', action='store_true')
    output.add_argument('--transparent', action='store_true')
    output.add_argument('--vis_smpl', action='store_true')
    output.add_argument('--write_smpl_full', action='store_true')
    parser.add_argument('--write_vertices', action='store_true')
    output.add_argument('--vis_mask', action='store_true')
    output.add_argument('--undis', action='store_true')
    output.add_argument('--sub_vis', type=str, nargs='+', default=[],
        help='the sub folder lists for visualization')
    parser.add_argument('--cam_as_world', type=str, default=None,
        help='the camera to use as the world coordinate system')
    parser.add_argument('--opts',
                        help="Modify config options using the command-line",
                        default=[],
                        nargs='+')
    parser.add_argument('--skel', action='store_true')
    args = parse_parser(parser)
    help="""
  Demo code for multiple views and one person:

    - Input : {} => {}
    - Output: {}
    - Body  : {}=>{}, {}
""".format(args.path, ', '.join(args.sub), args.out, 
    args.model, args.gender, args.body)
    print(help)
    skel_path = join(args.out, 'keypoints3d')
    dataset = MV1P2H(args.path, image_root=args.imagefolder, annot_root=args.annotfolder, annot_file=args.annot_file, cams=args.sub, out=args.out,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=False,  start_frame=args.start, end_frame=args.end,verbose=False, cam_as_world=args.cam_as_world)
    dataset.writer.save_origin = False

    if not os.path.exists(skel_path):
        mv1pmf_skel(dataset, check_repro=False, args=args)
    
    #mv1pmf_smpl(dataset, args)
    