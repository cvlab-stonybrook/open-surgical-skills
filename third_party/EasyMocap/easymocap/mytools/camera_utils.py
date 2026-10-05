import cv2
import numpy as np
import os

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        else:
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        cv2.FileStorage.release(self.fs)

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            cv2.FileStorage.write(self.fs, key, value)
        elif dt == 'list':
            if self.major_version == 4: # 4.4
                self.fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for elem in value:
                    self.fs.write('', elem)
                self.fs.endWriteStruct()
            else: # 3.4
                self.fs.write(key, '[')
                for elem in value:
                    self.fs.write('none', elem)
                self.fs.write('none', ']')

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def read_intri(intri_name):
    assert os.path.exists(intri_name), intri_name
    intri = FileStorage(intri_name)
    camnames = intri.read('names', dt='list')
    cameras = {}
    for key in camnames:
        cam = {}
        cam['K'] = intri.read('K_{}'.format(key))
        cam['invK'] = np.linalg.inv(cam['K'])
        cam['dist'] = intri.read('dist_{}'.format(key))
        cameras[key] = cam
    return cameras

def read_extri(extri_name):
    assert os.path.exists(extri_name), extri_name
    extri = FileStorage(extri_name)
    camnames = extri.read('names', dt='list')
    cameras = {}
    for key in camnames:
        cam = {}
        cam['Rvec'] = extri.read('R_{}'.format(key))
        cam['R'] = extri.read('Rot_{}'.format(key))
        cam['T'] = extri.read('T_{}'.format(key))
        cameras[key] = cam
    return cameras

def write_intri(intri_name, cameras):
    intri = FileStorage(intri_name, True)
    results = {}
    camnames = list(cameras.keys())
    intri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        K, dist = val['K'], val['dist']
        assert K.shape == (3, 3), K.shape
        assert dist.shape == (1, 5) or dist.shape == (5, 1), dist.shape
        intri.write('K_{}'.format(key), K)
        intri.write('dist_{}'.format(key), dist.reshape(1, 5))

def write_extri(extri_name, cameras):
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = list(cameras.keys())
    extri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])
    return 0

def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    #cam_names = intri.read('names', dt='list')
    if len(cam_names)==0:
        cam_names = extri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format( cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['Rvec'] = Rvec
        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['T'] = Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams

def transform_cameras_to_reference(cameras, reference_cam):
    """
    Transform all camera parameters to use a specific camera as the world coordinate system.
    
    Args:
        cameras (dict): Dictionary containing camera parameters for all cameras
        reference_cam (str): Name of the camera to use as reference (e.g., 'Cam2')
    
    Returns:
        dict: Transformed camera parameters where the reference camera's coordinate system is the world coordinate system
    """
    if reference_cam not in cameras:
        raise ValueError(f"Reference camera {reference_cam} not found in cameras")
    
    # Get the reference camera's transformation
    ref_R = cameras[reference_cam]['R']
    ref_T = cameras[reference_cam]['T']
    
    # Create a copy of the cameras dictionary to avoid modifying the original
    transformed_cams = {}
    
    for cam_name, cam in cameras.items():
        if cam_name == 'basenames':
            transformed_cams[cam_name] = cam
            continue
            
        # Create a new camera dictionary
        new_cam = {}
        
        # Intrinsic parameters remain the same
        new_cam['K'] = cam['K'].copy()
        new_cam['invK'] = cam['invK'].copy()
        new_cam['dist'] = cam['dist'].copy()
        
        # Transform the rotation and translation
        # R_new = R_cam * R_ref^T
        # T_new = T_cam - R_cam * R_ref^T * T_ref
        new_cam['R'] = cam['R'] @ ref_R.T
        new_cam['T'] = cam['T'] - new_cam['R'] @ ref_T
        
        # Update other parameters
        new_cam['Rvec'] = cv2.Rodrigues(new_cam['R'])[0]
        new_cam['RT'] = np.hstack((new_cam['R'], new_cam['T']))
        new_cam['P'] = new_cam['K'] @ new_cam['RT']
        
        transformed_cams[cam_name] = new_cam
    
    return transformed_cams

def write_camera(camera, path):
    from os.path import join
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['dist'])
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])

def camera_from_img(img):
    height, width = img.shape[0], img.shape[1]
    # focal = 1.2*max(height, width) # as colmap
    focal = 1.2*min(height, width) # as colmap
    K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
    camera = {'K':K ,'R': np.eye(3), 'T': np.zeros((3, 1)), 'dist': np.zeros((1, 5))}
    return camera

class Undistort:
    @staticmethod
    def image(frame, K, dist):
        return cv2.undistort(frame, K, dist, None)

    @staticmethod
    def points(keypoints, K, dist):
        # keypoints: (N, 3)
        assert len(keypoints.shape) == 2, keypoints.shape
        kpts = keypoints[:, None, :2]
        kpts = np.ascontiguousarray(kpts)
        kpts = cv2.undistortPoints(kpts, K, dist, P=K)
        keypoints[:, :2] = kpts[:, 0]
        return keypoints
    
    @staticmethod
    def bbox(bbox, K, dist):
        keypoints = np.array([[bbox[0], bbox[1], 1], [bbox[2], bbox[3], 1]])
        kpts = Undistort.points(keypoints, K, dist)
        bbox = np.array([kpts[0, 0], kpts[0, 1], kpts[1, 0], kpts[1, 1], bbox[4]])
        return bbox

def undistort(camera, frame=None, keypoints=None, output=None, bbox=None):
    # bbox: 1, 7
    print('This function is deprecated')
    raise NotImplementedError

def get_fundamental_matrix(cameras, basenames):
    skew_op = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: np.linalg.inv(K_0).T @ (
            R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
    fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: fundamental_op (K_0, RT_0[:, :3], RT_0[:, 3], K_1,
                                                                          RT_1[:, :3], RT_1[:, 3] )
    F = np.zeros((len(basenames), len(basenames), 3, 3))  # N x N x 3 x 3 matrix
    F = {(icam, jcam): np.zeros((3, 3)) for jcam in basenames for icam in basenames}
    for icam in basenames:
        for jcam in basenames:
            F[(icam, jcam)] += fundamental_RT_op(cameras[icam]['K'], cameras[icam]['RT'], cameras[jcam]['K'], cameras[jcam]['RT'])
            if F[(icam, jcam)].sum() == 0:
                F[(icam, jcam)] += 1e-12  # to avoid nan
    return F
