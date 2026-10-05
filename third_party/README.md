# Third-party code

## EasyMocap

`EasyMocap/` is a trimmed and modified copy of [EasyMocap](https://github.com/zju3dv/EasyMocap), which we use for the multi-view triangulation of the hand poses. It is distributed under its own license (`EasyMocap/LICENSE`: research and non-profit use only), which is different from the MIT license of the rest of this repository. Please cite EasyMocap if you use this part of the code.

The upstream documentation folder (`doc/`) is not included, so the image links in `EasyMocap/Readme.md` do not resolve; please refer to the upstream repository for the original documentation.

Files added by us:

- `apps/demo/hand_get3d.py`: triangulate the 2D poses of both hands from multiple views
- `easymocap/dataset/base_2hands.py`, `easymocap/dataset/mv_2hands.py`: dataset classes for multi-view two-hand data

Upstream files modified by us include `easymocap/mytools/writer.py` (3D hand visualization) and `easymocap/mytools/camera_utils.py` (use one camera as the world coordinate system).
