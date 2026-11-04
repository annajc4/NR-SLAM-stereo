import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

current_folder = os.path.dirname(os.path.abspath(__file__))
stereo_mis_folder = os.path.join(current_folder, "../../robust-pose-estimator")
sys.path.insert(0, stereo_mis_folder)
from core.metrics.trajectory_metrics import absolute_trajectory_error, relative_pose_error
np.linalg.linalg = np.linalg 

#args should be trajectory.csv and the sequence's groundtruth.txt
if len(sys.argv) != 3:
    sys.exit("usage: python scripts/trajectory_error.py trajectory.csv groundtruth.txt")

def to_matrices(rows):
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 3:7]).as_matrix()
    poses[:, :3, 3] = rows[:, 0:3]
    return poses

estimate = np.loadtxt(sys.argv[1], delimiter=",")
groundtruth = np.loadtxt(sys.argv[2])

files = estimate[:, 0].astype(int)
estimate_poses = to_matrices(estimate[:, 1:])
estimate_poses[:, :3, 3] /= 1000.0                     # mm -> m, like groundtruth.txt
groundtruth_poses = to_matrices(groundtruth[files - 1, 1:])   # file N is groundtruth row N - 1

ate, errors = absolute_trajectory_error(groundtruth_poses, estimate_poses)
rpe_trans, rpe_rot = relative_pose_error(groundtruth_poses, estimate_poses)
print(f"RPE trans {rpe_trans.mean() * 1000:.2f} mm   rot {np.degrees(rpe_rot.mean()):.2f} deg")
print(f"{len(errors)} poses, files {files[0]}-{files[-1]}, ATE RMSE {ate * 1000:.2f} mm")
