import numpy as np
from numpy import load
import os

# change scene_info_0
directory = 'PATH_TO_LOFTR/data/megadepth/index/scene_info_0.1_0.7'

for filename in os.listdir(directory):
    f_npz = os.path.join(directory, filename)
    data = load(f_npz, allow_pickle=True)
    for count, image_path in enumerate(data['image_paths']):
        if image_path is not None:
            if 'Undistorted_SfM' in image_path:
                data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg')

    data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
    new_file = 'PATH_TO_LOFTR/data/megadepth/index/scene_info_0.1_0.7_no_sfm/' + filename
    np.savez(new_file, **data)
    print("Saved to ", new_file)

# change scene_info_val_1500
directory = 'PATH_TO_LOFTR/data/megadepth/index/scene_info_val_1500'

for filename in os.listdir(directory):
    f_npz = os.path.join(directory, filename)
    data = load(f_npz, allow_pickle=True)
    for count, image_path in enumerate(data['image_paths']):
        if image_path is not None:
            if 'Undistorted_SfM' in image_path:
                data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg')

    data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
    new_file = 'PATH_TO_LOFTR/data/megadepth/index/scene_info_val_1500_no_sfm/' + filename
    np.savez(new_file, **data)
    print("Saved to ", new_file)
