import numpy as np
from numpy import load
import os
from PIL import Image


def modify_npz_index(directory1, directory2):
    """1.第一步修改索引文件
    您需要更改索引中的 NPZ 文件。在训练之前，运行这个脚本
    :param directory1: 'PATH_TO_LOFTR/data/megadepth/index/scene_info_0.1_0.7'  example
    :param directory2: 'PATH_TO_LOFTR/data/megadepth/index/scene_info_val_1500'
    :return:
    """
    # change scene_info_0
    for filename in os.listdir(directory1):
        f_npz = os.path.join(directory1, filename)
        data = load(f_npz, allow_pickle=True)
        for count, image_path in enumerate(data['image_paths']):
            if image_path is not None:
                if 'Undistorted_SfM' in image_path:
                    data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5',
                                                                                                              'jpg')

        data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
        # no_sfm_data_dir = r'PATH_TO_LOFTR/data/megadepth/index/scene_info_0.1_0.7_no_sfm/'
        no_sfm_data_dir = r'megadepth/index/scene_info_0.1_0.7_no_sfm/'
        os.makedirs(no_sfm_data_dir, exist_ok=True)
        new_file = no_sfm_data_dir + filename
        np.savez(new_file, **data)
        print("Saved to ", new_file)

    # change scene_info_val_1500
    for filename in os.listdir(directory2):
        f_npz = os.path.join(directory2, filename)
        data = load(f_npz, allow_pickle=True)
        for count, image_path in enumerate(data['image_paths']):
            if image_path is not None:
                if 'Undistorted_SfM' in image_path:
                    data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5',
                                                                                                              'jpg')

        data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
        # no_sfm_data_dir = r'PATH_TO_LOFTR/data/megadepth/index/scene_info_val_1500_no_sfm/'
        no_sfm_data_dir = r'megadepth/index/scene_info_val_1500_no_sfm/'
        os.makedirs(no_sfm_data_dir, exist_ok=True)
        new_file = no_sfm_data_dir + filename
        np.savez(new_file, **data)
        print("Saved to ", new_file)


def modify_suffix_jpg(root_directory):
    """2.将数据集图片后缀中的非jpg后缀，改为jpg后缀
    然后，还要运行以下脚本，以确保所有图像都有结尾 'jpg' (数据集中有一些隐藏的 JPG 和 png)
    :param root_directory: '/PATH_TO_DATASET/phoenix/S6/zl548/MegaDepth_v1' example
    :return:
    """
    for folder in os.listdir(root_directory):
        four_digit_directory = os.path.join(root_directory, folder)
        for dense_folder in os.listdir(four_digit_directory):
            image_directory = os.path.join(four_digit_directory, dense_folder, 'imgs')
            for image in os.listdir(image_directory):
                if 'JPG' in image:
                    new_name = image.replace('JPG', 'jpg')
                    old_path = os.path.join(image_directory, image)
                    new_path = os.path.join(image_directory, new_name)
                    os.rename(old_path, new_path)
                if 'png' in image:
                    new_name = image.replace('png', 'jpg')
                    old_path = os.path.join(image_directory, image)
                    new_path = os.path.join(image_directory, new_name)
                    png_img = Image.open(old_path)
                    png_img.save(new_path)


"""
3.修改megadepth数据的配置文件
更改了 LoFTR/configs/data/megadepth_trainval_640.py 中的以下行：
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7_no_sfm"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val_1500_no_sfm"

4.修改megadepth.py中的代码
需要在 LoFTR/src/datasets/megadepth.py 中将 第47行代码改为以下代码，将类型转换为dict：
self.scene_info = dict(np.load(npz_path, allow_pickle=True))
"""


# https://blog.csdn.net/qq_42940160/article/details/141855785
if __name__ == '__main__':
    directory1 = r"megadepth/index/scene_info_0.1_0.7"
    directory2 = r"megadepth/index/scene_info_val_1500"
    root_directory = r"megadepth/train/phoenix/S6/zl548/MegaDepth_v1"
    # modify_npz_index(directory1, directory2)
    modify_suffix_jpg(root_directory)
