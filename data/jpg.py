import os
from PIL import Image

root_directory = '/PATH_TO_DATASET/phoenix/S6/zl548/MegaDepth_v1'

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


# Then, I changed the following lines in LoFTR/configs/data/megadepth_trainval_640.py:
#
# cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7_no_sfm"
# cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val_1500_no_sfm"


# the code in line 47 in megadepth.py should be
# self.scene_info = dict(np.load(npz_path, allow_pickle=True))
