import argparse
import cv2
import numpy as np
from einops import rearrange
import demo_utils


def Additive_noise(image_path, SNR):
    # 读取图片
    image0 = cv2.imread(image_path, cv2.IMREAD_COLOR)  # IMREAD_GRAYSCALE
    image = demo_utils.resize(image0, 512)
    image = np.array(image / 255, dtype=float)
    H, W, Channel = image.shape  # 获取行列,通道信息

    noise = np.random.rand(H, W, Channel)
    noise = noise - np.mean(noise)  # 均值为0，方差接近1
    avg1 = np.mean(image)
    P_image = np.sum((image - avg1) ** 2) / (H * W * Channel)  # 图像平均功率
    P_noise = P_image * 10 ** (-SNR / 10)  # 噪声功率
    # P_noise = P_image / (10 ** (-SNR/10))
    noise = np.sqrt(P_noise) / np.std(noise) * noise  # 期望的噪声
    # noise = np.sqrt(P_noise)
    noise_img = image + noise
    # return noise_image = image + noise

    # 设置图片添加高斯噪声之后的像素值的范围
    noise_img = np.clip(noise_img, a_min=0, a_max=1.0)
    noise_img = np.uint8(noise_img * 255)
    # 保存图片
    # cv2.imwrite("noisy_img.png", noise_img)
    return noise_img


def stripe_noise(image_path, sigma):
    image0 = cv2.imread(image_path, cv2.IMREAD_COLOR)  # IMREAD_GRAYSCALE
    image = demo_utils.resize(image0, 512)
    image = np.array(image / 255, dtype=float)
    H, W, Channel = image.shape
    # noise = np.random.randn(H, W)
    # meann = np.mean(image)
    # noise = noise - np.mean(noise)   # 均值为0，方差接近1
    # stripe = sigma**2 * noise
    stripe = sigma ** 2 * np.random.randn(H, W)
    # multi = image @ stripe
    # noise_img = image + multi
    noise_img = image + stripe @ image  # ---竖条纹
    # multi = stripe @ rearrange(image, 'h w c -> c h w ')  # ---横条纹
    # noise_img = image + rearrange(multi, 'c h w -> h w c')
    # noise_img = image + stripe * image   # * 是元素相乘,得到的结果为斑点噪声
    # 设置图片添加高斯噪声之后的像素值的范围
    noise_img = np.clip(noise_img, a_min=0, a_max=1.0)
    noise_img = np.uint8(noise_img * 255)
    return noise_img


if __name__ == '__main__':
    # Load example images
    parser = argparse.ArgumentParser()
    parser.add_argument('--img0_path', type=str, default="./assets/4SARSets/pair2-1.tif")
    parser.add_argument('--img1_path', type=str, default="./assets/4SARSets/pair2-1.tif")
    args = parser.parse_args()

    orig_image = cv2.imread(args.img0_path, cv2.IMREAD_COLOR)
    add_noise = Additive_noise(args.img0_path, -1)
    stripe_noise = stripe_noise(args.img0_path, 0.1)
    cv2.imshow("original image", orig_image)
    cv2.imshow("additive noise image", add_noise)
    cv2.imshow("stripe noise image", stripe_noise)

    cv2.waitKey()
