import cv2
import numpy as np


def resize(image, long_dim):
    h, w = image.shape[0], image.shape[1]
    if h != w:
        image = cv2.resize(image, dsize=(long_dim, long_dim), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    else:
        image = cv2.resize(image, (int(w * long_dim / max(h, w)), int(h * long_dim / max(h, w))))
    return image


def draw_points(img, points, color=(0, 255, 0), radius=3):
    dp = [(int(points[i, 0]), int(points[i, 1])) for i in range(points.shape[0])]
    for i in range(points.shape[0]):
        cv2.circle(img, dp[i], radius=radius, color=color)
    return img


def draw_match(img1, img2, corr1, corr2, inlier=[True], color=None, radius1=1, radius2=1, resize=None):
    if resize is not None:
        scale1, scale2 = [img1.shape[1] / resize[0], img1.shape[0] / resize[1]], [img2.shape[1] / resize[0],
                                                                                  img2.shape[0] / resize[1]]
        img1, img2 = cv2.resize(img1, resize, interpolation=cv2.INTER_AREA), cv2.resize(img2, resize,
                                                                                        interpolation=cv2.INTER_AREA)
        corr1, corr2 = corr1 / np.asarray(scale1)[np.newaxis], corr2 / np.asarray(scale2)[np.newaxis]
    corr1_key = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], radius1) for i in range(corr1.shape[0])]
    corr2_key = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], radius2) for i in range(corr2.shape[0])]

    # cv2.KeyPoint是opencv中关键点检测函数detectAndCompute()返回的关键点的类，
    # 包含关键点的位置，方向等属性：
    # float   angle;    // 特征点的方向，值为[0, 360]，负值表示不使用
    # int     class_id; // 用于聚类的id
    # int     octave;   // 特征点所在的图像金字塔的组
    # point2f pt;       // 位置坐标
    # float   size;     // 特征点邻域直径
    # float   response;

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]

    # cv2.DMatch是opencv中匹配函数(例如：knnMatch)返回的用于匹配关键点描述符的类，
    # 这个DMatch 对象具有下列属性：
    # DMatch.distance - 描述符之间的距离。越小越好。
    # DMatch.imgIdx   - 目标图像的索引。
    # DMatch.queryIdx - 查询图像中描述符的索引。
    # DMatch.trainIdx - 目标图像中描述符的索引。

    if color is None:  # # if inlier 为 True, color = (0, 255, 0),else (0, 0, 255)
        color = [(0, 255, 0) if cur_inlier else (0, 0, 255) for cur_inlier in inlier]  # inlier = [True]
    if len(color) == 1:
        display = cv2.drawMatches(img1, corr1_key, img2, corr2_key, draw_matches, outImg=None, matchColor=color[0],
                                  singlePointColor=color[0], flags=4)

    # def drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor=None, singlePointColor=None,
    #                     matchesMask=None, flags=None)
    # @  param outImg Output image. Its content depends on the flags value defining what is drawn in the
    # output image. See possible flags bit values below.
    # @  param matchColor Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1)
    # the color is generated randomly.
    # @  param singlePointColor Color of single keypoints (circles), which means that keypoints do not
    # have the matches. If singlePointColor==Scalar::all(-1) , the color is generated randomly.
    # @  param matchesMask Mask determining which matches are drawn. If the mask is empty, all matches are drawn
    # @  param flags Flags setting drawing features. Possible flags bit values are defined by DrawMatchesFlags.

    # DRAW_MATCHES_FLAGS_DEFAULT：只绘制特征点的坐标点，显示在图像上就是一个个小圆点，每个小圆点的圆心坐标都是特征点的坐标。
    # DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：函数不创建输出的图像，而是直接在输出图像变量空间绘制，要求本身输出图像变量就是一个初始化好了的，
    # size与type都是已经初始化好的变量。
    # DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS ：单点的特征点不被绘制。
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：绘制特征点的时候绘制的是一个个带有方向的圆，这种方法同时显示图像的坐标，size和方向，
    # 是最能显示特征的一种绘制方式

    else:
        height, width = max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]
        display = np.zeros([height, width, 3], np.uint8)
        display[:img1.shape[0], :img1.shape[1]] = img1
        display[:img2.shape[0], img1.shape[1]:] = img2
        for i in range(len(corr1)):
            left_x, left_y, right_x, right_y = int(corr1[i][0]), int(corr1[i][1]), int(
                corr2[i][0] + img1.shape[1]), int(corr2[i][1])
            cur_color = (int(color[i][0]), int(color[i][1]), int(color[i][2]))
            cv2.line(display, (left_x, left_y), (right_x, right_y), cur_color, 1, lineType=cv2.LINE_AA)
    return display
