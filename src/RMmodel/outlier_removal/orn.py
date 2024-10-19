import torch
import torch.nn as nn
import numpy as np
# from loss import batch_episym
from collections import namedtuple


def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    F = F.reshape(-1, 1, 3, 3).repeat(1, num_pts, 1, 1)
    x2Fx1 = torch.matmul(x2.transpose(2, 3), torch.matmul(F, x1)).reshape(batch_size, num_pts)
    Fx1 = torch.matmul(F, x1).reshape(batch_size, num_pts, 3)
    Ftx2 = torch.matmul(F.transpose(2, 3), x2).reshape(batch_size, num_pts, 3)
    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[:, :, 0] ** 2 + Fx1[:, :, 1] ** 2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0] ** 2 + Ftx2[:, :, 1] ** 2 + 1e-15))
    return ys

class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=(1, 1))
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=(1, 1)),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=(1, 1))
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=(1, 1)),  # b*c*n*1
            trans(1, 2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=(1, 1))
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


# you can use this bottleneck block to prevent from overfiting when your dataset is small
class OAFilterBottleneck(nn.Module):
    def __init__(self, channels, points1, points2, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=(1, 1))
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=(1, 1)),  # b*c*n*1
            trans(1, 2))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points1),
            nn.ReLU(),
            nn.Conv2d(points1, points2, kernel_size=(1, 1)),
            nn.BatchNorm2d(points2),
            nn.ReLU(),
            nn.Conv2d(points2, points1, kernel_size=(1, 1))
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=(1, 1)))

    def forward(self, x):
        embed = self.conv(x)  # b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=(1, 1)))

    def forward(self, x_up, x_down):
        # x_up: b*c*n*1
        # x_down: b*c*k*1
        embed = self.conv(x_up)  # b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)  # b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class OANBlock(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        print('channels:' + str(channels) + ', layer_num:' + str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=(1, 1))

        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num // 2):
            self.l1_1.append(PointCN(channels))

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PointCN(2 * channels, channels))
        for _ in range(self.layer_num // 2 - 1):
            self.l1_2.append(PointCN(channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=(1, 1))

    def forward(self, data, xs):
        # data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2(torch.cat([x1_1, x_up], dim=1))

        logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        e_hat = weighted_8points(xs, logits)

        x1, x2 = xs[:, 0, :, :2], xs[:, 0, :, 2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual


class ORNet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        depth_each_stage = config.net_depth // (config.iter_num + 1)
        self.side_channel = (config.use_ratio == 2) + (config.use_mutual == 2)
        self.weights_init = OANBlock(config.net_channels, 4 + self.side_channel, depth_each_stage, config.clusters)
        self.weights_iter = [OANBlock(config.net_channels, 6 + self.side_channel, depth_each_stage, config.clusters) for
                             _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # data: b*1*n*c
        input = data['xs'].transpose(1, 3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1, 2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs'])
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()],
                          dim=1),
                data['xs'])
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class NNMatcher(object):
    """docstring for NNMatcher"""
    def __init__(self, ):
        super(NNMatcher, self).__init__()

    def run(self, nkpts, descs):
        # pts1, pts2: N*2 GPU torch tensor
        # desc1, desc2: N*C GPU torch tensor
        # corr: N*4
        # sides: N*2
        # corr_idx: N*2
        pts1, pts2, desc1, desc2 = nkpts[0], nkpts[1], descs[0], descs[1]
        d1, d2 = (desc1 ** 2).sum(1), (desc2 ** 2).sum(1)
        distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2 * torch.matmul(desc1, desc2.transpose(0, 1))).sqrt()
        dist_vals, nn_idx1 = torch.topk(distmat, k=2, dim=1, largest=False)
        nn_idx1 = nn_idx1[:, 0]
        _, nn_idx2 = torch.topk(distmat, k=1, dim=0, largest=False)
        nn_idx2 = nn_idx2.squeeze()
        mutual_nearest = (nn_idx2[nn_idx1] == torch.arange(nn_idx1.shape[0]).cuda())
        ratio_test = dist_vals[:, 0] / dist_vals[:, 1].clamp(min=1e-15)
        pts2_match = pts2[nn_idx1, :]
        corr = torch.cat([pts1, pts2_match], dim=-1)
        corr_idx = torch.cat([torch.arange(nn_idx1.shape[0]).unsqueeze(-1), nn_idx1.unsqueeze(-1).cpu()], dim=-1)
        sides = torch.cat([ratio_test.unsqueeze(1), mutual_nearest.float().unsqueeze(1)], dim=1)
        return corr, sides, corr_idx


def normalize_kpts(kpts):
    x_mean = np.mean(kpts, axis=0)
    dist = kpts - x_mean
    meandist = np.sqrt((dist ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2) / meandist
    T = np.zeros([3, 3])
    T[0, 0], T[1, 1], T[2, 2] = scale, scale, 1
    T[0, 2], T[1, 2] = -scale * x_mean[0], -scale * x_mean[1]
    nkpts = kpts * np.asarray([T[0, 0], T[1, 1]]) + np.array([T[0, 2], T[1, 2]])
    return nkpts


# class Outlier_Removal(object):
class Outlier_Removal(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.default_config = config

        self.default_config = namedtuple("Config", self.default_config.keys())(*self.default_config.values())

        self.model = ORNet(self.default_config)
        self.nn_matcher = NNMatcher()

    def forward(self, kpt_list, desc_list, batch):
        # kpts = {ndarray:(2000,2)}
        nkpts = [normalize_kpts(i) for i in kpt_list]                                       # nkpts:{list2}
        # descs = [torch.from_numpy(desc.astype(np.float32)).cuda() for desc in desc_list]  # descs:{list2}
        corr, sides, corr_idx = self.nn_matcher.run(nkpts, desc_list)
        corr, sides = corr.unsqueeze(0).unsqueeze(0), sides.unsqueeze(0)
        data = {'xs': corr}

        if self.default_config.use_ratio == 2 and self.default_config.use_mutual == 2:
            data['sides'] = sides
        elif self.default_config.use_ratio == 0 and self.default_config.use_mutual == 1:
            mutual = sides[0, :, 1] > 0
            data['xs'] = corr[:, :, mutual, :]
            data['sides'] = []
            corr_idx = corr_idx[mutual, :]
        elif self.default_config.use_ratio == 1 and self.default_config.use_mutual == 0:
            ratio = sides[0, :, 0] < 0.8
            data['xs'] = corr[:, :, ratio, :]
            data['sides'] = []
            corr_idx = corr_idx[ratio, :]
        elif self.default_config.use_ratio == 1 and self.default_config.use_mutual == 1:
            mask = (sides[0, :, 0] < 0.8) & (sides[0, :, 1] > 0)
            data['xs'] = corr[:, :, mask, :]
            data['sides'] = []
            corr_idx = corr_idx[mask, :]
        elif self.default_config.use_ratio == 0 and self.default_config.use_mutual == 0:
            data['sides'] = []
        else:
            raise NotImplementedError

        y_hat, e_hat = self.model(data)

        y = y_hat[-1][0, :].cpu().numpy()
        inlier_idx = np.where(y > self.default_config.inlier_threshold)
        matches = corr_idx[inlier_idx[0], :].numpy().astype('int32')
        corr0 = kpt_list[0][matches[:, 0]]
        corr1 = kpt_list[1][matches[:, 1]]

        batch.update({
            "mkpts0_orn": corr0,
            "mkpts1_orn": corr1,
            "y_hat": y_hat,
            "e_hat": e_hat
        })

        # return matches, corr0, corr1
