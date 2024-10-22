import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.stats import geom


def torch_skew_symmetric(v):
    zero = torch.zeros_like(v[:, 0])
    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=1)
    return M


def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])
    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)
    return M


def get_episym(x1, x2, dR, dt):
    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([x1, np.ones((num_pts, 1))], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([x2, np.ones((num_pts, 1))], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(np.reshape(np_skew_symmetric(dt), (-1, 3, 3)), dR).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2) +
            1.0 / (Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2))

    return ys.flatten()


# def unpack_K(geom):
def unpack_K(data_img, data_K):
    # img_size, K = geom['img_size'], geom['K']
    img = data_img.cpu().numpy()
    K = data_K[0].cpu().numpy()
    img_size = img.shape
    h = img_size[2]
    w = img_size[3]
    cx = (w - 1.0) * 0.5
    cy = (h - 1.0) * 0.5
    cx += K[0, 2]
    cy += K[1, 2]
    # Get focals
    fx = K[0, 0]
    fy = K[1, 1]
    return cx, cy, [fx, fy]


def norm_kp(cx, cy, fx, fy, kp):
    # New kp
    kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
    return kp


def classification_gt(data, kp_i, kp_j):
    cx1, cy1, f1 = unpack_K(data['image0'], data['K0'])
    cx2, cy2, f2 = unpack_K(data['image1'], data['K1'])
    x1 = norm_kp(cx1, cy1, f1[0], f1[1], kp_i)
    x2 = norm_kp(cx2, cy2, f2[0], f2[1], kp_j)
    R_i = data['T_0to1'][0, :3, :3].cpu().numpy()  # R_i={ndarry:(3,3)}
    R_j = data['T_1to0'][0, :3, :3].cpu().numpy()

    # R_i, R_j = geom_i["R"], geom_j["R"]
    dR = np.dot(R_j, R_i.T)

    t_i = data['T_0to1'][0, :3, 3].cpu().numpy()
    t_i = t_i.reshape([3, 1])
    t_j = data['T_1to0'][0, :3, 3].cpu().numpy()
    t_j = t_j.reshape([3, 1])
    # t_i, t_j = geom_i["t"].reshape([3, 1]), geom_j["t"].reshape([3, 1])

    dt = t_j - np.dot(dR, t_i)
    if np.sqrt(np.sum(dt ** 2)) <= 1e-5:
        return []
    dtnorm = np.sqrt(np.sum(dt ** 2))
    dt /= dtnorm
    idx_sort = computeNN(data['desc0_orn'], data['desc1_orn'])
    x2 = x2[idx_sort[1], :]
    geod_d = get_episym(x1, x2, dR, dt)
    # ys = geod_d.reshape(-1, 1)
    ys = torch.tensor(geod_d.reshape(-1, 1))
    return ys


def computeNN(desc_ii, desc_jj):
    # desc_ii, desc_jj = torch.from_numpy(desc_ii).cuda(), torch.from_numpy(desc_jj).cuda()

    d1 = (desc_ii ** 2).sum(1)
    d2 = (desc_jj ** 2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2 * torch.matmul(desc_ii, desc_jj.transpose(0, 1))).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:, 0]
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.cpu().numpy()]
    return idx_sort


class RMmodel_Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['rmmodel']['loss']
        self.match_type = self.config['rmmodel']['match_coarse']['match_type']
        self.sparse_spvs = self.config['rmmodel']['match_coarse']['sparse_spvs']

        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']

            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                    if self.match_type == 'sinkhorn' \
                    else conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                # else:
                #     # This is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                #     # we could also add 'pseudo negtive-samples'
                #     pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]

                loss = c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                    if self.match_type == 'sinkhorn' \
                    else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))

    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def essential_matrix_loss(self, data, e_hat):
        T_0to1 = data['T_0to1']
        R_gt = T_0to1[:, :3, :3]
        t_gt = T_0to1[:, :3, 3]

        ess_hat = e_hat

        # Get groundtruth Essential matrix
        e_gt_unnorm = torch.reshape(torch.matmul(torch.reshape(torch_skew_symmetric(t_gt), (-1, 3, 3)), R_gt), (-1, 9))
        e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, keepdim=True)

        # Essential/Fundamental matrix loss
        # geod = batch_episym(pts0, pts1, e_hat)
        # essential_loss = torch.min(geod, self.geo_loss_margin * geod.new_ones(geod.shape))
        # essential_loss = essential_loss.mean()
        if ess_hat is not None:
            essential_loss = torch.mean(torch.min(
                torch.sum(torch.pow(ess_hat - e_gt, 2), dim=1),
                torch.sum(torch.pow(ess_hat + e_gt, 2), dim=1)))
        else:
            essential_loss = torch.tensor(0)
        return essential_loss

    def classification_loss(self, data, logits):
        pts0 = data['mkpts0_orn']
        pts1 = data['mkpts1_orn']
        y_hat = logits
        y_in = []

        if (pts0.tolist()).__len__() >= 2:
            y_in = classification_gt(data, pts0, pts1)
        else:
            y_hat = None

        if y_hat is not None:
            gt_geod_d = y_in[:, 0]
            is_pos = (gt_geod_d < self.loss_config['obj_geod_th']).type(y_hat.type())
            is_neg = (gt_geod_d >= self.loss_config['obj_geod_th']).type(y_hat.type())
            c = is_pos - is_neg
            classif_losses = -torch.log(torch.sigmoid(c * y_hat) + np.finfo(float).eps.item())
            # balance
            num_pos = torch.relu(torch.sum(is_pos, dim=0) - 1.0) + 1.0
            num_neg = torch.relu(torch.sum(is_neg, dim=0) - 1.0) + 1.0
            classif_loss_p = torch.sum(classif_losses * is_pos, dim=0)
            classif_loss_n = torch.sum(classif_losses * is_neg, dim=0)
            classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)
        else:
            classif_loss = torch.tensor(0)
        return classif_loss

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' else data['conf_matrix'],
            data['conf_matrix_gt'],
            weight=c_weight)
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        # 3. essential matrix regression loss
        if data['e_hat'] is not None:
            for i in range(len(data['e_hat'])):
                essential_loss = self.essential_matrix_loss(data, data['e_hat'][i])
                loss += self.loss_config['essential_weight'] * essential_loss
                loss_scalars.update({"loss_ess": essential_loss.clone().detach().cpu()})

        # 4. classification loss
        if data['y_hat'] is not None:
            for i in range(len(data['y_hat'])):
                classif_loss = self.classification_loss(data, data['y_hat'][i])
                loss += self.loss_config['classif_weight'] * classif_loss
                loss_scalars.update({"loss_classif": classif_loss.clone().detach().cpu()})

        # -----log total loss------
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
