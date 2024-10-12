import torch
import torch.nn as nn
from torchvision import transforms
from einops.einops import rearrange
from .backbone import build_backbone
from src.RMmodel.preprocessing.preprocessing import preprocess
from src.RMmodel.transformer_module.position_encoding import PositionEncodingSine
from .transformer_module import FeatureEnhancementTransformer
from src.RMmodel.coarse_module.coarse_matching import CoarseMatching
from src.RMmodel.fine_module.fine_matching import FineMatching
from src.RMmodel.fine_module.fine_preprocess import FinePreprocess
from src.RMmodel.outlier_removal.orn import Outlier_Removal


# Robust Matching Model
class RMmodel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)              # output resolution are 1/8 and 1/2.
        self.pre_processing = preprocess(config['preprocess'])
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'], pre_scaling=[config['coarse']['train_res'], config['coarse']['test_res']])

        self.RM_coarse = FeatureEnhancementTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.RM_fine = FeatureEnhancementTransformer(config["fine"])
        self.fine_matching = FineMatching()

        self.outlier_removal = Outlier_Removal(config["orn"])

    def forward(self, data, online_resize=False):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        if online_resize:   # online_resize=Ture
            assert data['image0'].shape[0] == 1 and data['image1'].shape[1] == 1  # image张量的前两个参数如果不为1，会异常报错
            self.resize_input(data, self.config['coarse']['train_res'])  # 用训练时的图像分辨率train_res=[832, 832]
        else:
            data['pos_scale0'], data['pos_scale1'] = None, None

        # 1.-----------------------------------Local Feature Extractor-------------------------------------------------
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f, feats_r = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            # 返回 1/8的粗略特征图feats_c, 1/2的精细特征图feats_f
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
            (feat_r0, feat_r1) = feats_r.split(data['bs'])

        else:  # handle different input shapes
            (feat_c0, feat_f0, feat_r0) = self.backbone(data['image0'])
            (feat_c1, feat_f1, feat_r1) = self.backbone(data['image1'])
            # feats_c={Tensor:(2, 256, 64, 64)}, feats_f={Tensor:(2, 128, 256, 256)}
            # feat_c0={Tensor:(1, 256, 64, 64)}, feat_c1={Tensor:(1, 256, 64, 64)}
            # feat_f0={Tensor:(1, 128, 256, 256)}, feat_f1={Tensor:(1, 128, 256, 256)}
            # feat_r0={Tensor:(1, 128, 512, 256)}, feat_r1={Tensor:(1, 128, 512, 512)}

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:],
            'hw0_r': feat_r0.shape[2:], 'hw1_r': feat_r1.shape[2:]
        })
        # 'hw0_c'={Tensor:([64, 64])}, 'hw1_c'={Tensor:([64, 64])}
        # 'hw0_f'=torch.Size([256, 256]),'hw1_f'=torch.Size([256, 256])

        # 2. -------------------------------feature preprocessing module (FPM) module-----------------------------------
        feat_fpm0 = self.pre_processing(feat_c0)
        feat_fpm1 = self.pre_processing(feat_c1)

        # 3. ----------------------------------Feature Enhancement Transformer------------------------------------------

        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]

        feat_c0 = self.pos_encoding(feat_fpm0, data['pos_scale0'])
        feat_c1 = self.pos_encoding(feat_fpm1, data['pos_scale1'])

        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        feat_c0, feat_c1 = self.RM_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 4. -----------------------------------Coarse Matches Estimation----------------------------------------------

        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 5.-------------------------------------Fine Matches Estimation-----------------------------------------------
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)

        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.RM_fine(feat_f0_unfold, feat_f1_unfold)

        kpts0, kpts1 = self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        # 6. ----------------------------------------outlier_removal---------------------------------------------------
        # kpts0 = data['mkpts0_f'].cpu().numpy()
        # kpts1 = data['mkpts1_f'].cpu().numpy()

        feat_r0 = rearrange(feat_r0.squeeze(dim=0), 'c h w -> h w c')
        feat_r1 = rearrange(feat_r1.squeeze(dim=0), 'c h w -> h w c')
        desc0 = []
        desc1 = []
        for i in range(len(kpts0)):
            desc0[i] = feat_r0[int(kpts0[i][0])][int(kpts0[i][1])]
            desc1[i] = feat_r1[int(kpts1[i][0])][int(kpts1[i][1])]

        self.outlier_removal([kpts0, kpts1], [desc0, desc1], data)

        # ------------------------------------------------END----------------------------------------------------------

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def resize_input(self, data, train_res, df=32):
        h0, w0, h1, w1 = data['image0'].shape[2], data['image0'].shape[3], \
                         data['image1'].shape[2], data['image1'].shape[3]
        data['image0'], data['image1'] = self.resize_df(data['image0'], df), self.resize_df(data['image1'], df)

        if len(train_res) == 1:
            train_res_h = train_res_w = train_res
        else:
            train_res_h, train_res_w = train_res[0], train_res[1]
        data['pos_scale0'], data['pos_scale1'] = [train_res_h / data['image0'].shape[2],
                                                  train_res_w / data['image0'].shape[3]], \
                                                 [train_res_h / data['image1'].shape[2],
                                                  train_res_w / data['image1'].shape[3]]
        # 位置尺度pos_scale0,pos_scale1=训练图像分辨率/输入的图像分辨率=832/512=1.625
        data['online_resize_scale0'], data['online_resize_scale1'] = \
            torch.tensor([w0 / data['image0'].shape[3], h0 / data['image0'].shape[2]])[None].cuda(), \
            torch.tensor([w1 / data['image1'].shape[3], h1 / data['image1'].shape[2]])[None].cuda()

    def resize_df(self, image, df=32):
        h, w = image.shape[2], image.shape[3]
        h_new, w_new = h // df * df, w // df * df
        if h != h_new or w != w_new:
            img_new = transforms.Resize([h_new, w_new]).forward(image)
        else:
            img_new = image
        return img_new
