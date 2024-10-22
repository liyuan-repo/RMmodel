from yacs.config import CfgNode as CN


# CfgNode表示配置树中的一个内部节点。它是一个简单的字典式容器，允许基于属性的键访问。
# isinstance函数，第一个参数是一个对象，第二个参数是一个类型名或多个类型名组成的元组，只要该对象是其中一个类型，便返回True,否则False
def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()
# ResNet+FPN 参照 https://zhuanlan.zhihu.com/p/533021907
_CN.BACKBONE_TYPE = 'ResNetFPN'  # FPN:FeaturePyramidNetwork
_CN.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.FINE_CONCAT_COARSE_FEAT = True

# 1. RMmodel-backbone (local feature CNN) config
_CN.RESNETFPN = CN()
_CN.RESNETFPN.INITIAL_DIM = 128
_CN.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3

# 2. RMmodel-coarse module config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.D_FFN = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.COARSE.TEMP_BUG_FIX = False

# 3. Coarse-Matching config
_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.THR = 0.2
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax, 'sinkhorn']
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.SKH_ITERS = 3
_CN.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.MATCH_COARSE.SKH_PREFILTER = True
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.4  # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock

# 4. RMmodel-fine module config
_CN.FINE = CN()
_CN.FINE.D_MODEL = 128
_CN.FINE.D_FFN = 128
_CN.FINE.NHEAD = 8
_CN.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.FINE.ATTENTION = 'linear'

default_cfg = lower_config(_CN)
