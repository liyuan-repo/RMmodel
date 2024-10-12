from src.config.default import _CN as cfg

cfg.RMmodel.COARSE.TRAIN_RES = [640, 640]
cfg.RMmodel.COARSE.TEST_RES = [1152, 1152]
cfg.RMmodel.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.RMmodel.MATCH_COARSE.SPARSE_SPVS = False

cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.RMmodel.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
