import torch
import torch.nn as nn
from loguru import logger


class RMmodel_Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        
    def forward(self, data):

        loss_scalars = {}
        # 0. compute element-wise loss weight

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss()

        # 2. fine-level loss
        loss_f = self.compute_fine_loss()

        # 3.Essential/Fundamental matrix loss
        loss_essential = self.compute_Essential_matrix_loss()

        # 4. Classification loss
        loss_classif = compute_Classification_loss(self)


        loss = loss_c * self.loss_config['coarse_weight']    -----coarse_loss
        loss += loss_f * self.loss_config['fine_weight']     -----fine_loss
        loss += loss_classif * self.loss_classif             -----classif_loss
        loss += loss_essential * self.loss_essential         -----essential_loss

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
