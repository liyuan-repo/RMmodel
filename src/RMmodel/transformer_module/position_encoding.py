import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """
    def __init__(self, d_model, pre_scaling, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            pre_scaling :
        """
        super().__init__()
        self.d_model = d_model
        self.max_shape = max_shape
        self.pre_scaling = pre_scaling

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        # unsqueeze(0)将二维元组max_shape拓展为3维

        if pre_scaling[0] is not None and pre_scaling[1] is not None:
            train_res, test_res = pre_scaling[0], pre_scaling[1]
            x_position, y_position = x_position * train_res[1] / test_res[1], y_position * train_res[0] / test_res[0]

        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))

        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x, scaling=None):
        """
        Args:
            x: [N, C, H, W]
            scaling:
        """

        if scaling is None:  # onliner scaling overwrites pre_scaling
            return x + self.pe[:, :, :x.size(2), :x.size(3)]
        else:
            pe = torch.zeros((self.d_model, *self.max_shape))
            y_position = torch.ones(self.max_shape).cumsum(0).float().unsqueeze(0) * scaling[0]
            x_position = torch.ones(self.max_shape).cumsum(1).float().unsqueeze(0) * scaling[1]

            div_term = torch.exp(
                torch.arange(0, self.d_model // 2, 2).float() * (-math.log(10000.0) / (self.d_model // 2)))
            div_term = div_term[:, None, None]  # [C//4, 1, 1]
            pe[0::4, :, :] = torch.sin(x_position * div_term)
            pe[1::4, :, :] = torch.cos(x_position * div_term)
            pe[2::4, :, :] = torch.sin(y_position * div_term)
            pe[3::4, :, :] = torch.cos(y_position * div_term)
            pe = pe.unsqueeze(0).to(x.device)
            return x + pe[:, :, :x.size(2), :x.size(3)]
