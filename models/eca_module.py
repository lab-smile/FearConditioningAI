r"""Efficient Channel Attention (ECA) module: https://arxiv.org/abs/1910.03151

Used as the bottom-up attention mechanism in the VCA models' middleroad/shortcut
pathway (models/VGG_Model.py's Visual_Cortex_Amygdala), reweighting each channel of a
feature map by a learned, cheap (1D-conv) function of its globally-pooled activation.
"""

from torch import nn
from .filter_module import AddGaussianNoise


class eca_layer(nn.Module):
    """Constructs an ECA module.
    Args:
        channel: Number of channels of the input feature map (unused; kept for API parity).
        k_size: Kernel size of the 1D conv used to model cross-channel interaction.
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information: (N, C, H, W) -> (N, C, 1, 1)
        y = self.avg_pool(x)

        # squeeze to (N, 1, C), 1D-convolve across channels to capture local cross-channel
        # interaction, then reshape back to (N, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # per-channel attention weights in (0, 1)
        y = self.sigmoid(y)

        # rescale the original feature map channel-wise by its attention weight
        return x * y.expand_as(x)
