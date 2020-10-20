import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")
from core.models.deepnn import coordconv


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    used to help calculate convolution sizes
    Args:
        h_w: height and width
        kernel_size: kernel size (assumes x by x)
        stride: the stride between each convolution
        pad: side pdading
        dilation: the spread of a convolution

    Returns:

    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

class CraftCoordSimple(nn.Module):
    def __init__(self, craft_model):
        """
        preliminary needs to be tuned, as well as dilation and kernel size needs to be examined
         - a very simple comparison of adding coord conv ontop of the two initial channels of text craft, needs plenty of parameter and hypter parameter tuning

        Args:
            craft_model: a textcraft model
        """
        super().__init__()
        self._craft = craft_model
        self.coordconv = coordconv.CoordConv2d(2, 32, 1, with_r=True, use_cuda=True)
        self.conv1 = nn.Conv2d(32, 16, 1)
        self.conv2 = nn.Conv2d(16, 16, 1)
        self.conv3 = nn.Conv2d(16, 2, 1)

        self.threshold_it = torch.nn.Threshold(.5, 1.0, False) #set a cap at max threshold for values

    def forward(self, x):
        x = x.cuda()
        x, features = self._craft(x)
        x1 = x.permute(0, 3, 1, 2)
        x = self.threshold_it(x1)
        x = self.coordconv(x.cuda())

        x = F.relu(self.conv1(x + features))
        x = F.relu(self.conv2(x))
        x = self.conv3(x) + x1
        return x, x1
