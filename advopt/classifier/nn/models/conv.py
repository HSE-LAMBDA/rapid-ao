import torch
from torch import nn

from advopt.classifier.nn.utils import dropout

__all__ = [
  'Conv'
]

class Conv(nn.Module):
  def __init__(self, ndim, n, activation=nn.ReLU(), dropout=False):
    super(Conv, self).__init__()

    self.dropout = dropout

    in_channels = ndim[0]

    ### 32 x 32
    self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * n, kernel_size=3)
    self.conv1_2 = nn.Conv2d(in_channels=2 * n, out_channels=2 * n, kernel_size=3)
    self.pool1 = nn.MaxPool2d(kernel_size=2)

    ### 14 x 14
    self.conv2_1 = nn.Conv2d(in_channels=2 * n, out_channels=3 * n, kernel_size=3)
    self.pool2 = nn.MaxPool2d(kernel_size=2)

    ### 6 x 6
    self.conv3_1 = nn.Conv2d(in_channels=3 * n, out_channels=4 * n, kernel_size=3)

    self.dense = nn.Linear(in_features=4 * n, out_features=1)

    self.activation = activation

  def forward(self, X, p=None):
    if self.dropout:
      drop = lambda X: dropout(X, p)
    else:
      drop = lambda X: X

    result = X
    result = drop(self.activation(self.conv1_1(result)))
    result = drop(self.activation(self.conv1_2(result)))
    result = self.pool1(result)

    result = drop(self.activation(self.conv2_1(result)))
    result = self.pool2(result)

    result = drop(self.activation(self.conv3_1(result)))
    result, _ = torch.max(result, dim=2)
    result, _ = torch.max(result, dim=2)

    result = self.dense(result)

    return torch.squeeze(result, dim=1)