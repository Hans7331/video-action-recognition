import numpy as np
import torch as T


class ContrastiveLoss(T.nn.Module):
  def __init__(self, m=2.0):
    super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
    self.m = m  # margin or radius

  def forward(self, y1, y2, d=0):
    # d = 0 means y1 and y2 are supposed to be same
    # d = 1 means y1 and y2 are supposed to be different
    
    euc_dist = T.nn.functional.pairwise_distance(y1, y2)

    if d == 0:
      return T.mean(T.pow(euc_dist, 2))  # distance squared
    else:  # d == 1
      delta = self.m - euc_dist  # sort of reverse distance
      delta = T.clamp(delta, min=0.0, max=None)
      return T.mean(T.pow(delta, 2))  # mean over all rows