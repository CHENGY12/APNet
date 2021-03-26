import torch.nn.functional as F
import torch
import pdb
from .triplet_loss import TripletLoss
from .cross_entropy_loss import CrossEntropyLoss
# from .center_loss import CenterLoss




def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    cross_entropy = CrossEntropyLoss(num_classes=cfg.SOLVER.CLASSNUM,epsilon=cfg.SOLVER.SMOOTH)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            loss_id = cross_entropy(score, target) + triplet(feat, target)[0]
            # cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) #+PairwiseConfusion(score)/100.0
            return loss_id
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func
