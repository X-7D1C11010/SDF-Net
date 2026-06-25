# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def cross_modal_contrastive_loss(feat, target, target_cam, temperature=0.07):
    feat = F.normalize(feat, dim=1, p=2)
    target = target.view(-1)
    target_cam = target_cam.view(-1)
    batch_size = feat.size(0)
    device = feat.device

    logits = torch.matmul(feat, feat.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    same_id = target.unsqueeze(0).eq(target.unsqueeze(1))
    diff_modality = target_cam.unsqueeze(0).ne(target_cam.unsqueeze(1))
    positive_mask = same_id & diff_modality & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        return feat.new_tensor(0.0)

    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_log_prob = (log_prob * positive_mask.float()).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -pos_log_prob[valid_anchor].mean()


def cross_modal_prototype_loss(feat, target, target_cam):
    feat = F.normalize(feat, dim=1, p=2)
    losses = []

    for pid in torch.unique(target):
        pid_mask = target == pid
        opt_mask = pid_mask & (target_cam == 0)
        sar_mask = pid_mask & (target_cam == 1)
        if opt_mask.any() and sar_mask.any():
            opt_proto = F.normalize(feat[opt_mask].mean(dim=0, keepdim=True), dim=1, p=2)
            sar_proto = F.normalize(feat[sar_mask].mean(dim=0, keepdim=True), dim=1, p=2)
            losses.append(1.0 - (opt_proto * sar_proto).sum(dim=1))

    if not losses:
        return feat.new_tensor(0.0)
    return torch.cat(losses).mean()


def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(
        num_classes=num_classes, feat_dim=feat_dim, use_gpu=True
    )  # center loss
    if "triplet" in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print(
            "expected METRIC_LOSS_TYPE should be triplet"
            "but got {}".format(cfg.MODEL.METRIC_LOSS_TYPE)
        )

    if cfg.MODEL.IF_LABELSMOOTH == "on":
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        # print("label smooth on, num_classes: ", num_classes)

    def metric_aux_loss(feat, target, target_cam):
        if isinstance(feat, list):
            feat_main = feat[0]
        else:
            feat_main = feat
        aux_loss = feat_main.new_tensor(0.0)
        if target_cam is None:
            return aux_loss

        if cfg.MODEL.CM_CONTRAST_LOSS_WEIGHT > 0:
            aux_loss = aux_loss + cfg.MODEL.CM_CONTRAST_LOSS_WEIGHT * cross_modal_contrastive_loss(
                feat_main,
                target,
                target_cam,
                temperature=cfg.MODEL.CM_CONTRAST_TEMP,
            )
        if cfg.MODEL.CM_PROTO_LOSS_WEIGHT > 0:
            aux_loss = aux_loss + cfg.MODEL.CM_PROTO_LOSS_WEIGHT * cross_modal_prototype_loss(
                feat_main,
                target,
                target_cam,
            )
        return aux_loss

    if sampler == "softmax":

        def loss_func(score, feat, target, target_cam=None, f_struct=None):
            return F.cross_entropy(score, target) + metric_aux_loss(feat, target, target_cam)

    elif sampler == "softmax_triplet":

        def loss_func(score, feat, target, target_cam, f_struct=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == "triplet":
                if cfg.MODEL.IF_LABELSMOOTH == "on":
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    return (
                        cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS
                        + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                        + metric_aux_loss(feat, target, target_cam)
                    )
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(
                            score[0], target
                        )
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    return (
                        cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS
                        + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                        + metric_aux_loss(feat, target, target_cam)
                    )
            else:
                print(
                    "expected METRIC_LOSS_TYPE should be triplet"
                    "but got {}".format(cfg.MODEL.METRIC_LOSS_TYPE)
                )

    else:
        print(
            "expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center"
            "but got {}".format(cfg.DATALOADER.SAMPLER)
        )

    from .structure_loss import StructureConsistencyLoss

    structure_loss_func = StructureConsistencyLoss()

    return loss_func, center_criterion, structure_loss_func
