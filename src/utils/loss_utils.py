import torch
import torch.nn.functional as F
from typing import List

from src.utils import basic_utils


def sequence_loss(flow_preds, flow_gt, valids, vis=None, gamma=0.8, use_huber_loss=False, loss_only_for_visible=False):
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        B, S2, N = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            if use_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt[j], delta=6.0)
            else:
                i_loss = (flow_pred - flow_gt[j]).abs()
            i_loss = torch.mean(i_loss, dim=3)
            valid_ = valids[j].clone()
            if loss_only_for_visible:
                valid_ = valid_ * vis[j]
            flow_loss += i_weight * basic_utils.reduce_masked_mean(i_loss, valid_)
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)


def sequence_loss_dense(flow_preds, flow_gt, valids, vis=None, gamma=0.8, use_huber_loss=False, loss_only_for_visible=False):
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, D, H, W = flow_gt[j].shape
        B, S2, _, H, W = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            if use_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt[j], delta=6.0)
            else:
                i_loss = (flow_pred - flow_gt[j]).abs()
            i_loss_ = torch.mean(i_loss, dim=2)
            valid_ = valids[j].reshape(B, S, H, W)
            if loss_only_for_visible:
                valid_ = valid_ * vis[j].reshape(B, -1, H, W)
            flow_loss += i_weight * basic_utils.reduce_masked_mean(i_loss_, valid_, broadcast=True)
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)


def huber_loss(x, y, delta=1.0):
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff ** 2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def sequence_BCE_loss(vis_preds, vis_gts, valids=None, use_logits=False):
    total_bce_loss = 0.0
    for j in range(len(vis_preds)):
        n_predictions = len(vis_preds[j])
        bce_loss = 0.0
        for i in range(n_predictions):
            if use_logits:
                loss = F.binary_cross_entropy_with_logits(vis_preds[j][i], vis_gts[j], reduction='none')
            else:
                loss = F.binary_cross_entropy(vis_preds[j][i], vis_gts[j], reduction='none')
            if valids is None:
                bce_loss += loss.mean()
            else:
                bce_loss += (loss * valids[j]).mean()
        bce_loss = bce_loss / n_predictions
        total_bce_loss += bce_loss
    return total_bce_loss / len(vis_preds)


def sequence_prob_loss(tracks, confidence, target_points, visibility, expected_dist_thresh=12.0, use_logits=False):
    total_logprob_loss = 0.0
    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0
        for i in range(n_predictions):
            err = torch.sum((tracks[j][i].detach() - target_points[j]) ** 2, dim=-1)
            valid = (err <= expected_dist_thresh ** 2).float()
            if use_logits:
                loss = F.binary_cross_entropy_with_logits(confidence[j][i], valid, reduction="none")
            else:
                loss = F.binary_cross_entropy(confidence[j][i], valid, reduction="none")
            loss *= visibility[j]
            loss = torch.mean(loss, dim=[1, 2])
            logprob_loss += loss
        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss
    return total_logprob_loss / len(tracks)


def sequence_prob_loss_dense(tracks, confidence, target_points, visibility, expected_dist_thresh=12.0, use_logits=False):
    total_logprob_loss = 0.0
    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0
        for i in range(n_predictions):
            err = torch.sum((tracks[j][i].detach() - target_points[j]) ** 2, dim=2)
            positive = (err <= expected_dist_thresh ** 2).float()
            if use_logits:
                loss = F.binary_cross_entropy_with_logits(confidence[j][i].squeeze(2), positive, reduction="none")
            else:
                loss = F.binary_cross_entropy(confidence[j][i].squeeze(2), positive, reduction="none")
            loss *= visibility[j].squeeze(2)
            loss = torch.mean(loss, dim=[1, 2, 3])
            logprob_loss += loss
        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss
    return total_logprob_loss / len(tracks)


def masked_mean(data, mask, dim):
    if mask is None:
        return data.mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(mask_sum, min=1.0)
    return mask_mean


def masked_mean_var(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(mask_sum, min=1.0)
    mask_var = torch.sum(mask * (data - mask_mean) ** 2, dim=dim, keepdim=True) / torch.clamp(mask_sum, min=1.0)
    return mask_mean.squeeze(dim), mask_var.squeeze(dim)
