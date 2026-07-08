import torch
import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(eq=False)
class VideoData:
    video: torch.Tensor
    trajs: torch.Tensor
    visibs: torch.Tensor
    valids: Optional[torch.Tensor] = None
    seq_name: Optional[str] = None
    dname: Optional[str] = None
    aug_video: Optional[torch.Tensor] = None


def collate_fn(batch):
    video = torch.stack([b.video for b in batch], dim=0)
    trajs = torch.stack([b.trajs for b in batch], dim=0)
    visibs = torch.stack([b.visibs for b in batch], dim=0)
    seq_name = [b.seq_name for b in batch]
    dname = [b.dname for b in batch]
    return VideoData(video=video, trajs=trajs, visibs=visibs, seq_name=seq_name, dname=dname)


def collate_fn_train(batch):
    gotit = [gotit for _, gotit in batch]
    video = torch.stack([b.video for b, _ in batch], dim=0)
    trajs = torch.stack([b.trajs for b, _ in batch], dim=0)
    visibs = torch.stack([b.visibs for b, _ in batch], dim=0)
    valids = torch.stack([b.valids for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    dname = [b.dname for b, _ in batch]
    return (
        VideoData(video=video, trajs=trajs, visibs=visibs, valids=valids, seq_name=seq_name, dname=dname),
        gotit,
    )


def try_to_cuda(t: Any) -> Any:
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj
