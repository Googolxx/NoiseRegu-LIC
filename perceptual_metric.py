import pyiqa
import torch
import torch.nn as nn


def _mean_metric(value):
    if isinstance(value, tuple):
        value = value[0]
    return value.mean() if value.dim() > 0 else value


class PerceptualQualityMetrics(nn.Module):
    """PSNR / MS-SSIM / LPIPS / DISTS via pyiqa."""

    def __init__(self, device):
        super().__init__()
        self.psnr = pyiqa.create_metric("psnr", device=device)
        self.ms_ssim = pyiqa.create_metric("ms_ssim", device=device, as_loss=True)
        self.lpips = pyiqa.create_metric("lpips", device=device, as_loss=True)
        self.dists = pyiqa.create_metric("dists", device=device, as_loss=True)
        for metric in (self.psnr, self.ms_ssim, self.lpips, self.dists):
            metric.eval()
            for parameter in metric.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _clamp_pair(rec, gt):
        return rec.clamp(0.0, 1.0), gt.clamp(0.0, 1.0)

    @torch.no_grad()
    def evaluate(self, rec, gt):
        rec, gt = self._clamp_pair(rec, gt)
        return {
            "psnr": float(_mean_metric(self.psnr(rec, gt)).item()),
            "ms_ssim": float(_mean_metric(self.ms_ssim(rec, gt)).item()),
            "lpips": float(_mean_metric(self.lpips(rec, gt)).item()),
            "dists": float(_mean_metric(self.dists(rec, gt)).item()),
        }
