# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import logging
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.image_model import DMCI, DMCI_Large, DMCI_Medium
from tools.data import ImageFolder

LAMBDA_MIN = 16
LAMBDA_MAX = 4096
QP_MAX = 63
_LAMBDA_RATIO = (LAMBDA_MAX / LAMBDA_MIN) ** (1.0 / QP_MAX)



def setup_logger(log_dir, rank=0):
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(f"train_rank{rank}")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Only create file handler for rank 0
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        # For other ranks, add a null handler to avoid warnings
        logger.addHandler(logging.NullHandler())
    
    return logger


def setup_distributed(backend='nccl'):
    """Setup distributed training using environment variables set by torchrun."""
    dist.init_process_group(backend=backend)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    return dist.get_rank(), dist.get_world_size(), local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize_between_processes(self):
        """Synchronize meter across all processes."""
        if not dist.is_available() or not dist.is_initialized():
            return
        
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = int(t[1])
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    return optim.Adam(net.parameters(), lr=args.learning_rate)



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, return_type="all"):
        super().__init__()
        self.metric = nn.MSELoss()
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        assert "likelihoods" in output, "Model output must contain 'likelihoods' dict."
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = self.metric(output["x_hat"], target)
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]

        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]


def get_lambda_from_qp(qp: int) -> float:
    qp = max(0, min(QP_MAX, int(qp)))
    return float(LAMBDA_MIN * (_LAMBDA_RATIO ** qp))


def train_one_epoch(
    model,
    train_dataloader,
    optimizer,
    criterion,
    epoch,
    clip_max_norm,
    logger,
    rank=0,
    args=None,
):
    model.train()
    device = next(model.parameters()).device
    start_time = time.time()
    
    # Use epoch as seed so all ranks generate the same QP sequence
    qp_rng = random.Random(epoch)
    
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        if args and args.input_noise and epoch < args.noise_epochs:
            progress = epoch / (args.noise_epochs - 1) if args.noise_epochs > 1 else 0
            current_sigma = args.noise_sigma_start * math.exp(-4.6 * progress)
            current_sigma = torch.rand(1).item() * current_sigma
            noise = torch.randn_like(d) * current_sigma
            target = d
            d = torch.clamp(d + noise, 0, 1)
        else:
            current_sigma = 0.0
            target = d

        qp = args.fixed_qp if args and args.fixed_qp is not None else qp_rng.randint(0, QP_MAX)
        curr_lambda = get_lambda_from_qp(qp)
        criterion.lmbda = curr_lambda

        optimizer.zero_grad()

        out_net = model(d, qp=qp)
        out_dict = criterion(out_net, target)
        loss = out_dict["loss"]
        mse_loss = out_dict["mse_loss"]
        bpp_loss = out_dict["bpp_loss"]

        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 200 == 0 and rank == 0:
            elapsed = time.time() - start_time
            
            noise_info = ""
            if args and args.input_noise:
                noise_info = f"Noise: {current_sigma:.4f} | "

            qp_info = f"QP: {qp} | "

            logger.info(
                f"Train epoch {epoch}: ["
                f"{i}/{len(train_dataloader)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f"Loss: {loss.item():.6f} | "
                f"MSE loss: {mse_loss.item():.6f} | "
                f"Bpp: {bpp_loss.item():.6f} | "
                f"{qp_info}"
                f"Lambda: {curr_lambda:.5f} | "
                f"{noise_info}"
                f"Time: {elapsed:.1f}s"
            )


def test_epoch(epoch, test_dataloader, model, criterion, logger, rank=0, args=None):
    model.eval()
    device = next(model.parameters()).device

    if rank != 0:
        return 0.0

    test_qp_list = [args.fixed_qp] if args and args.fixed_qp is not None else [0, 21, 42, 63]

    all_qp_losses = []

    with torch.no_grad():
        for qp in test_qp_list:
            curr_lambda = get_lambda_from_qp(qp)
            criterion.lmbda = curr_lambda

            total_loss = 0.0
            total_bpp = 0.0
            total_mse = 0.0
            num_samples = 0

            start_time = time.time()

            for d in test_dataloader:
                try:
                    d = d.to(device)
                    target = d

                    out_net = model(d, qp=qp)
                    out_dict = criterion(out_net, target)
                    loss = out_dict["loss"]
                    mse_loss = out_dict["mse_loss"]
                    bpp_loss = out_dict["bpp_loss"]

                    batch_size = d.size(0)
                    total_loss += loss.item() * batch_size
                    total_bpp += bpp_loss.item() * batch_size
                    total_mse += mse_loss.item() * batch_size
                    num_samples += batch_size

                except Exception as e:
                    logger.warning(f"[QP={qp}] Skipping test batch due to error: {e}")
                    continue

            if num_samples > 0:
                avg_loss = total_loss / num_samples
                avg_bpp = total_bpp / num_samples
                avg_mse = total_mse / num_samples
                all_qp_losses.append(avg_loss)

                test_time = time.time() - start_time
                logger.info(
                    f"[Test] Epoch {epoch} | QP {qp} | Lambda {curr_lambda:.5f} | "
                    f"time {test_time:.1f}s | Samples {num_samples} | "
                    f"Loss {avg_loss:.6f} | MSE {avg_mse:.6f} | Bpp {avg_bpp:.6f}"
                )
            else:
                logger.warning(f"[QP={qp}] No valid test samples processed")

    if all_qp_losses:
        mean_loss = float(sum(all_qp_losses) / len(all_qp_losses))
        logger.info(
            f"[Test] Epoch {epoch} | Mean loss over {len(test_qp_list)} QPs: {mean_loss:.6f}"
        )
        return mean_loss
    else:
        logger.warning("No valid test samples processed at any QP")
        return float("inf")


def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="checkpoint_latest.pth.tar"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "checkpoint_best_loss.pth.tar")
        shutil.copyfile(filepath, best_filepath)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train DMCI with optional input noise")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--save_dir", default="exp/default")
    parser.add_argument("--model", default="dmci", choices=["dmci", "dmci_medium", "dmci_large"])
    parser.add_argument("--fixed_qp", type=int, nargs="?", const=63, default=None, help="Fixed QP in [0,63] (default when enabled: 63)")
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int, default=480)
    parser.add_argument("--lr_epoch", type=int, nargs="+")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--input_noise", action="store_true")
    parser.add_argument("--noise_sigma_start", type=float, default=0.5)
    parser.add_argument("--noise_epochs", type=int, default=None, help="Noise duration in epochs (default: epochs)")
    return parser.parse_args(argv)


def main(argv):
    """Main function for distributed training."""
    args = parse_args(argv)
    if args.fixed_qp is not None and not 0 <= args.fixed_qp <= QP_MAX:
        raise ValueError(f"fixed_qp must be in [0, {QP_MAX}]")
    if args.noise_epochs is None:
        args.noise_epochs = args.epochs

    # Check if running in distributed mode
    is_distributed = "WORLD_SIZE" in os.environ

    if is_distributed:
        rank, world_size, local_rank = setup_distributed("nccl")
    else:
        rank, world_size, local_rank = 0, 1, 0

    logger = setup_logger(args.save_dir, rank)

    if rank == 0:
        logger.info(f"Starting training with arguments: {args}")
        logger.info(f"Distributed: {is_distributed}, World size: {world_size}, Rank: {rank}")
        if args.fixed_qp is not None:
            logger.info(f"Fixed QP mode: QP={args.fixed_qp}, Lambda={get_lambda_from_qp(args.fixed_qp):.5f}")
        else:
            logger.info(f"Variable QP mode: QP range [0, {QP_MAX}]")
        if args.input_noise:
            logger.info(f"Input noise enabled: from {args.noise_sigma_start} to 0.0 over {args.noise_epochs} epochs")
        else:
            logger.info("Input noise disabled")
    
    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )


    train_dataset = ImageFolder(args.dataset, split="", transform=train_transforms)
    test_dataset = ImageFolder(args.val_dir, split="", transform=test_transforms)


    train_sampler = None
    test_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    device = f"cuda:{local_rank}"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=(device.startswith("cuda")),
        drop_last=True,  # Ensure all ranks have same number of batches for DDP sync
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # Set to 1 to handle images with different dimensions
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with test set
        shuffle=False,
        sampler=test_sampler,
        pin_memory=(device.startswith("cuda")),
    )

    fixed_qp_mode = args.fixed_qp is not None
    if args.model == "dmci":
        net = DMCI(N=256, z_channel=128, fixed_qp_mode=fixed_qp_mode).to(device)
    elif args.model == "dmci_large":
        net = DMCI_Large(N=512, z_channel=256, fixed_qp_mode=fixed_qp_mode).to(device)
    elif args.model == "dmci_medium":
        net = DMCI_Medium(N=368, z_channel=192, fixed_qp_mode=fixed_qp_mode).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if rank == 0:
        logger.info(f"Model: {args.model}, Parameters: {sum(p.numel() for p in net.parameters()):,}")

    if is_distributed:
        net = DDP(net, device_ids=[local_rank])
    elif torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer = configure_optimizers(net, args)
    criterion = RateDistortionLoss()
    
    last_epoch = 0
    if args.checkpoint:
        if rank == 0:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if is_distributed:
            net.module.load_state_dict(state_dict, strict=False)
        else:
            net.load_state_dict(state_dict, strict=False)
        if rank == 0:
            logger.info("Loaded model weights")

    # Create lr_scheduler AFTER loading checkpoint so that last_epoch is known.
    # Pass last_epoch-1 because MultiStepLR.__init__ calls step() once internally,
    # advancing the counter to last_epoch before the first training step.
    if args.lr_epoch:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_epoch, gamma=0.1,
            last_epoch=last_epoch - 1,
        )
    else:
        lr_scheduler = None

    best_loss = float("inf")
    
    if rank == 0:
        logger.info("Starting training loop...")
    
    for epoch in range(last_epoch, args.epochs):
        # Set epoch for distributed sampler
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"Epoch {epoch}/{args.epochs-1}")
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        train_one_epoch(
            net,
            train_dataloader,
            optimizer,
            criterion,
            epoch,
            args.clip_max_norm,
            logger,
            rank,
            args,
        )
        
        # Synchronize all ranks after training epoch
        if is_distributed:
            dist.barrier()

        # Only run test on rank 0.
        # Use net.module (unwrapped model) to avoid DDP triggering any collective
        # communication during eval-mode inference, which would deadlock because
        # other ranks are not participating in any forward pass here.
        if rank == 0:
            eval_model = net.module if is_distributed else net
            loss = test_epoch(epoch, test_dataloader, eval_model, criterion, logger, rank, args)
        else:
            loss = 0.0

        # Step learning rate scheduler
        if args.lr_epoch:
            lr_scheduler.step()  # MultiStepLR doesn't need loss

        # Only save on rank 0
        if rank == 0:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            # Get the actual model state dict
            if is_distributed:
                model_state_dict = net.module.state_dict()
            else:
                model_state_dict = net.state_dict()
            
            checkpoint_data = {
                "epoch": epoch,
                "state_dict": model_state_dict,
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            }
            
            # Save checkpoint every 5 epochs with unique filename
            if epoch % 70 == 0:
                filename = f"checkpoint_epoch_{epoch:03d}.pth.tar"
                save_checkpoint(checkpoint_data, False, args.save_dir, filename)
                logger.info(f"Checkpoint saved at epoch {epoch}: {filename}")
            
            # Save latest checkpoint (overwrite)
            save_checkpoint(checkpoint_data, is_best, args.save_dir)
            
            # Save final epoch checkpoint
            if epoch == args.epochs - 1:
                filename = f"checkpoint_final_epoch_{epoch:03d}.pth.tar"
                save_checkpoint(checkpoint_data, False, args.save_dir, filename)
                logger.info(f"Final checkpoint saved: {filename}")
            
            if is_best:
                logger.info(f"New best model saved with loss: {best_loss:.4f}")

        if is_distributed:
            dist.barrier()

    if rank == 0:
        logger.info("Training completed!")
    
    # Cleanup
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main(sys.argv[1:]) 