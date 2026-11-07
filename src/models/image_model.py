# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F
import math


from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from ..layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, round_and_to_int8

from .utils import LowerBound

g_ch_src = 3 * 8 * 8
g_ch_enc_dec = 368


class IntraEncoder(nn.Module):
    def __init__(self, N, ch=None):
        super().__init__()
        ch = ch if ch is not None else g_ch_enc_dec

        self.enc_1 = DepthConvBlock(g_ch_src, ch)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            nn.Conv2d(ch, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = F.pixel_unshuffle(x, 8)
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(out, quant_step)

        return self.forward_cuda(out, quant_step)

    def forward_torch(self, out, quant_step):
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)

    def forward_cuda(self, out, quant_step):
        out = self.enc_1(out, quant_step=quant_step)
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N, ch=None):
        super().__init__()
        ch = ch if ch is not None else g_ch_enc_dec

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(N, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
            DepthConvBlock(ch, ch),
        )
        self.dec_2 = DepthConvBlock(ch, g_ch_src)

    def forward(self, x, quant_step):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step)

        return self.forward_cuda(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out

    def forward_cuda(self, x, quant_step):
        out = self.dec_1[0](x)
        out = self.dec_1[1](out)
        out = self.dec_1[2](out)
        out = self.dec_1[3](out)
        out = self.dec_1[4](out)
        out = self.dec_1[5](out)
        out = self.dec_1[6](out)
        out = self.dec_1[7](out)
        out = self.dec_1[8](out)
        out = self.dec_1[9](out)
        out = self.dec_1[10](out)
        out = self.dec_1[11](out)
        out = self.dec_1[12](out, quant_step=quant_step)
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out

class DMCI(CompressionModel):
    def __init__(self, N=256, z_channel=128, fixed_qp_mode=False, ch=None):
        super().__init__(z_channel=z_channel)
        ch = ch if ch is not None else g_ch_enc_dec

        self.enc = IntraEncoder(N, ch)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2 + 2, 1),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2, 1),
        )

        self.dec = IntraDecoder(N, ch)

        self.fixed_qp_mode = fixed_qp_mode
        if fixed_qp_mode:
            self.q_scale_enc = nn.Parameter(torch.ones((1, ch, 1, 1)))
            self.q_scale_dec = nn.Parameter(torch.ones((1, ch, 1, 1)))
        else:
            self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), ch, 1, 1)))
            self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), ch, 1, 1)))
        self._init_bound()

    def load_state_dict(self, state_dict, strict=False):
        sd = dict(state_dict)  # shallow copy

        def adapt_q_scale(name: str, target: torch.nn.Parameter):
            if name not in sd:
                raise ValueError(f"Key {name} not found in state_dict")
            src = sd[name]
            dst = target
            if src.shape == dst.shape:
                return
            if src.shape[1:] != dst.shape[1:]:
                raise ValueError(f"Key {name} shape mismatch: {tuple(src.shape)} -> {tuple(dst.shape)}")
            if src.shape[0] == 1 and dst.shape[0] == self.get_qp_num():
                sd[name] = src.expand(self.get_qp_num(), -1, -1, -1).clone()
                print(f"[load_state_dict] expand {name}: {tuple(src.shape)} -> {tuple(sd[name].shape)}")
            else:
                raise ValueError(f"Key {name} unsupported mapping: {tuple(src.shape)} -> {tuple(dst.shape)}")


        adapt_q_scale("q_scale_enc", self.q_scale_enc)
        adapt_q_scale("q_scale_dec", self.q_scale_dec)

        result = super().load_state_dict(sd, strict=False)

        if result.missing_keys:
            print("[load_state_dict] missing_keys:")
            for k in result.missing_keys:
                print(f"  - {k}")

        # if result.unexpected_keys:
        #     print("[load_state_dict] unexpected_keys (from checkpoint):")
        #     for k in result.unexpected_keys:
        #         print(f"  - {k}")

        if strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                f"missing_keys={result.missing_keys}, "
                f"unexpected_keys={result.unexpected_keys}"
            )

        return result

    def quant(self, x):
        """Quantization method: add noise during training, round during inference"""
        if self.training:
            # Add uniform noise to simulate quantization during training
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            noise = noise.clone().detach()
            return x + noise
        else:
            return torch.round(x)

    def _init_bound(self):
        self.lowerbound_scale = LowerBound(0.11)
        self.likelihood_lower_bound = LowerBound(1e-9)

    @staticmethod
    def probs_to_bits(probs):
        """Convert probabilities to bits"""
        factor = -1.0 / math.log(2.0)
        bits = torch.log(probs + 1e-5) * factor
        bits = torch.clamp(bits, 0, None)
        return bits

    def forward_four_part_prior(self, y, params):
        q_enc, q_dec, scales, means = self.separate_prior(params)
        params_reduced = self.y_spatial_prior_reduction(params)
        
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        y = y * q_enc

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        spatial_params = torch.cat((y_hat_so_far, params_reduced), dim=1)
        scales, means = self.y_spatial_prior(self.y_spatial_prior_adaptor_1(spatial_params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        spatial_params = torch.cat((y_hat_so_far, params_reduced), dim=1)
        scales, means = self.y_spatial_prior(self.y_spatial_prior_adaptor_2(spatial_params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        spatial_params = torch.cat((y_hat_so_far, params_reduced), dim=1)
        scales, means = self.y_spatial_prior(self.y_spatial_prior_adaptor_3(spatial_params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        y_hat = y_hat_so_far + y_hat_3
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        # print(y_q + means == y_hat)     # True
        y_hat = y_hat * q_dec

        return y_res, y_q, y_hat, scales_hat

    def forward(self, x, qp=None):
        if qp is None:
            qp = self.get_qp_num() // 2
        device = x.device
        
        if self.fixed_qp_mode:
            curr_q_enc = self.q_scale_enc
            curr_q_dec = self.q_scale_dec
        else:
            curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
            curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        y_pad = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        
        z_hat = self.quant(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(y, params)

        # x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)
        x_hat = self.dec(y_hat, curr_q_dec)
        
        if qp is not None:
            _, _, H, W = x.size()
            pixel_num = H * W
            
            # During training, use residual (y_res) with noise for likelihood calculation
            # During inference, use quantized (y_q) for likelihood calculation
            # This matches the behavior in video_model.py
            if self.training:
                y_for_bit = self.add_noise(y_res)
                # z_for_bit = self.add_noise(z)
                z_for_bit = z_hat
            else:
                y_for_bit = y_q
                z_for_bit = z_hat
            
            scales_hat = self.lowerbound_scale(scales_hat)
            y_likelihood = self.get_y_gaussian_likelihood(y_for_bit, scales_hat)
            y_likelihood = self.likelihood_lower_bound(y_likelihood)
            bits_y = -torch.log2(y_likelihood)  # [B, C, H_y, W_y]

            z_likelihood = self.get_z_likelihood(z_for_bit, self.bit_estimator_z, qp)
            z_likelihood = self.likelihood_lower_bound(z_likelihood)
            bits_z = -torch.log2(z_likelihood)  # [B, C_z, H_z, W_z]
            
            bpp_y = torch.mean(torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num)
            bpp_z = torch.mean(torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num)
            bpp = bpp_y + bpp_z

            likelihoods = {
                "y": y_likelihood,
                "z": z_likelihood,
            }

            result = {
                "x_hat": x_hat,
                "bpp": bpp,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "likelihoods": likelihoods,
            }
        else:
            result = {"x_hat": x_hat}

        return result

    def compress(self, x, qp):
        device = x.device
        
        if self.fixed_qp_mode:
            curr_q_enc = self.q_scale_enc
            curr_q_dec = self.q_scale_dec
        else:
            curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
            curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        y_pad = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat, z_hat_write = round_and_to_int8(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat = \
            self.compress_prior_4x(
                y, params, self.y_spatial_prior_reduction,
                self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        cuda_event = torch.cuda.Event()
        cuda_event.record()
        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            cuda_event.wait()
            self.entropy_coder.reset()
            self.bit_estimator_z.encode_z(z_hat_write, qp)
            self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
            self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
            self.gaussian_encoder.encode_y(y_q_w_2, s_w_2)
            self.gaussian_encoder.encode_y(y_q_w_3, s_w_3)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def decompress(self, bit_stream, sps, qp):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        if self.fixed_qp_mode:
            curr_q_dec = self.q_scale_dec
        else:
            curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = self.get_downsampled_shape(sps['height'], sps['width'], 16)
        self.bit_estimator_z.decode_z(z_size, qp)
        z_q = self.bit_estimator_z.get_z(z_size, device, dtype)
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = params[:, :, :y_height, :y_width].contiguous()
        y_hat = self.decompress_prior_4x(params, self.y_spatial_prior_reduction,
                                         self.y_spatial_prior_adaptor_1,
                                         self.y_spatial_prior_adaptor_2,
                                         self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)
        return {"x_hat": x_hat}


class DMCI_Medium(DMCI):
    """
    DMCI_Large: ~4x parameters compared to DMCI.
    Achieved by doubling channel widths:
      - ch (g_ch_enc_dec): 368 -> 736
      - N: 256 -> 512
      - z_channel: 128 -> 256
    """
    def __init__(self, N=368, z_channel=192, fixed_qp_mode=False, ch=512):
        super().__init__(N=N, z_channel=z_channel, fixed_qp_mode=fixed_qp_mode, ch=ch)


class DMCI_Large(DMCI):
    """
    DMCI_Large: ~4x parameters compared to DMCI.
    Achieved by doubling channel widths:
      - ch (g_ch_enc_dec): 368 -> 736
      - N: 256 -> 512
      - z_channel: 128 -> 256
    """
    def __init__(self, N=512, z_channel=256, fixed_qp_mode=False, ch=736):
        super().__init__(N=N, z_channel=z_channel, fixed_qp_mode=fixed_qp_mode, ch=ch)


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Test both models and compare parameter counts
    model = DMCI(N=256, z_channel=128)
    model_large = DMCI_Large(N=512, z_channel=256)
    model_medium = DMCI_Medium(N=368, z_channel=192)
    def count_parameters(m):
        return sum(p.numel() for p in m.parameters())
    
    params_dmci = count_parameters(model)
    params_large = count_parameters(model_large)
    params_medium = count_parameters(model_medium)
    
    print(f"DMCI_Medium parameters: {params_medium:,}")
    print(f"DMCI parameters: {params_dmci:,}")
    print(f"DMCI_Large parameters: {params_large:,}")
    print(f"Ratio (Large/DMCI): {params_large / params_dmci:.2f}x")
    
    model.train()
    
    img = Image.open('roberto-nickson-48063.png').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    x = transform(img).unsqueeze(0)  # [1, 3, 256, 256]
    qp = 32
    
    output = model(x, qp)
    x_hat = output['x_hat']
    
    mse = F.mse_loss(x, x_hat)
    psnr = -10 * torch.log10(mse)
    
    output_img = transforms.ToPILImage()(x_hat.squeeze(0))
    output_img.save('reconstructed_output.png')
    
    print(f"Forward pass successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_hat.shape}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"MSE: {mse:.6f}")
    print(f"Reconstructed image saved as 'reconstructed_output.png'")
