# Noise-Regularized Training for Learned Image Compression

Official implementation of **Noise-Regularized Training for Learned Image Compression** (NeurIPS 2026 Spotlight).

## Introduction

Noise-regularized training (NRT) is a simple, architecture-agnostic training recipe for learned image compression. At each training step, Gaussian noise is added to the encoder input while the distortion target remains clean. The latent uniform proxy, hyperprior, and entropy coding pipeline are left unchanged.

NRT adds essentially no training or inference overhead and consistently improves rate-distortion performance across architectures and operating regimes. A second-order analysis shows that input noise induces structured regularization: an information-weighted encoder-Jacobian penalty on the rate term and a contractive penalty on the end-to-end reconstruction map.

## Code and Models

Code, trained models, and detailed instructions will be released soon.
