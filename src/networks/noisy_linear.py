"""NoisyLinear layer for learned exploration (Fortunato et al., 2018).

This module provides a NoisyLinear layer that replaces standard Linear layers
to enable learned, state-dependent exploration. Instead of using ε-greedy
exploration with a fixed schedule, NoisyLinear adds parametric noise to
network weights that is learned during training.

The implementation uses Factorized Gaussian Noise for efficiency, reducing
the number of noise parameters from p×q to p+q.

Theory:
    Standard Linear: y = Wx + b
    Noisy Linear: y = (W + σ^w ⊙ ε^w)x + (b + σ^b ⊙ ε^b)

    Where:
    - W, b: Learnable mean parameters
    - σ^w, σ^b: Learnable noise scale parameters
    - ε^w, ε^b: Noise samples (resampled each forward pass)

    Factorized Noise:
    - ε^w_{ij} = f(ε_i) · f(ε_j)
    - f(x) = sign(x) · √|x|

    This reduces parameters from O(pq) to O(p+q).

Classes:
    NoisyLinear: Factorized Gaussian NoisyLinear layer.

Example:
    >>> layer = NoisyLinear(64, 32)
    >>> x = torch.randn(4, 64)
    >>> y = layer(x)  # Uses noisy weights during training
    >>> layer.reset_noise()  # Resample noise for next forward pass
    >>> layer.eval()
    >>> y_eval = layer(x)  # Uses mean weights only

References:
    Fortunato, M., et al. (2018). "Noisy Networks for Exploration." ICLR.
    https://arxiv.org/abs/1706.10295
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized NoisyLinear layer for learned exploration.

    Implements factorized Gaussian noise for efficient parameter-space
    exploration. Noise is resampled on each call to reset_noise(), which
    should be called at the start of each forward pass during training.

    During evaluation (model.eval()), the layer uses only the mean weights
    without noise, providing deterministic behavior.

    Attributes:
        in_features (int): Size of input dimension.
        out_features (int): Size of output dimension.
        sigma_init (float): Initial noise scale parameter.
        weight_mu (nn.Parameter): Mean weight parameters.
        weight_sigma (nn.Parameter): Weight noise scale parameters.
        bias_mu (nn.Parameter): Mean bias parameters.
        bias_sigma (nn.Parameter): Bias noise scale parameters.
        weight_epsilon (torch.Tensor): Weight noise buffer (not learned).
        bias_epsilon (torch.Tensor): Bias noise buffer (not learned).

    Example:
        >>> # Create noisy layer
        >>> layer = NoisyLinear(128, 64, sigma_init=0.5)
        >>>
        >>> # Forward pass with noise
        >>> layer.train()
        >>> layer.reset_noise()
        >>> output = layer(input_tensor)
        >>>
        >>> # Forward pass without noise (evaluation)
        >>> layer.eval()
        >>> output_eval = layer(input_tensor)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
        bias: bool = True,
    ):
        """Initialize NoisyLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            sigma_init: Initial value for noise scale parameters.
                Rainbow paper uses 0.5 for factorized noise.
            bias: If True, adds a learnable bias. Default: True.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.use_bias = bias

        # Learnable mean parameters (μ)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))

        # Learnable noise scale parameters (σ)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_sigma", None)

        # Factorized noise buffers (not learnable parameters)
        # These are registered as buffers so they move with the model to GPU
        # persistent=False ensures noise is not saved in state_dict (should be resampled on load)
        self.register_buffer(
            "weight_epsilon", torch.empty(out_features, in_features), persistent=False
        )
        if bias:
            self.register_buffer(
                "bias_epsilon", torch.empty(out_features), persistent=False
            )
        else:
            self.register_buffer("bias_epsilon", None)

        # Initialize parameters and noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters.

        Weight means are initialized uniformly in [-1/√fan_in, 1/√fan_in].
        Noise scales are initialized to sigma_init / √fan_in.

        This follows the initialization scheme from the NoisyNet paper.
        """
        # Mean parameters: uniform initialization like standard Linear
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)

        # Noise scale: constant initialization
        sigma_value = self.sigma_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_value)

        if self.use_bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(sigma_value)

    def reset_noise(self) -> None:
        """Sample new factorized noise.

        Generates noise vectors ε_in and ε_out, then computes factorized
        noise as their outer product. This is more efficient than sampling
        a full noise matrix.

        Call this method at the start of each forward pass during training
        to get fresh exploration noise.
        """
        # Sample noise vectors
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Factorized noise: outer product of scaled vectors
        # weight_epsilon[i,j] = f(epsilon_out[i]) * f(epsilon_in[j])
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))

        if self.use_bias and self.bias_epsilon is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise vector.

        Applies the scaling function f(x) = sign(x) · √|x| to standard
        Gaussian noise. This scaling is used in factorized NoisyNet.

        Args:
            size: Length of the noise vector.

        Returns:
            Scaled noise tensor of shape (size,).
        """
        # Sample from standard Gaussian
        x = torch.randn(size, device=self.weight_mu.device)

        # Apply f(x) = sign(x) * sqrt(|x|)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional noisy weights.

        During training (self.training=True), uses noisy weights:
            W_noisy = μ_w + σ_w ⊙ ε_w
            b_noisy = μ_b + σ_b ⊙ ε_b

        During evaluation (self.training=False), uses mean weights only:
            W = μ_w
            b = μ_b

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).
        """
        if self.training:
            # Training: use noisy weights
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            if self.use_bias:
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                bias = None
        else:
            # Evaluation: use mean weights only (no noise)
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"sigma_init={self.sigma_init}, "
            f"bias={self.use_bias}"
        )


def reset_noise_recursive(module: nn.Module) -> None:
    """Reset noise in all NoisyLinear layers within a module.

    Recursively traverses the module tree and calls reset_noise()
    on all NoisyLinear layers found.

    Args:
        module: PyTorch module (potentially containing NoisyLinear layers).

    Example:
        >>> model = MyNetwork(use_noisy=True)
        >>> reset_noise_recursive(model)  # Resets all NoisyLinear layers
    """
    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.reset_noise()
