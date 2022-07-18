from dataclasses import dataclass
from typing import List, Literal


@dataclass
class GPTConfig:
    n_layer: int
    n_head: int
    n_embedding: int
    vocab_size: int
    block_size: int
    embedding_percentage_drop: float
    attention_percentage_drop: float


@dataclass
class VQVAEConfig:
    beta: float
    num_embeddings: int
    embedding_dim: int


@dataclass
class AutoencoderConfig:
    z_channels: int
    channels: int
    channels_multiplier: List[int]
    num_res_blocks: int
    attention_resolution: List[int]
    resolution: int
    dropout: float


@dataclass
class DiscriminatorConfig:
    num_layers: int
    filters: int


@dataclass
class DiscriminatorLossConfig:
    loss: Literal["hinge, vanilla"]
    factor: float
    iter_start: int
    weight: float


@dataclass
class VQVAELossConfig:
    codebook_weight: float
    perceptual_weight: float


@dataclass
class LossConfig:
    discriminator: DiscriminatorLossConfig
    vqvae: VQVAELossConfig


@dataclass
class ModelConfig:
    vqvae_config: VQVAEConfig
    autoencoder_config: AutoencoderConfig
    discriminator_config: DiscriminatorConfig
    loss_config: LossConfig
