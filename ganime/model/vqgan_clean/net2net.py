from tensorflow.keras import Model, layers

from ganime.configs.model_configs import GPTConfig, ModelConfig


class Net2Net(Model):
    def __init__(
        self,
        transformer_config: GPTConfig,
        first_stage_config: ModelConfig,
        cond_stage_config: ModelConfig,
    ):
        pass
