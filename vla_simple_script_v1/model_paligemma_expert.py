# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import torch
import torch.version
from pytest import Cache
from torch import nn
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING
from model_gemma_expert import Gemma2Model
from model_config import Gemma2Config

# from lerobot.common.policies.pi0.flex_attention import flex_attention_forward
from dataclasses import dataclass


@dataclass
class ActionExpertConfig:

    hidden_size = 1024
    intermediate_size = 2048
    hidden_activation = "gelu_pytorch_tanh"


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        d_half, dtype=torch.float32, device=device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(
        torch.float32
    )

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        self.paligemma = PaliGemmaForConditionalGeneration(
            config=config.paligemma_config
        )
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        # self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.model.embed_tokens(tokens)

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):

        # models = [self.paligemma.language_model.model, self.gemma_expert.model]

        action_expert_config = ActionExpertConfig()

        test_config = Gemma2Config(
            action_expert_config=action_expert_config,
        )
        def get_device(inputs_embeds):
            if isinstance(inputs_embeds, (list, tuple)) and inputs_embeds[0] is not None:
                return inputs_embeds[0].device
            else:
                return inputs_embeds[1].device
        device = get_device(inputs_embeds)
        test_model = Gemma2Model(test_config).to(device=device)

        if isinstance(inputs_embeds, List):
            inputs_embeds = tuple(inputs_embeds)

        return test_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # position_ids = position_ids,
            use_cache=True,
            return_dict=True,
        )
        
    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self,
        attention_mask,
        batch_size,
        head_dim,
        query_states,
        key_states,
        value_states,
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self,
        attention_mask,
        batch_size,
        head_dim,
        query_states,
        key_states,
        value_states,
    ):
        num_att_heads = self.config.paligemma_config.text_config.num_attention_heads
        num_key_value_heads = (
            self.config.paligemma_config.text_config.num_key_value_heads
        )
        num_key_value_groups = num_att_heads // num_key_value_heads

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        masked_att_weights = torch.where(
            attention_mask[:, None, :, :], att_weights, big_neg
        )

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )

        return att_output


def test_paligemma_with_expert():
    print("开始测试 PaliGemmaWithExpertModel...")

    # 1. 创建配置
    print("创建模型配置...")
    config = PaliGemmaWithExpertConfig()
    print("初始化模型...")
    model = PaliGemmaWithExpertModel(config)

    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)

    # 3. 准备输入数据
    print("准备测试输入数据...")
    batch_size = 2
    seq_len = 10
    hidden_dim1 = 2048  # 主模型嵌入维度
    hidden_dim2 = 1024  # 动作专家嵌入维度

    # 创建模拟的嵌入
    main_embeds = torch.randn(batch_size, seq_len, hidden_dim1, device=device)
    action_embeds = torch.randn(batch_size, seq_len // 2, hidden_dim2, device=device)
    inputs_embeds = (main_embeds, action_embeds)

    # 创建注意力掩码
    # 准备合并后的掩码
    prefix_mask = torch.ones((batch_size, seq_len), device=device)
    subfix_mask = torch.ones((batch_size, seq_len // 2), device=device)

    # 创建标准的注意力掩码
    attention_mask = torch.zeros(
        (batch_size, 1, seq_len + seq_len // 2, seq_len + seq_len // 2), device=device
    )
    # 允许看到自己和之前的token
    for i in range(seq_len + seq_len // 2):
        attention_mask[:, :, :, : i + 1] = 1

    # 4. 执行前向传播
    print("执行模型前向传播...")
    with torch.no_grad():
        try:
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
            )

            # 5. 验证输出
            print("\n输出验证:")
            # if isinstance(outputs, tuple):
            #     print(f"输出是一个元组，长度为: {len(outputs)}")
            #     hidden_states, past_key_values = outputs
            # else:
            print(f"输出类型: {type(outputs)}")
            hidden_states = outputs[0]
            # past_key_values = outpußts.past_key_values

            print(f"隐藏状态类型: {type(hidden_states)}")
            if isinstance(hidden_states, tuple):
                print(f"主模型最终隐藏状态形状: {hidden_states[0].shape}")
                print(f"动作专家最终隐藏状态形状: {hidden_states[1].shape}")

            # print(f"缓存类型: {type(past_key_values)}")
            print("测试成功完成!")

        except Exception as e:
            print(f"测试遇到错误: {e}")
            raise


# if __name__ == "__main__":
#     test_paligemma_with_expert()
