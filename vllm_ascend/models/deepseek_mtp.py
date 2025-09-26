#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/deepseek_mtp.py
# Copyright 2023 The vLLM team.
#
# This file is a part of the vllm-ascend project.
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

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
                         get_current_vllm_config)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP, DeepSeekMultiTokenPredictor, DeepSeekMultiTokenPredictorLayer,
    SharedHead)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors


class CustomDeepSeekShareHead(SharedHead):

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size,
                                   config.hidden_size,
                                   quant_config=quant_config,
                                   prefix=maybe_prefix(prefix, "head"))


class CustomDeepSeekMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        print("try_get_current_vllm_config, CustomDeepSeekMultiTokenPredictorLayer--------------------------------------")
        vllm_config = get_current_vllm_config()

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2,
                                 config.hidden_size,
                                 bias=False)
        self.shared_head = CustomDeepSeekShareHead(config=config,
                                                   quant_config=quant_config,
                                                   prefix=maybe_prefix(
                                                       prefix, "shared_head"))
        self.mtp_block = DeepseekV2DecoderLayer(vllm_config=vllm_config,
                                                prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where((positions == 0).unsqueeze(-1),
                                    torch.zeros_like(inputs_embeds),
                                    inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        hidden_states, residual = self.mtp_block(positions=positions,
                                                 hidden_states=hidden_states,
                                                 residual=None)
        hidden_states = residual + hidden_states
        return hidden_states


class CustomDeepSeekMultiTokenPredictor(DeepSeekMultiTokenPredictor):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict({
            str(idx):
            CustomDeepSeekMultiTokenPredictorLayer(
                config,
                f"{prefix}.layers.{idx}",
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
            )
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        })
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Note: torch._dynamo.exc.Unsupported: builtin: str
        self.layers_list = [
            self.layers[str(idx)]
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        ]
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: torch.Tensor,
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        step_kv_cache = kv_caches[
            current_step_idx] if kv_caches is not None else None
        return self.layers_list[current_step_idx](
            input_ids,
            positions,
            step_kv_cache,
            attn_metadata,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,  # type: ignore
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        mtp_layer = self.layers_list[current_step_idx]
        logits = self.logits_processor(mtp_layer.shared_head.head,
                                       mtp_layer.shared_head(hidden_states),
                                       sampling_metadata)
        return logits


class CustomDeepSeekMTP(DeepSeekMTP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.model = CustomDeepSeekMultiTokenPredictor(vllm_config=vllm_config,
                                                       prefix=maybe_prefix(
                                                           prefix, "model"))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, previous_hidden_states,
                                   inputs_embeds, spec_step_idx)
        return hidden_states
