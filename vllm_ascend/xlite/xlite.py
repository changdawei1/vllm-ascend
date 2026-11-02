#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#
"""Xlite integration module for vLLM-Ascend."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import regex as re
import torch
import torch.nn as nn
import torch_npu
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from xlite._C import (
    AttnCxA,
    AttnDSA,
    AttnMeta,
    AttnMetaV2,
    AttnMHA,
    AttnMLA,
    Runtime,
    ScoringFuncSigmoid,
    ScoringFuncSoftmax,
    ScoringFuncSqrtsoftplus,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding
from vllm_ascend.xlite.utils import (
    _DUMMY_TENSOR,
    AttnMetadataRouter,
    WeightGetterConfig,
    XModel,
    XModelConfig,
    get_dotted_attr,
    get_layer_weights,
    set_dummy_tensor,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v1.xlite.xlite_model_runner import XliteModelRunner as XliteModelRunnerV1
    from vllm_ascend.xlite.xlite_model_runner import XliteModelRunnerV2

    XliteModelRunner: TypeAlias = XliteModelRunnerV1 | XliteModelRunnerV2

XliteForwardResult: TypeAlias = torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]

_architecture_strategy_map: dict[str, type[XliteModelBase]] = {}
"""Mapping from model architecture names in `config.json` to their corresponding xlite adapter classes."""


class XliteModelBase(ABC):
    """Base adapter for converting vLLM models into xlite runtime models.

    Subclasses are responsible for mapping architecture-specific configuration and weights into the `xlite._C.Model`
    interface.

    Attributes:
        npu_runner (XliteModelRunner): The NPU model runner instance.
        npu_runnable (nn.Module): The original runnable model used by vLLM-ascend. Used as the source of truth for
            weight extraction for xlite model construction.
        vllm_config (VllmConfig): The configuration object provided by vLLM. Used to build xlite configuration at
            runtime.
        device (torch.device): The device on which the model is loaded. Typically an Ascend NPU device.
        dtype (torch.dtype): The data type used for model weights and computations. Derived from `vllm_config`.
        rottary_embed (AscendRotaryEmbedding | None): The rotary embedding module extracted from the NPU runnable model.
        xlite_config (XModelConfig): Native xlite configuration object populated by subclasses.
        xlite_model (XModel): Native xlite model container populated by subclasses.
        rope_cossin_cache (torch.Tensor): Precomputed RoPE frequency cache tensor for rotary positional embeddings.
    """

    from vllm_ascend.attention.attention_v1 import AscendMetadata

    _attn_metadata_type: type | tuple[type, ...] = AscendMetadata
    """The expected type of attention metadata in the forward context for this architecture. Used for runtime checks
    before forwarding. See :meth:`XliteWrapper.__call__` for usage."""
    _supported_architectures: Sequence[str] | str
    """The list of model architecture names (from HuggingFace `config.json` "architectures" field) supported by this
    adapter. Used for automatic adapter selection and registration."""
    _decoder_layer_mlp_module: str = "mlp"
    """The module name used to identify MLP modules (including MoE) in the decoder layers (nn.Module) of the runnable
    model: e.g., in `Glm4MoeForCausalLM`, the MLP modules need to be accessed via `.model.layers[i].mlp`, thus `mlp`."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register subclasses in the architecture strategy map and metadata type set."""
        ts = getattr(cls, "_attn_metadata_type", None)
        if ts is None or (not isinstance(ts, type) and not all(isinstance(t, type) for t in ts)):
            raise ValueError(
                f"XliteModel subclass {cls.__name__} must define _attn_metadata_type as a type or a tuple of types."
            )

        arcs = getattr(cls, "_supported_architectures", None)
        if arcs is None:
            raise ValueError(f"XliteModel subclass {cls.__name__} must define _supported_architectures attribute.")
        if isinstance(arcs, str):
            arcs = [arcs]
        for arc in arcs:
            if arc in _architecture_strategy_map:
                raise ValueError(f"Duplicate xlite adapter for architecture {arc}: {_architecture_strategy_map[arc]}")
            _architecture_strategy_map[arc] = cls
        super().__init_subclass__(**kwargs)

    def __init__(self, npu_runner: XliteModelRunner, vllm_config: VllmConfig) -> None:
        """Initialize the xlite model adapter.

        Args:
            npu_runner (XliteModelRunner): The NPU model runner instance.
            vllm_config (VllmConfig): Runtime configuration used for model setup.

        Notes:
            The constructor stores the runnable model and vLLM config, and prepares empty xlite configuration and model
            containers for subclass-specific population.
        """
        self.npu_runner: XliteModelRunner = npu_runner
        self.npu_runnable: nn.Module = npu_runner.fallback_model
        self.vllm_config = vllm_config
        self.device = self.npu_runner.device
        self.dtype = vllm_config.model_config.dtype
        self.rottary_embed: AscendRotaryEmbedding | None = self._get_rope()
        if self.rottary_embed:
            logger.info_once(
                "xlite: found rotary embedding module `%s` in the NPU runnable.",
                re.sub(r"\s+", " ", repr(self.rottary_embed).replace("\n", "").strip()),
            )
        else:
            logger.warning_once("xlite: no rotary embedding module found in `%s`", self.npu_runnable)

        self.xlite_config = XModelConfig()
        self.xlite_model = XModel()
        self._build_model_config()
        self._build_model()
        self.xlite_model.init(self.xlite_config, torch.distributed.get_rank())
        self.rope_cossin_cache = self._precompute_rope_cache()

        if torch.distributed.get_rank() == 0:
            logger.info_once("xlite: built model config: %s", self.xlite_config)
            logger.info_once("xlite: built model weights: %s", self.xlite_model)

    def __del__(self) -> None:
        self.npu_runner = None  # type: ignore[assignment]
        self.npu_runnable = None  # type: ignore[assignment]

    @abstractmethod
    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        """Extract xlite-compatible KV cache from the vLLM-ascend KV cache.

        Args:
            kv_caches (list[torch.Tensor]): The KV cache from the upstream vLLM/vLLM-ascend.

        Returns:
            list[tuple[torch.Tensor, ...]]: The KV cache in the format expected by xlite model implementations.
        """

    @abstractmethod
    def _build_model_config(self) -> None:
        """Build architecture-specific xlite model configuration.

        This method extracts necessary configuration attributes from the vLLM config (e.g., HuggingFace metadata) and
        populates an xlite :class:`ModelConfig` object.

        Returns:
            None: `self` attribute :attr:`xlite_config` is updated in-place.
        """

    @abstractmethod
    def _build_model(self) -> None:
        """Build architecture-specific xlite model weights.

        This method traverses the runnable model's parameters and maps them into the xlite :class:`Model` interface
        according to the architecture's specific structure.

        Returns:
            None: `self` attribute :attr:`xlite_model` is updated in-place.

        Notes:
            :meth:`_build_model_config` should be called prior to this method to ensure the xlite configuration is
            populated before weight mapping.
        """

    def _get_layers_and_model_prefix(self) -> tuple[Sequence[nn.Module], str]:
        """Extract transformer layers and parameter prefix from runnable.

        Returns:
            tuple[Sequence[nn.Module], str]: A pair of `(layers, model_prefix)` for model traversal.
        """
        if hasattr(self.npu_runnable, "language_model"):
            layers = cast(
                Sequence[nn.Module], get_dotted_attr(self.npu_runnable.language_model, "model.layers", default=[])
            )
            prefix = "language_model."
        else:
            layers = cast(Sequence[nn.Module], get_dotted_attr(self.npu_runnable, "model.layers", default=[]))
            prefix = ""
        return layers, prefix

    def _get_rope(self, layer_idx: int = 0) -> AscendRotaryEmbedding | None:
        """Extract the rotary embedding module from the NPU runnable model :data:`npu_runnable`.

        Args:
            layer_idx (int): The index of the transformer layer to extract the rotary embedding from. Defaults to 0.

        Returns:
            AscendRotaryEmbedding | None: The rotary embedding module found in the specified transformer layer, or None
            if not found.
        """
        layers, _ = self._get_layers_and_model_prefix()
        rottary_embed: AscendRotaryEmbedding | None = get_dotted_attr(layers, f"{layer_idx}.self_attn.rotary_emb")
        if not rottary_embed:
            logger.warning_once("xlite adapter: Rotary embedding not found in the model layers. RoPE will be disabled.")
        return rottary_embed

    def _precompute_rope_cache(self) -> torch.Tensor:
        """Precomputes frequency-based complex exponential values for rotary positional embeddings (RoPE).

        This method generates the RoPE frequency cache (cosine and sine values) required by the xlite attention
        implementation. The cache should be precomputed on the NPU device to avoid unnecessary host-device transfers
        during inference.

        Returns:
            torch.Tensor: The precomputed RoPE frequency cache tensor ready for use in xlite attention computations.

        Notes:
            :meth:`_build_model_config` should be called prior to this method.
        """
        if not self.rottary_embed:
            return torch.empty(2, dtype=torch.float32, device=self.device)
        cossin_cache: torch.Tensor | None = getattr(self.rottary_embed, "cossin_cache", None)
        if not isinstance(cossin_cache, torch.Tensor):
            cossin_cache = self.rottary_embed._compute_cos_sin_cache()
        return cossin_cache.to(dtype=torch.float32, device=self.device)

    @staticmethod
    def is_tensor_nz(t: torch.Tensor) -> bool:
        """Check if a tensor is in NZ format.

        Args:
            t (torch.Tensor): The tensor to check.

        Returns:
            bool: True if the tensor is in NZ format, False otherwise.
        """
        format = torch_npu.get_npu_format(t)
        return format == torch_npu.Format.FRACTAL_NZ

    @staticmethod
    def all_tensors_zero(tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None) -> bool:
        """Check if all tensors in the list/tuple are zero tensors.

        Args:
            tensors (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None): The tensors to check.

        Returns:
            bool: True if all tensors are zero tensors (or empty), False otherwise.
        """
        if tensors is None:
            return True
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        if len(tensors) == 0:
            return True
        return all(torch.allclose(t, t.new_zeros(1)) for t in tensors)

    @staticmethod
    def _transform_deq_scale(deq_scale: torch.Tensor) -> torch.Tensor:
        """Repack a dequantization scale into the fixpipe hardware's expected format.

        Data is stored in ``uint64_t``: the upper 32 bits are 0 and the lower 32 bits hold an FP32 value whose lower 10
        bits are not involved in computation, making the effective data format TF32.

        Args:
            deq_scale (torch.Tensor): The original dequantization scale tensor.

        Returns:
            torch.Tensor: The repacked scale tensor for fixpipe computation.
        """
        if deq_scale.dtype == torch.int64:
            return deq_scale
        deq_scale_fp32 = deq_scale.to(torch.float32)
        scale = deq_scale_fp32.new_zeros(deq_scale.shape[0] * 2)
        scale[0::2] = deq_scale_fp32[0::1]
        return scale

    @property
    def hf_text_config(self) -> PretrainedConfig:
        """Convenience property to access HuggingFace text configuration from vLLM config.

        Returns:
            PretrainedConfig: The HuggingFace text configuration object extracted from vLLM config.
        """
        hf_config = self.vllm_config.model_config.hf_text_config
        return cast(PretrainedConfig, getattr(hf_config, "text_config", hf_config))

    @property
    def hf_vision_config(self) -> PretrainedConfig | None:
        """Convenience property to access HuggingFace vision configuration from vLLM config, if exists.

        Returns:
            PretrainedConfig | None: The HuggingFace vision configuration object extracted from vLLM config, or None if
                not present.
        """
        return getattr(self.vllm_config.model_config.hf_config, "vision_config", None)


class StandardXliteModel(XliteModelBase):
    """xlite adapter base for standard architectures.

    This is the *de facto* base adapter for all xlite-supported architectures. Configurations for various model types,
    dense and MoE, are included.

    Subclasses of :class:`XliteModelBase`, i.e., xlite model adapters, should inherit from :class:`StandardXliteModel`,
    unless there is a major divergence.

    **Developer Guide**: The :class:`StandardXliteModel` class is designed to be a flexible base for xlite adapters.
    When developing a new adapter for a specific architecture to dock with the xlite backend, consider adding weight
    loading lines in :meth:`StandardXliteModel._build_model` directly if feasible, using the highly flexible and
    error-checked `get_layer_weights` utility. For :meth:`_build_model_config`, override it in the subclass only if the
    architecture has unique configuration needs.
    """

    _supported_architectures = [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3VLForConditionalGeneration",
    ]

    def _build_model_config(self) -> None:
        xlite_config, vllm_config, hf_config = self.xlite_config, self.vllm_config, self.hf_text_config

        xlite_config.vocab_size = hf_config.vocab_size
        xlite_config.hidden_size = hf_config.hidden_size
        xlite_config.n_layers = hf_config.num_hidden_layers
        xlite_config.n_heads = hf_config.num_attention_heads
        xlite_config.n_kv_heads = hf_config.num_key_value_heads
        if hasattr(hf_config, "head_dim"):
            xlite_config.head_dim = hf_config.head_dim
        else:
            xlite_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        xlite_config.rope_head_dim = self.rottary_embed.rotary_dim if self.rottary_embed else xlite_config.head_dim
        xlite_config.norm_eps = hf_config.rms_norm_eps
        if hasattr(hf_config, "rope_theta"):
            xlite_config.rope_theta = hf_config.rope_theta
        else:
            xlite_config.rope_theta = getattr(hf_config, "rope_parameters", {}).get("rope_theta", 10000.0)
        xlite_config.softmax_scale = xlite_config.head_dim**-0.5
        xlite_config.n_dense_layers = hf_config.num_hidden_layers
        xlite_config.intermediate_size = hf_config.intermediate_size
        xlite_config.def_tp_size = tp_size = get_tensor_model_parallel_world_size()
        xlite_config.def_dp_size = vllm_config.parallel_config.data_parallel_size
        try:
            ep_word_size = get_ep_group().world_size
            xlite_config.moe_ep_size = ep_word_size if vllm_config.parallel_config.enable_expert_parallel else 1
            xlite_config.moe_tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else ep_word_size
        except AssertionError:
            xlite_config.moe_ep_size, xlite_config.moe_tp_size = 1, 1
        xlite_config.experts_weight_transpose = True

        xlite_config.attn_type = AttnMHA
        xlite_config.scoring_func = ScoringFuncSoftmax
        xlite_config.weight_nz = get_ascend_config().weight_nz_mode == 2
        max_m = vllm_config.scheduler_config.max_num_batched_tokens
        if not get_ascend_config().xlite_graph_config.full_mode:
            # decode-sized token budget: max_num_seqs * (1 + num_speculative_tokens) instead of max_num_batched_tokens
            n_spec_tokens = getattr(vllm_config.speculative_config, "num_speculative_tokens", 0)
            max_m = min(max_m, vllm_config.scheduler_config.max_num_seqs * (n_spec_tokens + 1))
        xlite_config.max_m = (max_m + tp_size - 1) // tp_size * tp_size  # round up to nearest multiple of tp_size
        xlite_config.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        xlite_config.max_seq_len = vllm_config.model_config.max_model_len
        xlite_config.block_size = vllm_config.cache_config.block_size

        rope_parameters = getattr(hf_config, "rope_parameters", {})
        xlite_config.deepstack_num_level = len(getattr(self.hf_vision_config, "deepstack_visual_indexes", []))
        xlite_config.mrope_section = rope_parameters.get("mrope_section", [])
        xlite_config.mrope_interleaved = rope_parameters.get("mrope_interleaved", False)
        self.quantization = vllm_config.quant_config is not None

    def _build_model(self) -> None:
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, model_prefix = self._get_layers_and_model_prefix()

        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)

        xlite_model.attn_norm = get_layer_weights(layers, "input_layernorm.weight")
        self.init_matmul_weights(layers, "mha_qkv", "self_attn.qkv_proj")
        self.init_matmul_weights(layers, "attn_out", "self_attn.o_proj")

        with xlite_model.condition(lambda tensors: len(tensors) == xlite_config.n_layers):
            xlite_model.mha_q_norm = get_layer_weights(layers, "self_attn.q_norm.weight")
            xlite_model.mha_k_norm = get_layer_weights(layers, "self_attn.k_norm.weight")
            xlite_config.qk_norm = bool(xlite_model.mha_q_norm) and bool(xlite_model.mha_k_norm)

        # Dense MLP weights
        self.init_matmul_weights(layers, "mlp_up_gate", "mlp.gate_up_proj")
        self.init_matmul_weights(layers, "mlp_down", "mlp.down_proj")
        xlite_model.mlp_norm = get_layer_weights(layers, "post_attention_layernorm.weight")

        # MoE-specific weights
        mlp_prefix = self._decoder_layer_mlp_module
        xlite_model.gate = get_layer_weights(layers, f"{mlp_prefix}.gate.weight")
        xlite_config.gate_captured = self.is_tensor_nz(xlite_model.gate[0]) if xlite_model.gate else False
        xlite_model.gate_bias = get_layer_weights(
            layers,
            f"{mlp_prefix}.gate.e_score_correction_bias",
            f"{mlp_prefix}.e_score_correction_bias",  # for MiniMax-2.x compatibility
            post_processor=lambda b: b.to(torch.float32),  # type conversion for numerical stability in xlite backend
        )
        self.init_matmul_weights(layers, "se_up_gate", f"{mlp_prefix}.shared_experts.gate_up_proj")
        self.init_matmul_weights(layers, "se_down", f"{mlp_prefix}.shared_experts.down_proj")

        re_prefix = f"{mlp_prefix}.experts.routed_experts"
        re_kwargs: WeightGetterConfig = {"secondary_flattening": f"{re_prefix}.local_num_experts"}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{re_prefix}.w13_weight", **re_kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{re_prefix}.w2_weight", **re_kwargs)
        xlite_config.experts_weight_nz = bool(xlite_model.re_up_gate) and self.is_tensor_nz(xlite_model.re_up_gate[0])

        # bias terms
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mha_qkv_bias = get_layer_weights(layers, "self_attn.qkv_proj.bias")
            xlite_config.qkv_bias = len(xlite_model.mha_qkv_bias) == xlite_config.n_layers
            xlite_model.norm_bias = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.bias", raises=True)
            xlite_model.attn_norm_bias = get_layer_weights(layers, "input_layernorm.bias")
            xlite_model.mlp_norm_bias = get_layer_weights(layers, "post_attention_layernorm.bias")
            if xlite_config.qk_norm:
                xlite_model.mha_q_norm_bias = get_layer_weights(layers, "self_attn.q_norm.bias")
                xlite_model.mha_k_norm_bias = get_layer_weights(layers, "self_attn.k_norm.bias")

        if not self.quantization:
            return

        xlite_config.quant_attn_weight_nz = bool(xlite_model.mha_qkv) and self.is_tensor_nz(xlite_model.mha_qkv[0])
        xlite_config.quant_attn_weight_transpose = bool(xlite_model.mha_qkv)

        xlite_model.re_up_gate_scale_bias = get_layer_weights(layers, f"{re_prefix}.w13_scale_bias", **re_kwargs)
        xlite_model.re_down_scale_bias = get_layer_weights(layers, f"{re_prefix}.w2_scale_bias", **re_kwargs)

        re_kwargs["post_processor"] = self._transform_deq_scale
        xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{re_prefix}.w13_weight_scale", **re_kwargs)
        xlite_model.re_down_scale = get_layer_weights(layers, f"{re_prefix}.w2_weight_scale", **re_kwargs)

    def init_matmul_weights(
        self,
        layers: Sequence[torch.nn.Module],
        xlite_prefix: str,
        model_prefix: str,
        *,
        mask: Sequence[bool] | None = None,
    ) -> None:
        """
        Initialize MatMul-related weights with quantization support.

        Args:
            layers (Sequence[torch.nn.Module]): The transformer layers to extract weights from.
            xlite_prefix (str): The prefix for the xlite model attributes to set.
            model_prefix (str): The prefix for the model attributes to look up in each layer.
            mask (Sequence[bool] | None): Optional mask to select specific layers for weight extraction. See
                :meth:`get_layer_weights` for details.
        """
        xlite_model = self.xlite_model
        setattr(xlite_model, xlite_prefix, get_layer_weights(layers, f"{model_prefix}.weight", mask=mask))
        if not self.quantization:
            return

        def set_xlite_attr(xlite_attr: str, layer_attr: str) -> None:
            setattr(xlite_model, xlite_attr, get_layer_weights(layers, layer_attr, mask=mask))

        wt_kwargs: WeightGetterConfig = {"mask": mask, "post_processor": self._transform_deq_scale}
        deq_scale = get_layer_weights(layers, f"{model_prefix}.deq_scale", **wt_kwargs)
        if len(deq_scale) > 0:  # static quant
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", deq_scale)
            set_xlite_attr(f"{xlite_prefix}_input_scale", f"{model_prefix}.aclnn_input_scale_reciprocal")
            set_xlite_attr(f"{xlite_prefix}_input_offset", f"{model_prefix}.aclnn_input_offset")
            set_xlite_attr(f"{xlite_prefix}_quant_bias", f"{model_prefix}.quant_bias")
        else:
            weight_scale = get_layer_weights(layers, f"{model_prefix}.weight_scale", **wt_kwargs)
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", weight_scale)

    def _precompute_rope_cache(self) -> torch.Tensor:
        # TODO: remove the type conversion once `xlite` accepts and defaults the cache in FP32 format
        return super()._precompute_rope_cache().to(dtype=self.dtype, device=self.device)

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        kv_caches = kv_caches[: 2 * self.xlite_config.n_layers]
        if wrong_dims := set(cache.shape[-1] for cache in kv_caches) - set([self.xlite_config.head_dim]):
            raise ValueError(f"xlite: kv_cache dimension mismatch (wrong dims: {', '.join(map(str, wrong_dims))})")
        return [(k_cache, v_cache) for k_cache, v_cache in zip(kv_caches[0::2], kv_caches[1::2])]


class QwenMoeXliteModel(StandardXliteModel):
    """xlite adapter for Qwen MoE architectures."""

    _supported_architectures = ["Qwen3MoeForCausalLM", "Qwen3VLMoeForConditionalGeneration"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob


class Glm4MoeXliteModel(StandardXliteModel):
    """xlite adapter for GLM4 MoE architectures."""

    _supported_architectures = ["Glm4MoeForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.n_dense_layers = getattr(hf_config, "first_k_dense_replace", 0)
        xlite_config.n_routed_experts = hf_config.n_routed_experts
        xlite_config.n_shared_experts = hf_config.n_shared_experts
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob
        xlite_config.scoring_func = ScoringFuncSigmoid
        xlite_config.route_scale = hf_config.routed_scaling_factor


class DeepseekV3XliteModel(Glm4MoeXliteModel):
    """xlite adapter for DeepseekV3 MoE architectures with MLA attention."""

    from vllm_ascend.attention.mla_v1 import AscendMLAMetadata

    _attn_metadata_type = AscendMLAMetadata  # type: ignore[assignment]
    _supported_architectures = ["DeepseekV3ForCausalLM", "Glm4MoeLiteForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        # MLA attention type
        xlite_config.attn_type = AttnMLA
        xlite_config.n_kv_heads = 1  # MLA uses latent cache
        xlite_config.head_dim = 0  # Ignored by MLA

        # MLA dimensions (override Llama's head_dim-based rope_head_dim)
        xlite_config.rope_head_dim = hf_config.qk_rope_head_dim
        xlite_config.nope_head_dim = hf_config.qk_nope_head_dim
        xlite_config.q_lora_rank = hf_config.q_lora_rank
        xlite_config.kv_lora_rank = hf_config.kv_lora_rank
        xlite_config.v_head_dim = hf_config.v_head_dim
        xlite_config.softmax_scale = (hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim) ** -0.5
        # correct softmax_scale for yarn-style RoPE if max_seq_len > original_max_position_embeddings
        rope_params: dict[str, int | float | str] = getattr(hf_config, "rope_parameters", {})
        original_max_len: int = rope_params.get("original_max_position_embeddings", hf_config.max_position_embeddings)  # type: ignore[assignment]
        if xlite_config.max_seq_len > original_max_len and "mscale" in rope_params and "factor" in rope_params:
            mscale: float = 1.0 + 0.1 * rope_params["mscale"] * math.log(rope_params["factor"])  # type: ignore[operator,arg-type]
            xlite_config.softmax_scale *= mscale**2

        # MoE configuration (from Glm4MoeXliteModel, adapted for DeepseekV3)
        xlite_config.n_expert_groups = getattr(hf_config, "n_group", 1)
        xlite_config.n_limited_groups = getattr(hf_config, "topk_group", 1)

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, _ = self._get_layers_and_model_prefix()

        # MLA attention weights
        self.init_matmul_weights(layers, "mla_qkv_a", "self_attn.fused_qkv_a_proj")
        self.init_matmul_weights(layers, "mla_q_b", "self_attn.q_b_proj")
        xlite_model.mla_q_norm = get_layer_weights(layers, "self_attn.q_a_layernorm.weight")
        xlite_model.mla_kv_norm = get_layer_weights(layers, "self_attn.kv_a_layernorm.weight")
        xlite_model.mla_wuv = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UV")
        xlite_model.mla_wuk_t = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UK_T")
        xlite_config.mla_wuv_weight_nz = bool(wuv_lst := xlite_model.mla_wuv) and self.is_tensor_nz(wuv_lst[0])
        xlite_config.mla_wukt_weight_nz = bool(wuk_lst := xlite_model.mla_wuk_t) and self.is_tensor_nz(wuk_lst[0])

        if not self.quantization:
            return

        self.xlite_config.quant_attn_weight_nz = bool(wt_lst := xlite_model.mla_qkv_a) and self.is_tensor_nz(wt_lst[0])
        self.xlite_config.quant_attn_weight_transpose = True
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mla_q_norm_bias = get_layer_weights(layers, "self_attn.q_a_layernorm.bias")
            xlite_model.mla_kv_norm_bias = get_layer_weights(layers, "self_attn.kv_a_layernorm.bias")

    def _precompute_rope_cache(self) -> torch.Tensor:
        # TODO: remove this method once `xlite` no longer requires complex64 format
        cossin_cache = super(StandardXliteModel, self)._precompute_rope_cache()
        if not self.rottary_embed:
            return cossin_cache.to(dtype=torch.complex64, device=self.device)
        # For MLA attention, we need to return the cache as a complex instead of separate cos/sin tensors
        if not (isinstance(cossin_cache, torch.Tensor) and cossin_cache.ndim == 2 and cossin_cache.shape[1] % 2 == 0):
            raise ValueError("RoPE cache must be a 2D tensor with even second dimension for MLA attention.")
        sep = cossin_cache.shape[-1] // 2
        return torch.complex(cossin_cache[:, :sep], cossin_cache[:, sep:]).to(dtype=torch.complex64, device=self.device)

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        if self.xlite_config.kv_lora_rank == self.xlite_config.rope_head_dim:
            raise RuntimeError("xlite does not support `kv_lora_rank == rope_head_dim` for MLA attention.")
        n_layers = self.xlite_config.n_layers
        kv_nope_caches = list(filter(lambda x: x.shape[-1] == self.xlite_config.kv_lora_rank, kv_caches))[:n_layers]
        k_rope_caches = list(filter(lambda x: x.shape[-1] == self.xlite_config.rope_head_dim, kv_caches))[:n_layers]
        if (n_nope := len(kv_nope_caches)) != n_layers:
            raise ValueError(f"xlite: failed to extract MLA KV nope caches ({n_nope} vs. {n_layers})")
        if (n_rope := len(k_rope_caches)) != n_layers:
            raise ValueError(f"xlite: failed to extract MLA K rope caches ({n_rope} vs. {n_layers})")
        return [(k_nope, k_rope) for k_nope, k_rope in zip(kv_nope_caches, k_rope_caches)]


class DeepseekV32XliteModel(DeepseekV3XliteModel):
    """xlite adapter for Deepseek-V3.2/GLM-5.x architectures with Deepseek sparse attention (DSA)."""

    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

    _attn_metadata_type = AscendSFAMetadata  # type: ignore[assignment]
    _supported_architectures = ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.attn_type = AttnDSA
        xlite_config.index_head_dim = hf_config.index_head_dim
        xlite_config.index_n_heads = hf_config.index_n_heads
        xlite_config.index_topk = hf_config.index_topk
        xlite_config.index_rope_interleaved = getattr(hf_config, "indexer_rope_interleave", False)
        index_types: list[str] = list(map(lambda x: str(x).lower(), getattr(hf_config, "indexer_types", []) or []))
        xlite_config.index_full_mask = list(map(lambda x: x.startswith("full"), index_types))[: xlite_config.n_layers]

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model = self.xlite_model
        layers, _ = self._get_layers_and_model_prefix()

        mask: list[bool] | None = getattr(self.xlite_config, "index_full_mask", None) or None
        self.init_matmul_weights(layers, "index_q_b", f"{(prefix := 'self_attn.indexer')}.wq_b", mask=mask)
        xlite_model.index_k_weights_proj = get_layer_weights(layers, f"{prefix}.wk_weights_proj.weight", mask=mask)
        # xlite backend expects index_k_norm and index_k_norm_bias to have dtype=torch.float32 or the model dtype
        xlite_model.index_k_norm = get_layer_weights(layers, f"{prefix}.k_norm.weight", mask=mask)
        xlite_model.index_k_norm_bias = get_layer_weights(layers, f"{prefix}.k_norm.bias", mask=mask)

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        # For DSA, `kv_caches` is passed as [<index_k_cache (optional)>, kv_nope_cache, k_pe_cache, ..., <mtp_layer>].
        # Under MRV2, all `index_k_cache` are passed first; under V1, they are interleaved with the kv caches.
        # For xlite, the `kv_caches` is expected as [(kv_nope_cache, k_pe_cache, index_k_cache), ...] for each layer.
        # TODO: consider the compatibility with `enable_sparse_sfa_c8` and `enable_sparse_li_c8`
        xcfg = self.xlite_config
        if xcfg.index_head_dim == xcfg.kv_lora_rank or xcfg.index_head_dim == xcfg.rope_head_dim:
            raise RuntimeError("xlite does not support `index_head_dim == kv_lora_rank or rope_head_dim` for DSA.")
        index_mask: list[bool] = (getattr(xcfg, "index_full_mask", None) or [True] * xcfg.n_layers)[: xcfg.n_layers]

        mla_kv_caches: list[torch.Tensor] = list(filter(lambda x: x.shape[-1] != xcfg.index_head_dim, kv_caches))
        mla_caches: list[tuple[torch.Tensor, ...]] = super().extract_kv_cache(mla_kv_caches)
        index_caches: list[torch.Tensor] = list(filter(lambda x: x.shape[-1] == xcfg.index_head_dim, kv_caches))
        if (n_index_layers := sum(index_mask)) != len(index_caches := index_caches[:n_index_layers]):
            raise ValueError(f"xlite: index_k_cache count mismatch ({len(index_caches)} vs. {n_index_layers})")
        if wrong_dims := set(cache.shape[-1] for cache in index_caches) - set([xcfg.index_head_dim]):
            raise ValueError(f"xlite: index_k_cache dimension mismatch (wrong dims: {', '.join(map(str, wrong_dims))})")
        for i, mask in enumerate(index_mask):
            if not mask:
                index_caches.insert(i, _DUMMY_TENSOR)
        return [(mla_cache[0], mla_cache[1], i_cache) for mla_cache, i_cache in zip(mla_caches, index_caches)]


class DeepseekV4XliteModel(DeepseekV3XliteModel):
    """xlite adapter for DeepSeek-V4 with CXA (compress-index-sparse) attention.

    V4 uses a distinct weight set (per-layer compressor & indexer, multi-stage
    hyper-connections) and a 5-tuple KV cache, so neither V3's MLA weights nor
    V3.2's DSA index weights apply. CXA runs only on the V2 device-tensor forward.
    """

    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadata

    _attn_metadata_type = AscendDSAMetadata  # type: ignore[assignment]
    _supported_architectures = ["DeepseekV4ForCausalLM"]
    # V4 CXA only has a V2 device-tensor forward path.
    _uses_v2: bool = True
    _flatten_kv_cache_contiguous_check: bool = False
    _cxa_block_sizes = [8, 128, 128, 32, 128]

    def __init__(self, npu_runner: XliteModelRunner, vllm_config: VllmConfig) -> None:
        try:
            from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

            _bs = vllm_config.cache_config.block_size
            _planes = DSV4_BLOCK_SIZES[_bs][0]  # [mla, swa, c4_state, c128_state]
            self._cxa_block_sizes = [
                _planes[2],  # indexer_state = c4_state
                _planes[0],  # indexer_k    = mla
                _planes[0],  # compress_kv  = mla
                _planes[3],  # state        = c128_state
                _planes[1],  # swa_kv       = swa
            ]
        except Exception:
            pass  # fall back to the hardcoded class attr if the table is unavailable
        super().__init__(npu_runner, vllm_config)

    def _build_model_config(self) -> None:
        hf_config = self.hf_text_config
        if getattr(hf_config, "intermediate_size", None) is None:
            hf_config.intermediate_size = hf_config.moe_intermediate_size
        super(DeepseekV3XliteModel, self)._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.attn_type = AttnCxA
        xlite_config.n_kv_heads = 1
        xlite_config.head_dim = hf_config.head_dim
        xlite_config.rope_head_dim = hf_config.qk_rope_head_dim
        xlite_config.nope_head_dim = hf_config.head_dim - hf_config.qk_rope_head_dim
        xlite_config.v_head_dim = hf_config.head_dim  # v4: v_head_dim == head_dim
        xlite_config.q_lora_rank = hf_config.q_lora_rank
        xlite_config.kv_lora_rank = hf_config.head_dim  # v4: wkv projects dim -> head_dim
        xlite_config.softmax_scale = hf_config.head_dim**-0.5

        # V4 MoE: sqrtsoftplus scoring; hash routing for early layers.
        xlite_config.scoring_func = ScoringFuncSqrtsoftplus
        xlite_config.norm_topk_prob = True
        xlite_config.n_hash_layers = hf_config.num_hash_layers

        # CXA block sizes + per-layer compress ratios.
        xlite_config.block_sizes = list(self._cxa_block_sizes)
        xlite_config.compress_ratios = list(hf_config.compress_ratios)
        # Indexer lives on compress_ratio==4 layers. HF compress_ratios may carry a trailing
        # MTP-draft entry; trim to the decoder layer count.
        xlite_config.index_c8_mask = [r == 4 for r in hf_config.compress_ratios][: xlite_config.n_layers]

        # V4-specific scalar fields.
        xlite_config.o_groups = hf_config.o_groups
        xlite_config.o_lora_rank = hf_config.o_lora_rank
        xlite_config.window_size = hf_config.sliding_window
        xlite_config.compress_rope_theta = hf_config.compress_rope_theta
        xlite_config.index_head_dim = hf_config.index_head_dim
        xlite_config.index_n_heads = hf_config.index_n_heads
        xlite_config.index_topk = hf_config.index_topk
        rope_params: dict[str, Any] = getattr(hf_config, "rope_parameters", {}) or getattr(
            hf_config, "rope_scaling", {}
        )
        xlite_config.original_seq_len = rope_params.get(
            "original_max_position_embeddings", getattr(hf_config, "max_position_embeddings", 0)
        )
        xlite_config.rope_factor = rope_params.get("factor", 1.0)
        xlite_config.beta_fast = rope_params.get("beta_fast", getattr(hf_config, "beta_fast", 32))
        xlite_config.beta_slow = rope_params.get("beta_slow", getattr(hf_config, "beta_slow", 1))
        xlite_config.hc_mult = hf_config.hc_mult
        xlite_config.hc_sinkhorn_iters = hf_config.hc_sinkhorn_iters
        xlite_config.hc_eps = hf_config.hc_eps
        xlite_config.swiglu_limit = hf_config.swiglu_limit

    def _build_model(self) -> None:
        # V4 CXA shares neither V3's MLA weights nor Standard's MHA/dense-MLP weights.
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, model_prefix = self._get_layers_and_model_prefix()
        compress_ratios = list(xlite_config.compress_ratios)
        # Trim a possible trailing MTP-draft entry so the list matches the decoder layers.
        compress_ratios = compress_ratios[: len(layers)]
        compress_mask = [r > 0 for r in compress_ratios]  # layers with a compressor
        indexer_mask = [r == 4 for r in compress_ratios]  # layers with an indexer

        # Norm kernels read weights as bf16 (matching native RMSNorm's weight.to(dtype));
        # fp32 would be byte-reinterpreted and corrupt, so cast every norm weight to bf16.
        def _bf16(w: torch.Tensor) -> torch.Tensor:
            return w.to(torch.bfloat16).contiguous()

        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = _bf16(get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True))
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)
        xlite_model.attn_norm = get_layer_weights(layers, "input_layernorm.weight", post_processor=_bf16)
        xlite_model.mlp_norm = get_layer_weights(layers, "post_attention_layernorm.weight", post_processor=_bf16)

        # CxA attention weights. wq_a/wkv/wq_b are W8A8_DYNAMIC (int8 + weight_scale); wo_a/wo_b
        # are FLOAT (bf16) read directly via get_layer_weights.
        self.init_matmul_weights(layers, "attn_wq_a", "self_attn.wq_a")
        self.init_matmul_weights(layers, "attn_wkv", "self_attn.wkv")
        self.init_matmul_weights(layers, "mla_q_b", "self_attn.wq_b")  # wq_b reuses mla_q_b field
        self.init_matmul_weights(layers, "idx_wq_b", "self_attn.indexer.wq_b", mask=indexer_mask)

        # vllm stores wo_a grouped as [1, in, out] for its batched o-proj; xlite wants 2D
        # [o_lora_rank, hidden], so squeeze the group dim and transpose. wo_b is already 2D.
        def _wo_a_view(w: torch.Tensor) -> torch.Tensor:
            if w.ndim == 3:
                w = w.squeeze(0)  # [in, out]
            return w.t().contiguous() if w.shape[0] != self.xlite_config.o_lora_rank else w

        xlite_model.attn_wo_a = get_layer_weights(layers, "self_attn.wo_a.weight", post_processor=_wo_a_view)
        xlite_model.attn_wo_b = get_layer_weights(layers, "self_attn.wo_b.weight")
        xlite_model.attn_sink = get_layer_weights(layers, "self_attn.attn_sink")
        xlite_model.mla_q_norm = get_layer_weights(layers, "self_attn.q_norm.weight", post_processor=_bf16)
        xlite_model.mla_kv_norm = get_layer_weights(layers, "self_attn.kv_norm.weight", post_processor=_bf16)
        xlite_model.attn_out = []  # v4 has no separate attn output projection

        # Compressor (exists iff compress_ratio > 0): ape (fp32 param), wkv/wgate (fp32),
        # norm (bf16 to match the kernel's bf16 weight read).
        xlite_model.comp_ape = get_layer_weights(layers, "self_attn.compressor.ape", mask=compress_mask)
        # comp_w_kv/comp_w_gate MUST be fp32: the compressor GEMM output is fp32, so a bf16
        # weight would mismatch the output dtype and throw.
        xlite_model.comp_w_kv = get_layer_weights(
            layers,
            "self_attn.compressor.wkv.weight",
            mask=compress_mask,
            post_processor=lambda w: w.to(torch.float32).contiguous(),
        )
        xlite_model.comp_w_gate = get_layer_weights(
            layers,
            "self_attn.compressor.wgate.weight",
            mask=compress_mask,
            post_processor=lambda w: w.to(torch.float32).contiguous(),
        )
        xlite_model.comp_norm = get_layer_weights(
            layers,
            "self_attn.compressor.norm.weight",
            mask=compress_mask,
            post_processor=lambda w: w.to(torch.bfloat16).contiguous(),
        )

        # Indexer (exists iff compress_ratio == 4).
        xlite_model.idx_weights_proj = get_layer_weights(
            layers, "self_attn.indexer.weights_proj.weight", mask=indexer_mask
        )
        xlite_model.idx_comp_ape = get_layer_weights(layers, "self_attn.indexer.compressor.ape", mask=indexer_mask)
        # Same fp32 requirement as comp_w_kv/w_gate (indexer GEMM output is hardcoded fp32).
        xlite_model.idx_comp_w_kv = get_layer_weights(
            layers,
            "self_attn.indexer.compressor.wkv.weight",
            mask=indexer_mask,
            post_processor=lambda w: w.to(torch.float32).contiguous(),
        )
        xlite_model.idx_comp_w_gate = get_layer_weights(
            layers,
            "self_attn.indexer.compressor.wgate.weight",
            mask=indexer_mask,
            post_processor=lambda w: w.to(torch.float32).contiguous(),
        )
        xlite_model.idx_comp_norm = get_layer_weights(
            layers,
            "self_attn.indexer.compressor.norm.weight",
            mask=indexer_mask,
            post_processor=lambda w: w.to(torch.bfloat16).contiguous(),
        )

        # Multi-stage Hyper-Connections (per-layer direct params on the decoder layer).
        for attr in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale"):
            setattr(xlite_model, attr, get_layer_weights(layers, attr))
        # Transformer-level MHC head (direct params on the model).
        xlite_model.hc_head_fn = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.hc_head_fn", raises=True)
        xlite_model.hc_head_base = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.hc_head_base", raises=True)
        xlite_model.hc_head_scale = get_dotted_attr(
            self.npu_runnable, f"{model_prefix}model.hc_head_scale", raises=True
        )

        # MoE: gate (FLOAT), gate_bias (e_score_correction_bias, non-hash layers only),
        # tid2eid (hash layers only), shared + routed experts (W8A8_DYNAMIC).
        mlp_prefix = self._decoder_layer_mlp_module
        xlite_model.gate = get_layer_weights(layers, f"{mlp_prefix}.gate.weight")
        xlite_config.gate_captured = self.is_tensor_nz(xlite_model.gate[0]) if xlite_model.gate else False
        # gate_bias: fp32 for topk layers, empty tensor for hash layers (which use tid2eid).
        # Keep the list n_layers-long so runtime can index by absolute layer id.
        xlite_model.gate_bias = [
            (
                b.to(torch.float32)
                if (b := get_dotted_attr(layer, f"{mlp_prefix}.gate.e_score_correction_bias")) is not None
                else torch.empty(0, dtype=torch.float32)
            )
            for layer in layers
        ]
        # tid2eid is hash-layers only; vllm sets it None on topk layers, so get_layer_weights
        # (which skips None) yields exactly the hash ones.
        xlite_model.tid2eid = [t for t in get_layer_weights(layers, f"{mlp_prefix}.gate.tid2eid") if t.numel() > 0]
        self.init_matmul_weights(layers, "se_up_gate", f"{mlp_prefix}.shared_experts.gate_up_proj")
        self.init_matmul_weights(layers, "se_down", f"{mlp_prefix}.shared_experts.down_proj")

        re_prefix = f"{mlp_prefix}.experts.routed_experts"
        re_kwargs: WeightGetterConfig = {"secondary_flattening": f"{re_prefix}.local_num_experts"}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{re_prefix}.w13_weight", **re_kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{re_prefix}.w2_weight", **re_kwargs)
        xlite_config.experts_weight_nz = bool(xlite_model.re_up_gate) and self.is_tensor_nz(xlite_model.re_up_gate[0])
        xlite_config.experts_weight_transpose = True
        if self.quantization:
            re_kwargs["post_processor"] = self._transform_deq_scale
            xlite_model.re_up_gate_deq_scale = get_layer_weights(layers, f"{re_prefix}.w13_weight_scale", **re_kwargs)
            xlite_model.re_down_deq_scale = get_layer_weights(layers, f"{re_prefix}.w2_weight_scale", **re_kwargs)
            xlite_config.quant_attn_weight_nz = bool(xlite_model.attn_wkv) and self.is_tensor_nz(
                xlite_model.attn_wkv[0]
            )
            xlite_config.quant_attn_weight_transpose = True

    # ----- RoPE: per-layer complex64 with compress/non-compress YaRN distinction -----

    @staticmethod
    def _yarn_freqs_cis(
        dim: int,
        end: int,
        original_seq_len: int,
        base: float,
        factor: float,
        beta_fast: float,
        beta_slow: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Precompute YaRN-scaled complex64 freqs_cis for RoPE."""
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if end > original_seq_len and original_seq_len > 0:

            def find_correction_dim(num_rotations):
                return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(find_correction_dim(beta_slow)), dim // 2 - 1)
            if low == high:
                high += 0.001
            ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
            smooth = 1 - ramp
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        t = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        return torch.polar(torch.ones_like(freqs), freqs).to(device=device, dtype=torch.complex64)

    def _precompute_rope_cache(self) -> list[torch.Tensor]:
        """Per-layer complex64 freqs_cis list.

        Compress layers (ratio>1) use ``compress_rope_theta`` with YaRN enabled; non-compress
        layers use ``rope_theta`` with YaRN disabled.
        """
        cfg = self.xlite_config
        dim = cfg.rope_head_dim
        max_seq_len = cfg.max_seq_len
        compress_ratios = list(cfg.compress_ratios)
        freqs: list[torch.Tensor] = []
        for ratio in compress_ratios:
            if ratio > 1:
                base = cfg.compress_rope_theta
                osl = cfg.original_seq_len
            else:
                base = cfg.rope_theta
                osl = 0
            freqs.append(
                self._yarn_freqs_cis(
                    dim, max_seq_len, osl, base, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow, self.device
                )
            )
        return freqs

    # ----- KV cache: vllm per-plane pools -> xlite CXA tuple -----

    @staticmethod
    def _unwrap_cache(kv_cache: Any) -> Any:
        """Strip single-element list/tuple wrappers."""
        while isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 1:
            kv_cache = kv_cache[0]
        return kv_cache

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        """Build the per-layer CXA tuple
        ``(indexer_state, indexer_k, compress_kv, state, swa_kv[, indexer_k_scale])``.
        """
        layers, _ = self._get_layers_and_model_prefix()
        compress_ratios = list(self.xlite_config.compress_ratios)
        n_layers = self.xlite_config.n_layers
        out: list[tuple[torch.Tensor, ...]] = []
        for i, layer in enumerate(layers[:n_layers]):
            ratio = compress_ratios[i] if i < len(compress_ratios) else 0
            swa = self._unwrap_cache(layer.self_attn.dsa_attn.swa_cache_layer.kv_cache)
            if ratio > 1:
                compress = self._unwrap_cache(layer.self_attn.dsa_attn.dsa_attn.kv_cache)
                state = self._unwrap_cache(layer.self_attn.compressor.state_cache.kv_cache)
            else:
                compress = torch.empty(0, dtype=torch.bfloat16, device=self.device)
                state = torch.empty(0, dtype=torch.bfloat16, device=self.device)
            indexer_scale: torch.Tensor | None = None
            if ratio == 4:
                indexer_state = self._unwrap_cache(layer.self_attn.indexer.compressor.state_cache.kv_cache)
                indexer_k, indexer_scale = self._unwrap_cache(layer.self_attn.indexer.k_cache.kv_cache)
            else:
                indexer_state = torch.empty(0, dtype=torch.bfloat16, device=self.device)
                indexer_k = torch.empty(0, dtype=torch.bfloat16, device=self.device)
            if indexer_scale is not None:
                out.append((indexer_state, indexer_k, compress, state, swa, indexer_scale))
            else:
                out.append((indexer_state, indexer_k, compress, state, swa))
        return out

    # ----- AttnMetaV2: translate vllm DSA metadata -> xlite device-tensor metadata -----

    def _build_attn_meta_v2(self, attn_metadata: dict, positions: torch.Tensor, num_tokens: int) -> AttnMetaV2:
        """Build an :class:`AttnMetaV2` from vllm's flat DSA metadata dict.

        Sources the 5 planes (compress/swa/state/indexer/indexer_state) from the first
        ``compress_ratio==4`` layer; its row doubles as the flat default. Slot mapping per plane:
        * idx0 indexer_state, idx3 state: per-token absolute position (write path).
        * idx1 indexer_k: per-block physical slot from the indexer block table; its length
          sets nTotalBlocks.
        * idx2 compress_kv: per-block physical slot from this row's compress block table —
          the compressor write path; numel must equal the derived positions count.
        * idx4 swa_kv: vllm's per-token slot mapping.
        """
        compress_ratios = list(self.xlite_config.compress_ratios)
        c4 = next((i for i, r in enumerate(compress_ratios) if r == 4), None)
        if c4 is None:
            raise RuntimeError("xlite V4: no compress_ratio==4 layer to source CXA metadata from")
        prefix = f"model.layers.{c4}.self_attn"
        plane_keys = {
            "compress": f"{prefix}.attn",
            "swa": f"{prefix}.swa_cache",
            "state": f"{prefix}.compressor.state_cache",
            "indexer": f"{prefix}.indexer.k_cache",
            "indexer_state": f"{prefix}.indexer.compressor.state_cache",
        }
        planes = {}
        for name, key in plane_keys.items():
            if key not in attn_metadata:
                raise RuntimeError(f"xlite V4: DSA metadata key '{key}' not found in forward context")
            planes[name] = attn_metadata[key]  # AscendDSAMetadata

        swa_meta = planes["swa"]
        swa_req = swa_meta.req_metadata
        num_actual_tokens = int(swa_meta.num_actual_tokens)
        seq_lens = torch.as_tensor(swa_req.seq_lens, device="cpu")
        qsl = torch.as_tensor(swa_req.query_start_loc, device="cpu")  # [num_reqs+1], 0-prefixed
        num_reqs = max(1, qsl.shape[0] - 1)
        # query_start_loc is cumulative ([0, q0, q0+q1, ...]); diff yields per-req query lens.
        query_lens = torch.diff(qsl)[:num_reqs]
        cached_lens = torch.clamp(seq_lens[:num_reqs] - query_lens, min=0)

        meta = AttnMetaV2()
        meta.lens = query_lens.to(self.device, dtype=torch.int32)
        meta.cached_lens = cached_lens.to(self.device, dtype=torch.int32)
        meta.lens_cpu = query_lens.tolist()
        meta.cached_lens_cpu = cached_lens.tolist()
        # xlite query_start_loc is [num_reqs] (per-req query start, 0-prefixed) — drop the tail.
        meta.query_start_loc = qsl[:num_reqs].to(self.device, dtype=torch.int32)
        positions_flat = positions.reshape(-1)
        meta.positions = positions_flat[:num_actual_tokens].to(self.device, dtype=torch.int64)

        # slot_mapping per plane: idx0/idx3 (indexer_state, state) use per-token abs-pos; the
        # write path derives each token's state slot from its absolute position. idx1/idx2
        # (indexer_k, compress_kv) are per-block (computed per row below); idx4 (swa_kv) is
        # vllm's per-token slot mapping.
        positions = positions_flat[:num_actual_tokens].to(self.device, dtype=torch.int32)
        sm_fallback = torch.zeros(num_actual_tokens, dtype=torch.int32, device=self.device)
        # Flatten vllm's swa slot mapping: non-A5 hardware gives 2D [block_idx, block_offset]
        # pairs; convert to the 1D physical slot (block_idx*bs + offset). Padding pairs
        # (block_idx < 0) keep -1 so writes skip them. Absent slot mapping falls back to zeros.
        _swa_bs_default = self._cxa_block_sizes[4]

        def _flatten_swa_sm(rm):
            sm = rm.slot_mapping
            if sm is None:
                return sm_fallback
            if sm.ndim == 2:
                swa_bs = int(getattr(rm, "storage_block_size", 0) or _swa_bs_default)
                _bidx, _boff = sm[:, 0], sm[:, 1]
                sm = torch.where(_bidx >= 0, _bidx * swa_bs + _boff, _bidx)
            return sm.reshape(-1)[:num_actual_tokens]

        # Per-layer block_tables/slot_mapping. vllm shares one template row per distinct
        # ratio>0 (sourced from its first layer); ratio==0 layers each own their row (independent
        # swa pools, trailing MTP phantom layer stays all-empty). Rows are references, so
        # expanding to per-layer adds no allocation. The flat default (c4 layer's row) serves
        # the non-per-layer path.
        _suf = {
            "indexer_state": ".indexer.compressor.state_cache",
            "indexer": ".indexer.k_cache",
            "compress": ".attn",
            "state": ".compressor.state_cache",
            "swa": ".swa_cache",
        }
        # plane order matches the CXA cache index: INDEXER_STATE=0, INDEXER_K=1,
        # COMPRESS_KV=2, STATE=3, SWA_KV=4.
        _order = ["indexer_state", "indexer", "compress", "state", "swa"]
        _empty = torch.zeros(0, dtype=torch.int32, device=self.device)

        _first_of_ratio = {}
        for i, r in enumerate(compress_ratios):
            _first_of_ratio.setdefault(r, i)

        def _row(i):
            """Layer i's 5-plane (block_table, slot_mapping) row. Planes absent from this ratio
            (indexer planes on ratio==128, everything but swa on ratio==0) stay _empty."""
            bt_row, sm_row = [], []
            for pn in _order:
                entry = attn_metadata.get(f"model.layers.{i}.self_attn{_suf[pn]}")
                if entry is None or entry.req_metadata is None:
                    bt_row.append(_empty)
                    sm_row.append(_empty)
                    continue
                rm = entry.req_metadata
                bt = rm.block_table
                bt_row.append(bt if torch.is_tensor(bt) and bt.numel() else _empty)
                if pn in ("indexer_state", "state"):
                    sm_row.append(positions)
                elif pn == "compress":
                    # Per-block write slots of THIS row's ratio — numel must match the
                    # derived positions count (nTotalBlocks).
                    sm_row.append(
                        self._compress_block_slot_mapping(
                            bt_row[-1], compress_ratios[i], self._cxa_block_sizes[2], cached_lens, query_lens, num_reqs
                        )
                    )
                elif pn == "swa":
                    sm_row.append(_flatten_swa_sm(rm))
                elif pn == "indexer":
                    sm_row.append(
                        self._compress_block_slot_mapping(
                            bt_row[-1], 4, self._cxa_block_sizes[1], cached_lens, query_lens, num_reqs
                        )
                        if compress_ratios[i] == 4
                        else _empty
                    )
            return bt_row, sm_row

        # ratio>0 peers share one template row per ratio; ratio==0 layers carry their own row
        # (independent pools, trailing MTP phantom all-empty).
        _r0_rows = {i: _row(i) for i, r in enumerate(compress_ratios) if r == 0}
        _tpl = {r: _row(i) for r, i in _first_of_ratio.items() if r > 0}
        _expanded = [_r0_rows[i] if r == 0 else _tpl[r] for i, r in enumerate(compress_ratios)]
        # Flat default = the c4 layer's row (c4 is the first ratio==4 layer, so
        # the guard at the top of this function ensures _tpl[4] exists).
        meta.block_tables, meta.slot_mapping = _tpl[4]
        meta.block_tables_per_layer = [bt for bt, _ in _expanded]
        meta.slot_mapping_per_layer = [sm for _, sm in _expanded]

        return meta

    def _compress_block_slot_mapping(
        self,
        block_table: torch.Tensor,
        ratio: int,
        block_size: int,
        cached_lens: torch.Tensor,
        query_lens: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Compute the per-block slot mapping of a compressed cache (compress_kv or indexer_k)
        from that cache's block table.

        For each request ``b``, the new compressed blocks are ``j = 0..n_blocks_b-1`` where
        ``n_blocks_b = ((cl_b+q_b)//r) - (cl_b//r)``. Each block's physical slot is
        ``block_table[b][(cl_b//r + j)//bs]*bs + (cl_b//r + j)%bs`` (bs = the cache's block
        size). For compress_kv the total length must equal the derived positions count.
        """
        bt_cpu = torch.as_tensor(block_table, device="cpu", dtype=torch.int32)
        slots: list[torch.Tensor] = []
        for b in range(num_reqs):
            cl = int(cached_lens[b].item())
            q = int(query_lens[b].item())
            n_blocks = ((cl + q) // ratio) - (cl // ratio)
            if n_blocks <= 0:
                continue
            logical = cl // ratio + torch.arange(n_blocks, dtype=torch.int32)  # cl/r + j
            blk_idx = logical // block_size
            offset = logical % block_size
            row = bt_cpu[b]
            phys = row[blk_idx] * block_size + offset
            slots.append(phys)
        if not slots:
            return torch.zeros(0, dtype=torch.int32, device=self.device)
        return torch.cat(slots).to(self.device, dtype=torch.int32)


class MiniMaxM2XliteModel(StandardXliteModel):
    """xlite adapter for MiniMax M2 architectures."""

    _supported_architectures = ["MiniMaxM2ForCausalLM"]
    _decoder_layer_mlp_module = "block_sparse_moe"

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.rope_head_dim = hf_config.rotary_dim
        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_local_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.intermediate_size
        xlite_config.norm_topk_prob = True
        xlite_config.qk_norm_full = True
        xlite_config.scoring_func = ScoringFuncSigmoid


def get_adapter_xlite_model(npu_runner: XliteModelRunner, vllm_config: VllmConfig) -> XliteModelBase:
    """Look up and initialize the appropriate xlite model adapter based on the architecture specified in vLLM config and
    the runnable model.

    Args:
        npu_runner (XliteModelRunner): The NPU model runner instance.
        vllm_config (VllmConfig): Runtime configuration for model execution.

    Raises:
        ValueError: If the model architecture is not supported by xlite.

    Returns:
        XliteModelBase: An initialized xlite model adapter ready for inference.
    """
    architecture = vllm_config.model_config.architectures[0]
    if not (strategy_class := _architecture_strategy_map.get(architecture)):
        raise ValueError(f"{architecture} not supported!")
    return strategy_class(npu_runner, vllm_config)


class XliteWrapper:
    """A graph-based wrapper that dispatches between xlite and runnable paths.

    Since v0.28.0, batches are routed by token count (max across DP ranks) instead of the batch attention state:
    batches within the preallocated budget (:attr:`max_tokens`, sized by ``full_mode``) run on xlite, others fall
    back to the native runnable.
    """

    def __init__(self, npu_runner: XliteModelRunner, vllm_config: VllmConfig, device: torch.device) -> None:
        """Initialize xlite runtime, model tensors, and hidden-state workspace.

        Args:
            npu_runner (XliteModelRunner): The native NPU model runner.
            vllm_config (VllmConfig): Runtime configuration for execution.
            device (torch.device): The device to initialize the xlite model on.

        Raises:
            ValueError: If xlite runtime tensor-pool initialization fails.
        """
        self.npu_runner: XliteModelRunner = npu_runner
        self.npu_runnable: nn.Module = npu_runner.fallback_model
        self.npu_update_stream: torch.npu.Stream | None = getattr(npu_runner, "update_stream", None)
        self.device = device
        # `full_mode` only sizes the token budget (see the comment in `StandardXliteModel._build_config`).
        self.full_mode: bool = get_ascend_config().xlite_graph_config.full_mode
        self.data_parallel_size = vllm_config.parallel_config.data_parallel_size
        self.speculative: bool = vllm_config.speculative_config is not None

        set_dummy_tensor(torch.empty(1, device=device, dtype=vllm_config.model_config.dtype))
        self.adapter_xlite_model = get_adapter_xlite_model(npu_runner, vllm_config)  # python adapter instance
        self.xlite_model = self.adapter_xlite_model.xlite_model  # xlite C++ model instance
        self.freq_cis = self.adapter_xlite_model.rope_cossin_cache
        xlite_config = self.adapter_xlite_model.xlite_config
        self.xlite_rt = Runtime(
            devid=device.index,
            size=0,
            rank=torch.distributed.get_rank(),
            tp_size=xlite_config.def_tp_size,
            dp_size=xlite_config.def_dp_size,
            moe_tp_size=xlite_config.moe_tp_size,
            moe_ep_size=xlite_config.moe_ep_size,
        )
        # The xlite token budget: batches with more tokens (max across DP ranks) are routed to the native runnable.
        self.max_tokens = xlite_config.max_m
        self.hidden_states = torch.empty(
            self.max_tokens, xlite_config.hidden_size, device=self.device, dtype=self.adapter_xlite_model.dtype
        )

        rt_pool_size = self.xlite_model.get_tensor_pool_size()
        if torch.distributed.get_rank() == 0:
            logger.info("xlite runtime pool size: %s MB", rt_pool_size)
        if self.xlite_rt.init_tensor_pool(rt_pool_size) != 0:
            raise ValueError(f"xlite wrapper init failed! runtime pool size: {rt_pool_size} MB")

    def __del__(self) -> None:
        self.npu_runner = None  # type: ignore[assignment]
        self.npu_runnable = None  # type: ignore[assignment]
        self.adapter_xlite_model.npu_runner = None  # type: ignore[assignment]
        self.adapter_xlite_model.xlite_model = None  # type: ignore[assignment]

    def __getattr__(self, key: str) -> Any:
        """Proxy unknown attributes to the wrapped runnable model.

        Args:
            key (str): The attribute name requested by the caller.

        Raises:
            AttributeError: If neither wrapper nor runnable has the attribute.

        Returns:
            Any: Attribute value resolved from the runnable.
        """
        try:
            return getattr(self.npu_runnable, key)
        except Exception as e:  # runnable may raise various exceptions
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {key}") from e

    def register_kv_caches(self, kv_caches: list[torch.Tensor]) -> None:
        """Register KV cache references used by xlite runtime.

        Args:
            kv_caches (list[torch.Tensor]): Runtime KV cache handles or tensors.
        """
        self.kv_caches = self.adapter_xlite_model.extract_kv_cache(kv_caches)

    def __call__(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> XliteForwardResult:
        """Run one forward step through xlite graph or fallback runnable path.

        Args:
            input_ids (torch.Tensor | None): Token IDs for current step.

            positions (torch.Tensor): Position IDs used by attention.
            intermediate_tensors (IntermediateTensors | None): Optional intermediate tensors from pipeline stages.
            inputs_embeds (torch.Tensor | None): Optional external input embeddings (e.g. multimodal/deepstack
                scenarios).
            **model_kwargs (Any): Additional keyword arguments for the runnable.

        Returns:
            XliteForwardResult: Forward outputs from xlite graph or the original runnable implementation.
        """
        forward_context = get_forward_context()
        attn_metadata: Any = forward_context.attn_metadata
        attn_metadata = attn_metadata[0] if isinstance(attn_metadata, list) else attn_metadata
        attn_metadata = attn_metadata.get("model.layers.0.self_attn.attn", next(iter(attn_metadata.values()), None))
        # Since v0.28.0, batches are routed by token count (max across DP ranks) instead of the batch attention
        # state: batches within the preallocated budget run on xlite, others fall back to the native acl runnable.
        num_tokens: int
        if (num_tokens := _EXTRA_CTX.max_tokens_across_dp) is None:
            if (dp_metadata := forward_context.dp_metadata) is not None:
                num_tokens = dp_metadata.num_tokens_across_dp_cpu.max().item()
            else:
                num_tokens = attn_metadata.num_actual_tokens
        if num_tokens > self.max_tokens:
            if self.full_mode:
                logger.warning_once(
                    "xlite: token number exceeded expected limit (%d >%d), consider opening an issue at %s",
                    num_tokens,
                    self.max_tokens,
                    "https://atomgit.com/openeuler/GVirt",
                )
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        # sync the device streams to avoid stale D2H copies in metadata preparation
        curr_stream = torch.npu.current_stream()
        if self.npu_update_stream is not None and self.npu_update_stream != curr_stream:
            curr_stream.wait_stream(self.npu_update_stream)

        # V4 CXA attention runs on the V2 device-tensor forward path (AttnMetaV2 + forward_v2);
        # the V1 host-path (AttnMeta + forward) below does not apply. `attn_metadata` was
        # clobbered to a single layer entry above, so re-fetch the full DSAMetadataDict here.
        if getattr(self.adapter_xlite_model, "_uses_v2", False):
            dsa_dict = forward_context.attn_metadata
            if isinstance(dsa_dict, list):
                dsa_dict = dsa_dict[0]
            num_actual_tokens = attn_metadata.num_actual_tokens
            attn_meta_v2 = self.adapter_xlite_model._build_attn_meta_v2(dsa_dict, positions, num_tokens)
            h = self.hidden_states[:num_tokens]
            stream = torch.npu.current_stream().npu_stream
            input_ids = cast(torch.Tensor, input_ids)
            self.xlite_model.forward_v2(
                rt=self.xlite_rt,
                input=input_ids[:num_actual_tokens],
                attn_meta=attn_meta_v2,
                kv_cache=self.kv_caches,
                freqs_cis=self.freq_cis,
                output=h,
                curr_stream=stream,
            )
            return h[:num_actual_tokens]

        attn_metadata_router = AttnMetadataRouter(attn_metadata=attn_metadata, device="cpu")
        num_actual_tokens = attn_metadata_router.num_actual_tokens
        seq_lens = attn_metadata_router.seq_lens
        cum_query_lens = attn_metadata_router.cu_query_lens[-seq_lens.size(0) :]
        query_lens = torch.diff(cum_query_lens, prepend=seq_lens.new_zeros(1))
        if self.speculative:
            # host-side `seq_lens` may be optimistic, rejected draft tokens still counted as cached,
            # cf.gpu_model_runner._prepare_inputs; each request's first position is its true cached length, so cap by it
            _pos = torch.atleast_2d(positions)[0, :num_actual_tokens].to(device=seq_lens.device)
            req_starts = torch.cat([cum_query_lens.new_zeros(1), cum_query_lens[:-1]]).clamp(max=_pos.numel() - 1)
            cached_lens = _pos[req_starts].to(dtype=seq_lens.dtype)
        else:
            cached_lens = torch.clamp(seq_lens - query_lens, min=0)

        block_tables = attn_metadata_router.block_tables
        num_reqs = max(1, (block_tables[:, 0] != 0).sum().item())  # number of non-empty requests

        xlite_attn_metadata = AttnMeta()
        xlite_attn_metadata.lens = query_lens[:num_reqs].tolist()
        xlite_attn_metadata.cached_lens = cached_lens[:num_reqs].tolist()
        xlite_attn_metadata.block_tables_cpu = block_tables[:num_reqs].tolist()
        pos: torch.Tensor = torch.atleast_2d(positions)[:, :num_actual_tokens].contiguous()
        xlite_attn_metadata.positions = pos.reshape(-1).reshape(pos.shape)  # rematerialize with correct stride

        # under DP, `num_tokens` is the max number of tokens across all DP ranks for data alignment
        h = self.hidden_states[:num_tokens]
        input_ids = cast(torch.Tensor, input_ids)
        if inputs_embeds is None or inputs_embeds.size(0) == 0:
            self.xlite_model.forward(
                rt=self.xlite_rt,
                input=input_ids[:num_actual_tokens],
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=self.freq_cis,
                output=h,
                curr_stream=curr_stream.npu_stream,
            )
        else:
            inputs_embeds = inputs_embeds[:num_actual_tokens]
            deepstack_input_embeds = getattr(self.npu_runnable, "deepstack_input_embeds", [])
            xlite_deepstack_input_embeds = [
                deepstack_input[: inputs_embeds.size(0)] for deepstack_input in deepstack_input_embeds
            ]
            self.xlite_model.forward_with_inputs_embeds(
                rt=self.xlite_rt,
                input=inputs_embeds,
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=self.freq_cis,
                output=h,
                curr_stream=curr_stream.npu_stream,
                deepstack_input=xlite_deepstack_input_embeds,
            )
            if deepstack_input_embeds and callable(
                _clear_stack := getattr(self.npu_runnable, "_clear_deepstack_input_embeds", None)
            ):
                _clear_stack(inputs_embeds.size(0))
        return h[:num_actual_tokens]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | torch.nn.Module:
        # TODO: consider implementing a custom logits computation path for xlite to avoid falling back to the native
        return self.npu_runnable.compute_logits(hidden_states)  # type: ignore[attr-defined]
