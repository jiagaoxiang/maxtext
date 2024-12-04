"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
from typing import Any, Optional, Tuple, Union, Callable

import jax
from jax.sharding import Mesh
from jax import lax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from flax import linen as nn
from flax.linen.attention import dot_product_attention_weights

from layers import attentions
from layers import initializers
from layers import linears
from layers import models
from layers import quantizations
import math

AttentionOp = attentions.AttentionOp

import common_types

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV

DenseGeneral = linears.DenseGeneral
NdInitializer = initializers.NdInitializer
Initializer = initializers.Initializer
nd_dense_init = initializers.nd_dense_init
Quant = quantizations.AqtQuantization
KVQuant = quantizations.KVQuant


# -----------------------------------------
# The Normalization Layer specific for GPT3
# -----------------------------------------


class LayerNorm(nn.Module):
  """Llama3.2 vision model Layer normalization operating on the last axis of the input data."""

  epsilon: float = 1e-5
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  kernel_axes: Tuple[str, ...] = ()
  scale_init: Initializer = nn.initializers.ones
  use_bias: bool = True
  reductions_in_fp32: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    if self.reductions_in_fp32:
      x = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=[-1], keepdims=True)
    normed_inputs = (x - mean) * lax.rsqrt(var + self.epsilon)
    if self.reductions_in_fp32:
      normed_inputs = normed_inputs.astype(self.dtype)

    features = x.shape[-1]
    scale = self.param(
        "scale", nn.with_logical_partitioning(self.scale_init, self.kernel_axes), (features,), self.weight_dtype
    )

    scale = jnp.asarray(scale, self.dtype)
    output = normed_inputs * scale

    if self.use_bias:
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(initializers.default_bias_init, self.kernel_axes),
          (features,),
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output


# -----------------------------------------
# The Attention Layer specific for llama3.2 vision model
# -----------------------------------------


class VisionMultiHeadAttention(nn.Module):
  """Multi-head attention in llama3.2 vision model.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    head_dim: dimension of each head.
    dtype: the dtype of the computation.
    dropout_rate: dropout rate
    kernel_init: initializer for the kernel of the Dense layers.
    float32_logits: bool, if True then cast logits to float32 before softmax to avoid
      numerical issues with bfloat16.
    quant: Quant, stores quantization config, defaults to None implying no quantization.
    use_bias: whether to add bias in linear transformation.
  """

  config: Config
  num_heads: int = 16
  head_dim: int = 80
  attention_kernel: str
  dtype: DType = jnp.float32
  weight_dtype: DType = jnp.float32
  dropout_rate: float = 0.0
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  float32_logits: bool = True  # cast logits in float32 for stability.
  quant: Optional[Quant] = None

  query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  key_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  value_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)

  def projection(self, inputs: Array, proj_name: str) -> Array:
    """individual projection for one of q, k and v."""
    proj = DenseGeneral(
        features=self.num_heads * self.head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant,
        use_bias=False,
        matmul_precision=self.config.matmul_precision,
    )(inputs)
    return proj

  def out_projection(self, output_dim: int, out: Array) -> Array:
    """output projection"""
    out_proj = DenseGeneral(
        features=output_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("heads", "kv", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="out",
        quant=self.quant,
        use_bias=False,
        matmul_precision=self.config.matmul_precision,
    )(out)
    return out_proj
  
  def cudnn_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      attention_mask: Array = None,
      deterministic: bool = True,
  ) -> Array:
    """CUDNN Flash Attention with Transformer Engine.
    1. Stable API, supports GQA
    2. Supports head_dim till 128; head_dim=256 support will be added soon
    """
    # These imports are only meant to work in a GPU build.
    from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

    _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

    dpa_layer = DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=self.num_query_heads,
        num_gqa_groups=self.num_kv_heads,
        attn_mask_type="NO_MASK",  # 'no_mask', 'padding', 'causal', or 'padding_causal'
        attn_bias_type="POST_SCALE_BIAS",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
        attention_dropout=self.dropout_rate,
        dropout_rng_name="aqt",
        dtype=self.dtype,
        float32_logits=self.float32_logits,
        qkv_layout="BSHD_BSHD_BSHD",  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
        scale_factor=1.0 / math.sqrt(head_dim),
        transpose_batch_sequence=False,
    )
    return dpa_layer(query, key, value, bias=attention_mask, deterministic=deterministic)
  
  def _split_heads(self, hidden_states):
    return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

  def _merge_heads(self, hidden_states):
    return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      attention_mask: Array = None,
      *,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
      deterministic: bool = True,
  ):

    query = self.projection(inputs_q, proj_name="query")
    key = self.projection(inputs_q, proj_name="key")
    value = self.projection(inputs_q, proj_name="value")

    query = self._split_heads(query) #[1, 4128, 16, 80]
    key = self._split_heads(key) #[1, 4128, 16, 80]
    value = self._split_heads(value)

    if attention_mask is not None:# Ensure the attention mask matches the key sequence length
      causal_mask = attention_mask[:, :, :, : key.shape[-3]] # (batch_size, 1, q_seq_len, kv_seq_len)

    # annotate with sharding constraint.
    query = nn.with_logical_constraint(query, self.query_axis_names)
    key = nn.with_logical_constraint(key, self.key_axis_names)
    value = nn.with_logical_constraint(value, self.value_axis_names)
    
    if self.use_flash_attention:
      out = self.cudnn_flash_attention(
        query,
        key,
        value,
        attention_mask=causal_mask,
        deterministic=deterministic,
      )
    else:
      attn_weights = dot_product_attention_weights(
        query,
        key,
        bias=causal_mask,
        deterministic=deterministic,
        dtype=self.dtype,
        )
      out = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)

    out = self._merge_heads(out)
    out = nn.with_logical_constraint(out, self.out_axis_names)

    # apply output projection,  output dim is set to the input dim.
    out = self.out_projection(inputs_q.shape[-1], out)
    # out = checkpoint_name(out, "out_proj")
    return out

class VisionMlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: computation data type for the dense layer.
    weight_dtype: weight data type for the dense layer.
    use_bias: whether to add bias in all feedforward layers.
    use_pre_norm: whether to add pre layer norm in mlp layers.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  intermediate_dim: int = 5120
  activations: Union[str, Callable[..., Any]] = "gelu"
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    x = DenseGeneral(
        self.intermediate_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "mlp"),
        name="fc1",
        quant=self.quant,
        use_bias=True,
        matmul_precision=self.config.matmul_precision,
    )(inputs)
    if cfg.activations_in_float32:
      x = x.astype(jnp.float32)
    x = getattr(nn, self.activations)(x)

    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_mlp"))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        name="fc2",
        quant=self.quant,
        use_bias=True,
        matmul_precision=self.config.matmul_precision,
    )(x)

    return output

class VisionEncoderLayer(nn.Module):
  config: Config
  is_gated: bool = False

  def setup(self):
    self.self_attn = VisionMultiHeadAttention(
      config=self.config,
      num_heads=self.config.num_heads,
      head_dim=self.config.head_dim,
      attention_kernel=self.config.attention,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      dropout_rate=self.config.dropout_rate,
      quant=self.config.quant,
      float32_logits=self.config.float32_logits,
    )
    self.input_layernorm = LayerNorm(
      epsilon=self.config.normalization_layer_epsilon, 
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      kernel_axes=("norm",),
      )
    self.mlp = VisionMlpBlock(
      config=self.config,
      intermediate_dim=self.config.mlp_dim,
      activations=self.config.mlp_activations,
      intermediate_dropout_rate=self.config.dropout_rate,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,

    )
    self.post_attention_layernorm = LayerNorm(
      epsilon=self.config.normalization_layer_epsilon, 
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      kernel_axes=("norm",),
      )

    if self.is_gated:
      self.gate_attn = self.param('gate_attn', nn.initializers.constant(math.pi / 4), (1,), self.config.weight_dtype)
      self.gate_ffn = self.param('gate_ffn', nn.initializers.constant(math.pi / 4), (1,), self.config.weight_dtype)

  def __call__(
      self,
      hidden_states,
      attention_mask,
      deterministic: bool = True
  ):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        inputs_q=hidden_states,
        attention_mask=attention_mask,
        deterministic=deterministic,
    )
    # Apply residual connection with optional gating for attention
    if self.is_gated:
        hidden_states = jnp.tanh(self.gate_attn) * hidden_states
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if self.is_gated:
      hidden_states = jnp.tanh(self.gate_ffn) * hidden_states
    hidden_states = residual + hidden_states

    return hidden_states
  

class VisionEncoder(nn.Module):
  config: Config
  num_layers: 32
  is_gated: bool = False

  def setup(self):
    self.layers = [
        VisionEncoderLayer(self.config, name="layers."+str(i), is_gated=self.is_gated)
        for i in range(self.num_layers)
    ]

  def __call__(
      self,
      hidden_states,
      attention_mask=None,
      deterministic: bool = True,
      output_hidden_states: bool = False,
  ):
    all_hidden_states = () if output_hidden_states else None

    for layer in self.layers:
      if output_hidden_states:
        all_hidden_states += (hidden_states,)

      hidden_states = layer(hidden_states, attention_mask, deterministic=deterministic)

    if output_hidden_states:
      all_hidden_states += (hidden_states,)

    return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
  

class VisionPrecomputedPositionEmbedding(nn.Module):
    """
    VisionPrecomputedPositionEmbedding is a neural network module that computes position embeddings for input hidden states.
    It uses precomputed position embeddings and tile position embeddings based on aspect ratio IDs.

    Args:
        config: Configuration object containing model hyperparameters.
    """
    config: Config

    @nn.compact
    def __call__(self, hidden_state: jnp.ndarray, aspect_ratio_ids: jnp.ndarray) -> jnp.ndarray:
        max_num_tiles = self.config.max_num_tiles  # 4
        max_aspect_ratio_id = self.config.max_aspect_ratio_id  # 8
        num_patches = (self.config.image_size // self.config.patch_size) ** 2 + 1  # 1025
        hidden_size = self.config.hidden_size  # 1280
        scale = hidden_size ** -0.5

        # Learnable gate parameter
        gate = self.param("gate", jax.nn.initializers.zeros, (1,), self.config.weight_dtype)

        # Position embedding
        position_embedding = self.param(
            "embedding", 
            lambda key, shape: scale * jax.random.normal(key, shape), 
            (num_patches, hidden_size), 
            self.config.weight_dtype
        )

        # Tile position embedding
        tile_embedding = nn.Embed(
            name="tile_embedding",
            num_embeddings=max_aspect_ratio_id + 1, 
            features=max_num_tiles * num_patches * hidden_size,
            dtype=self.config.dtype,
            param_dtype=self.config.weight_dtype,
        )
        # Apply gated position embedding
        gated_position_embedding = (1 - jnp.tanh(gate)) * position_embedding
        hidden_state = hidden_state + gated_position_embedding.reshape(1, 1, num_patches, hidden_size)

        # Precomputed tile position embeddings
        tile_position_embedding = tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, max_num_tiles, num_patches, hidden_size
        )
        gated_tile_position_embedding = jnp.tanh(gate) * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


class VisionPrecomputedAspectRatioEmbedding(nn.Module):
    config: Config
    is_gated: bool = True

    def setup(self):
        self.max_num_tiles = self.config.max_num_tiles
        self.hidden_size = self.config.hidden_size
        self.max_aspect_ratio_id = self.config.max_aspect_ratio_id

        self.embedding = nn.Embed(
            num_embeddings=self.max_aspect_ratio_id + 1,
            features=self.max_num_tiles * self.hidden_size,
            dtype=self.config.dtype,
            param_dtype=self.config.weight_dtype,
        )
        if self.is_gated:
            self.gate = self.param("gate", jax.nn.initializers.zeros, (1,), self.config.weight_dtype)

    def __call__(self, hidden_state, aspect_ratio_ids):
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * jnp.tanh(self.gate)

        hidden_state = hidden_state + embeddings
        return hidden_state
    

def _prepare_aspect_ratio_attention_mask(

    aspect_ratio_mask: jnp.ndarray,
    num_patches: int,
    target_length: int,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Prepares an attention mask based on the aspect ratio mask for a given number of patches and target length.
    The return is actually an attention bias that should be added to the attention weights.

    Args:
      aspect_ratio_mask (jnp.ndarray): The input aspect ratio mask of shape (batch_size, max_num_tiles).
      num_patches (int): The number of patches in the input.
      target_length (int): The target length to which the aspect ratio mask should be expanded.
      dtype (jnp.dtype, optional): The data type of the attention mask. Defaults to jnp.float32.

    Returns:
      jnp.ndarray: The prepared attention mask of shape (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length).
    """
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = jnp.expand_dims(aspect_ratio_mask, (2, 3)).astype(dtype)  # (batch_size, max_num_tiles, 1, 1)
    attention_mask = jnp.tile(attention_mask, (1, 1, target_length, 1))  # (batch_size, max_num_tiles, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    if pad_patches > 0:
        attention_mask = attention_mask.at[:, :, -pad_patches:].set(0)

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = jnp.matmul(attention_mask, attention_mask.swapaxes(2, 1)) * jnp.finfo(dtype).min
    attention_mask = jnp.expand_dims(attention_mask, axis=1)

    return attention_mask

class VisionTransformer(nn.Module):
  config: Config

  def setup(self):
    self.image_size = self.config.image_size #448
    self.patch_size = self.config.patch_size #14
    self.max_num_tiles = self.config.max_num_tiles #4
    self.hidden_size = self.config.hidden_size #1280
    self.num_channels = self.config.num_channels #3
    self.intermediate_layers_indices = self.config.intermediate_layers_indices #[3, 7, 15, 23, 30]

    self.num_patches = (self.image_size // self.patch_size) ** 2 + 1 #1025
    self.scale = self.config.hidden_size**-0.5
  
    self.patch_embedding = nn.Conv(
        self.hidden_size,
        kernel_size=(self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        padding="VALID",
        use_bias=False,
        dtype=self.config.dtype,
        kernel_init=jax.nn.initializers.normal(),
    )

    self.class_embedding = self.param("class_embedding", jax.nn.initializers.normal(stddev=0.02), (self.config.hidden_size,))
    self.gated_positional_embedding = FlaxMllamaPrecomputedPositionEmbedding(self.config, dtype=self.dtype,)

    self.pre_tile_positional_embedding = FlaxMllamaPrecomputedAspectRatioEmbedding(self.config, is_gated=True, dtype=self.dtype,)
    self.post_tile_positional_embedding = FlaxMllamaPrecomputedAspectRatioEmbedding(self.config, is_gated=True, dtype=self.dtype,)

    self.layernorm_pre = nn.LayerNorm(epsilon=self.config.norm_eps, dtype=self.dtype,)
    self.layernorm_post = nn.LayerNorm(epsilon=self.config.norm_eps, dtype=self.dtype,)
    
    self.transformer = FlaxMllamaVisionEncoder(self.config, self.config.num_hidden_layers, dtype=self.dtype,)
    self.global_transformer = FlaxMllamaVisionEncoder(self.config, self.config.num_global_layers, dtype=self.dtype, is_gated=True)

  def get_input_embeddings(self):
      """
      This function is used to fetch the first embedding layer to activate grads on inputs.
      """
      return self.patch_embedding
  
  def apply_class_embedding(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
      batch_size, _, hidden_size = hidden_state.shape
      class_embedding = jnp.expand_dims(self.class_embedding, axis=(0, 1))
      class_embedding = jnp.tile(class_embedding, (batch_size, 1, 1))
      hidden_state = jnp.concatenate([class_embedding, hidden_state], axis=1)
      return hidden_state
  
  def __call__(
      self,
      pixel_values: jnp.ndarray,
      aspect_ratio_ids: jnp.ndarray,
      aspect_ratio_mask: jnp.ndarray,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
      return_dict: bool = True,
  ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape((batch_size * num_concurrent_media * num_tiles, num_channels, height, width))
        aspect_ratio_ids = aspect_ratio_ids.reshape((batch_size * num_concurrent_media, -1))

        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values.transpose((0, 2, 3, 1)))
        patch_embeds = patch_embeds.transpose((0, 3, 1, 2))
        hidden_state = patch_embeds.reshape(batch_size * num_concurrent_media * num_tiles, self.hidden_size, -1).swapaxes(1, 2)

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles, -1, dim))
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media * num_tiles, num_patches, dim))
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles, num_patches, dim))
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        padding = [(0, 0), (0, 0), (0, num_padding_patches), (0, 0)]
        hidden_state = jnp.pad(hidden_state, padding, mode="constant", constant_values=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape((batch_size * num_concurrent_media, -1))
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, -1, dim))
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        hidden_state = output[0]
        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim))
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim))
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_state = global_output[0]

        # Remove padding from hidden state
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim))
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape((batch_size, num_concurrent_media, num_tiles, num_patches, dim))

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = output[1]
        intermediate_hidden_states = jnp.stack(all_intermediate_hidden_states, axis=-1)
        intermediate_hidden_states = intermediate_hidden_states[..., self.intermediate_layers_indices]

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape((batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1))
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape((batch_size, num_concurrent_media, num_tiles, num_patches, -1))

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = jnp.concatenate([hidden_state, intermediate_hidden_states], axis=-1)

        if output_hidden_states:
            hidden_states = tuple(all_intermediate_hidden_states) + tuple(global_output[1])
        else:
            hidden_states = None

        if output_attentions:
            global_attn = tuple(global_output[2]) if output_hidden_states else tuple(global_output[1])
            attentions = tuple(output[2]) + global_attn
        else:
            attentions = None

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states, attentions] if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )