# Copyright (c) 2020, Soohwan Kim. All rights reserved.
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
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from builder.models.src.transformer.attention import MultiHeadAttention
from builder.models.src.transformer.module import PositionalEncoding, PositionwiseFeedForward, LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
            block_mask: list = None,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.attention_prenorm = LayerNorm(d_model)
        self.feed_forward_prenorm = LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, block_mask)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(self, inputs: Tensor, self_attn_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, attn


class TransformerEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, d_input, n_layers, n_head,
                 d_model, d_ff, dropout=0.1, pe_maxlen=5000, use_pe=True, block_mask=None):
        super(TransformerEncoder, self).__init__()
        # parameters
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen
        self.use_pe = use_pe
        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=n_head,
                d_ff=d_ff,
                dropout_p=dropout,
                block_mask=block_mask,
            ) for _ in range(n_layers)
        ])

    def forward(self, padded_input, input_lengths=None, return_attns=False):
        enc_slf_attn_list = []

        # Prepare masks
        # non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        # length = padded_input.size(1)
        # slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)

        # Forward
        if self.use_pe:
            enc_output = self.dropout(
                self.layer_norm_in(self.linear_in(padded_input)) +
                self.positional_encoding(padded_input.size(1)))
        else:
            enc_output = self.dropout(
                self.layer_norm_in(self.linear_in(padded_input)))
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output
