"""
AlphaGenome Model Architecture Implementation

This module implements the core AlphaGenome model following the paper architecture:
- U-Net style encoder-decoder with skip connections
- Transformer tower for long-range dependencies
- Pairwise interaction blocks for 3D genome modeling
- Multi-task prediction heads
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Optional, Tuple, Any
import numpy as np


class AlphaGenomeModel(hk.Module):
    """
    AlphaGenome unified sequence-to-function model.
    
    Processes 1Mb DNA sequences to predict diverse genomic tracks including:
    - Gene expression (RNA-seq, CAGE, PRO-cap)
    - Chromatin accessibility (ATAC-seq, DNase-seq)
    - Histone modifications and TF binding (ChIP-seq)
    - 3D genome organization (contact maps)
    - Splicing patterns (sites, usage, junctions)
    """
    
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config
        self.sequence_length = config.get('sequence_length', 1048576)  # 1Mb
        self.num_organisms = config.get('num_organisms', 2)  # human, mouse
        
    def __call__(self, 
                 dna_sequence: jnp.ndarray, 
                 organism_id: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Forward pass of AlphaGenome model.
        
        Args:
            dna_sequence: [B, L, 4] one-hot encoded DNA sequence
            organism_id: [B] organism identifier (0=human, 1=mouse)
            
        Returns:
            Dictionary of predictions for different genomic tracks
        """
        # 1. Sequence encoder: 1bp -> 128bp resolution
        trunk, intermediates = self._sequence_encoder(dna_sequence)
        
        # 2. Add organism-specific embeddings to trunk (1536 channels)
        organism_embed = hk.Embed(1536, name='trunk_organism_embed')(organism_id)
        trunk = trunk + organism_embed[:, None, :]
        
        # 3. Transformer tower with pairwise interactions
        trunk, pair_activations = self._transformer_tower(trunk)
        
        # 4. Sequence decoder with U-Net skip connections
        decoded_seq = self._sequence_decoder(trunk, intermediates)
        
        # 5. Generate output embeddings
        embeddings_128bp = self._output_embedder(trunk, organism_id)
        embeddings_1bp = self._output_embedder(decoded_seq, organism_id, embeddings_128bp)
        embeddings_pair = self._output_pair_embedder(pair_activations, organism_id)
        
        # 6. Task-specific prediction heads
        predictions = self._prediction_heads(
            embeddings_1bp, embeddings_128bp, embeddings_pair, organism_id
        )
        
        return predictions
    
    def _sequence_encoder(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Sequence encoder: progressively downsample 1bp -> 128bp resolution.
        7 stages: 1,2,4,8,16,32,64 bp with channels 768,896,1024,1152,1280,1408,1536
        
        Args:
            x: [B, L, 4] input DNA sequence
            
        Returns:
            encoded sequence and intermediates for U-Net skip connections
        """
        intermediates = {}
        
        # Stage 0: DNA embedding (1bp resolution, 768 channels)
        x = hk.Conv1D(768, 15, name='dna_embedder_conv')(x)
        x = x + self._conv_block(x, 768, name='dna_embedder_residual')
        intermediates['bin_size_1'] = x
        x = hk.max_pool(x, window_shape=2, strides=2, padding='SAME')
        
        # Stages 1-6: Downsampling with channel increase by 128 each stage
        for stage in range(1, 7):
            bin_size = 2 ** stage
            channels = 768 + stage * 128  # 896, 1024, 1152, 1280, 1408, 1536
            
            x = self._downres_block(x, channels, name=f'downres_block_{stage}')
            intermediates[f'bin_size_{bin_size}'] = x
            
            # Max pool for all stages except the last
            if stage < 6:
                x = hk.max_pool(x, window_shape=2, strides=2, padding='SAME')
        
        return x, intermediates
    
    def _transformer_tower(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Transformer tower: 9 layers with pairwise interaction blocks.
        
        Args:
            x: [B, S, C] sequence embeddings at 128bp resolution (S=8192, C=1536)
            
        Returns:
            transformed sequence and pairwise activations
        """
        pair_activations = None
        
        for i in range(9):  # 9 transformer blocks
            # Update pairwise representations every other layer (0,2,4,6,8)
            if i % 2 == 0:
                pair_activations = self._pair_update_block(
                    x, pair_activations, name=f'pair_update_{i}'
                )
            
            # Multi-head attention with pairwise bias
            attention_bias = self._attention_bias_block(
                pair_activations, name=f'attention_bias_{i}'
            ) if pair_activations is not None else jnp.zeros((x.shape[0], 8, x.shape[1], x.shape[1]))
            
            attn_out = self._mha_block(x, attention_bias, name=f'mha_{i}')
            x = x + attn_out
            
            # MLP block
            mlp_out = self._mlp_block(x, name=f'mlp_{i}')
            x = x + mlp_out
            
        return x, pair_activations
    
    def _sequence_decoder(self, 
                         x: jnp.ndarray, 
                         intermediates: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Sequence decoder: upsample 128bp -> 1bp with U-Net skip connections.
        
        Args:
            x: [B, S, C] encoded sequence at 128bp resolution
            intermediates: skip connections from encoder
            
        Returns:
            decoded sequence at 1bp resolution
        """
        # 7 upsampling stages (reverse of encoder)
        for bin_size in [64, 32, 16, 8, 4, 2, 1]:
            unet_skip = intermediates[f'bin_size_{bin_size}']
            x = self._upres_block(x, unet_skip, name=f'upres_block_{bin_size}')
            
        return x
    
    def _prediction_heads(self,
                         embeddings_1bp: jnp.ndarray,
                         embeddings_128bp: jnp.ndarray, 
                         embeddings_pair: jnp.ndarray,
                         organism_id: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Task-specific prediction heads for different genomic modalities.
        """
        predictions = {}
        
        # RNA-seq, CAGE, ATAC, DNase, PRO-cap predictions (1bp and 128bp)
        for modality in ['rna_seq', 'cage', 'atac_seq', 'dnase_seq', 'pro_cap']:
            num_tracks = self.config.get(f'{modality}_tracks', 100)
            
            # Both resolutions for these modalities
            predictions[f'{modality}_1bp'] = self._tracks_scaled_predictions(
                embeddings_1bp, num_tracks, f'{modality}_1bp'
            )
            predictions[f'{modality}_128bp'] = self._tracks_scaled_predictions(
                embeddings_128bp, num_tracks, f'{modality}_128bp'
            )
        
        # ChIP-seq predictions (128bp only)
        for chip_type in ['chip_tf', 'chip_histone']:
            num_tracks = self.config.get(f'{chip_type}_tracks', 200)
            predictions[chip_type] = self._tracks_scaled_predictions(
                embeddings_128bp, num_tracks, chip_type
            )
        
        # Contact maps (2048bp resolution from pairwise embeddings)
        num_contact_tracks = self.config.get('contact_tracks', 30)
        predictions['contact_maps'] = self._contact_map_head(
            embeddings_pair, num_contact_tracks
        )
        
        # Splicing predictions (1bp resolution)
        predictions['splice_sites'] = self._splice_sites_head(embeddings_1bp)
        predictions['splice_usage'] = self._splice_usage_head(embeddings_1bp)
        predictions['splice_junctions'] = self._splice_junctions_head(embeddings_1bp)
        
        return predictions
    
    # ========== Core Building Blocks ==========
    
    def _conv_block(self, x: jnp.ndarray, num_channels: int, 
                   width: int = 5, name: str = None) -> jnp.ndarray:
        """Conv block: RMSBatchNorm -> GeLU -> StandardizedConv1D"""
        x = self._rms_batch_norm(x, name=f'{name}_norm')
        x = jax.nn.gelu(x)
        if width == 1:
            x = hk.Linear(num_channels, name=f'{name}_linear')(x)
        else:
            x = hk.Conv1D(num_channels, width, name=f'{name}_conv')(x)
        return x
    
    def _downres_block(self, x: jnp.ndarray, num_channels: int, name: str) -> jnp.ndarray:
        """Downsampling block with channel increase and skip connection."""
        # First conv block increases channels
        out = self._conv_block(x, num_channels, name=f'{name}_conv1')
        
        # Residual connection with zero padding for channel mismatch
        channel_diff = num_channels - x.shape[-1]
        if channel_diff > 0:
            x_padded = jnp.pad(x, [(0, 0), (0, 0), (0, channel_diff)])
        else:
            x_padded = x
        out = out + x_padded
        
        # Second conv block
        out = out + self._conv_block(out, num_channels, name=f'{name}_conv2')
        
        return out
    
    def _rms_batch_norm(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """RMS Batch Normalization as described in paper."""
        return hk.RMSNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)
    
    def _upres_block(self, x: jnp.ndarray, unet_skip: jnp.ndarray, name: str) -> jnp.ndarray:
        """
        Upsampling block exactly as described in paper.
        """
        num_channels = unet_skip.shape[-1]
        
        # Conv block to match channels + residual connection
        out = self._conv_block(x, num_channels, name=f'{name}_conv1')
        out = out + x[:, :, :num_channels]  # Crop if needed
        
        # Upsample by repetition and scale
        out = jnp.repeat(out, 2, axis=1)
        residual_scale = hk.get_parameter(f'{name}_residual_scale', 
                                        shape=(), init=hk.initializers.Constant(0.9))
        out = out * residual_scale
        
        # Process skip connection
        skip_processed = self._conv_block(unet_skip, num_channels, width=1, name=f'{name}_skip')
        out = out + skip_processed
        
        # Final conv block with residual
        final_out = self._conv_block(out, num_channels, name=f'{name}_conv2')
        return out + final_out
    
    # ========== Transformer Components ==========
    
    def _mha_block(self, x: jnp.ndarray, attention_bias: jnp.ndarray, name: str) -> jnp.ndarray:
        """Multi-query attention block exactly as in paper."""
        x_norm = self._rms_batch_norm(x, f'{name}_norm')
        
        # Multi-query attention: 8 query heads, 1 shared key/value head
        q = hk.Linear(8 * 128, with_bias=False, name=f'{name}_q')(x_norm)
        k = hk.Linear(1 * 128, with_bias=False, name=f'{name}_k')(x_norm)
        v = hk.Linear(1 * 192, with_bias=False, name=f'{name}_v')(x_norm)
        
        # Layer norm for each projection
        q = hk.LayerNorm(axis=-1, create_scale=True, create_offset=False, name=f'{name}_q_norm')(q)
        k = hk.LayerNorm(axis=-1, create_scale=True, create_offset=False, name=f'{name}_k_norm')(k)
        v = hk.LayerNorm(axis=-1, create_scale=True, create_offset=False, name=f'{name}_v_norm')(v)
        
        # Reshape for multi-head
        B, S, _ = x.shape
        q = q.reshape(B, S, 8, 128)
        k = k.reshape(B, S, 1, 128) 
        v = v.reshape(B, S, 1, 192)
        
        # Apply RoPE
        q = self._apply_rope(q, max_position=8192)
        k = self._apply_rope(k, max_position=8192)
        
        # Attention computation
        attention_logits = jnp.einsum('bshc,bSkc->bhsS', q, k) / jnp.sqrt(128)
        
        # Add bias and soft-clip logits
        attention_logits = attention_logits + attention_bias
        attention_logits = jnp.tanh(attention_logits / 5.0) * 5.0
        
        # Softmax and attend
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        y = jnp.einsum('bhsS,bSkc->bshc', attention_weights, v)
        
        # Reshape and project back
        y = y.reshape(B, S, -1)
        y = hk.Linear(x.shape[-1], name=f'{name}_proj')(y)
        
        # Dropout and final norm
        y = hk.dropout(hk.next_rng_key(), 0.3, y) if hk.running_init() else y
        return self._rms_batch_norm(y, f'{name}_out_norm')
    
    def _apply_rope(self, x: jnp.ndarray, max_position: int = 8192) -> jnp.ndarray:
        """Apply RoPE with paper's modified frequency calculation."""
        positions = jnp.arange(x.shape[1])
        num_freq = x.shape[-1] // 2
        
        # Paper's modified frequency calculation
        freq = 1.0 / (jnp.arange(num_freq) + jnp.geomspace(1, max_position - num_freq + 1, num_freq))
        theta = jnp.repeat(jnp.outer(positions, freq), 2, axis=-1)
        
        # Apply rotation
        x_rotated = jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)
        return x * jnp.cos(theta) + x_rotated * jnp.sin(theta)
    
    def _mlp_block(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """MLP block with 2x expansion and ReLU."""
        x_norm = self._rms_batch_norm(x, f'{name}_norm')
        hidden_dim = x.shape[-1] * 2
        
        x_hidden = hk.Linear(hidden_dim, name=f'{name}_linear1')(x_norm)
        x_hidden = hk.dropout(hk.next_rng_key(), 0.3, jax.nn.relu(x_hidden)) if hk.running_init() else jax.nn.relu(x_hidden)
        x_out = hk.Linear(x.shape[-1], name=f'{name}_linear2')(x_hidden)
        
        return self._rms_batch_norm(x_out, f'{name}_out_norm')
    
    # ========== Pairwise Interaction Blocks ==========
    
    def _pair_update_block(self, sequence_input: jnp.ndarray, 
                          pair_input: Optional[jnp.ndarray], 
                          name: str) -> jnp.ndarray:
        """Update pairwise representations from sequence."""
        y = self._sequence_to_pair_block(sequence_input, name)
        
        if pair_input is None:
            x = y
        else:
            x = pair_input + y
            
        x = x + self._row_attention_block(x, f'{name}_row_attn')
        x = x + self._pair_mlp_block(x, f'{name}_pair_mlp')
        
        return x
    
    def _sequence_to_pair_block(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Transform sequence to pairwise representation (512x512x128)."""
        # Downsample to P=512 length (factor 16: 8192->512)
        x = hk.avg_pool(x, window_shape=16, strides=16, padding='SAME')
        x = self._rms_batch_norm(x, f'{name}_seq_norm')
        
        # 32 query and key heads with 128 features each
        q = hk.Linear(32 * 128, with_bias=False, name=f'{name}_q')(x)
        k = hk.Linear(32 * 128, with_bias=False, name=f'{name}_k')(x)
        
        # Generate positional relative encodings
        pos_features = self._central_mask_features(sequence_length=512, feature_size=64)
        pos_encoding = hk.Linear(32 * 128, name=f'{name}_pos_encoding')(pos_features)
        
        # Learnable biases for relative attention
        q_bias = hk.get_parameter(f'{name}_q_r_bias', shape=(1, 1, 32, 128), init=hk.initializers.RandomNormal(0.01))
        k_bias = hk.get_parameter(f'{name}_k_r_bias', shape=(1, 1, 32, 128), init=hk.initializers.RandomNormal(0.01))
        
        # Relative attention terms
        q = q.reshape(x.shape[0], -1, 32, 128)
        k = k.reshape(x.shape[0], -1, 32, 128)
        pos_encoding = pos_encoding.reshape(-1, 32, 128)
        
        rel_q_a = self._relative_shift(jnp.einsum('bqhc,phc->bqph', q + q_bias, pos_encoding))
        rel_k_a = self._relative_shift(jnp.einsum('bkhc,phc->bkph', k + k_bias, pos_encoding))
        
        # Combine query-key and relative terms
        a = jnp.einsum('bqhc,bkhc->bqkh', q, k) + (rel_q_a + jnp.swapaxes(rel_k_a, 1, 2)) / 2
        
        # Additional projection from outer sum
        y_q = hk.Linear(128, with_bias=False, name=f'{name}_y_q')(jax.nn.gelu(x))
        y_k = hk.Linear(128, with_bias=False, name=f'{name}_y_k')(jax.nn.gelu(x))
        
        pair_activations = hk.Linear(128, name=f'{name}_proj')(a) + y_q[:, :, None, :] + y_k[:, None, :, :]
        
        # Dropout
        pair_activations = hk.dropout(hk.next_rng_key(), 0.3, pair_activations) if hk.running_init() else pair_activations
        
        return pair_activations
    
    def _central_mask_features(self, sequence_length: int, feature_size: int) -> jnp.ndarray:
        """Generate central mask features for relative positions."""
        # relative_positions spans from -(L-1) to +(L-1)
        relative_positions = jnp.arange(2 * sequence_length - 1) - (sequence_length - 1)
        center_widths = jnp.arange(feature_size // 2) + jnp.geomspace(
            1, sequence_length - feature_size // 2 + 1, feature_size // 2)
        
        embeddings = (center_widths[None, :] > jnp.abs(relative_positions)[:, None]).astype(jnp.float32)
        
        # Add directionality
        directional = jnp.sign(relative_positions)[:, None] * embeddings
        return jnp.concatenate([embeddings, directional], axis=-1)
    
    def _relative_shift(self, x: jnp.ndarray) -> jnp.ndarray:
        """Relative shift operation for efficient relative attention."""
        *batch_shapes, seq_length, num_diagonals = x.shape
        x = jnp.pad(x, [(0, 0)] * len(batch_shapes) + [(0, 0), (1, 0)])
        x = x.reshape(batch_shapes + [num_diagonals + 1, seq_length])
        return x[..., 1:, :].reshape(batch_shapes + [seq_length, num_diagonals])
    
    def _row_attention_block(self, pair_input: jnp.ndarray, name: str) -> jnp.ndarray:
        """Row attention over pairwise matrix."""
        x = self._rms_batch_norm(pair_input, f'{name}_norm')
        
        # Single head attention with 128 features
        k = hk.Linear(128, with_bias=False, name=f'{name}_k')(x)
        q = hk.Linear(128, with_bias=False, name=f'{name}_q')(x)
        v = hk.Linear(128, name=f'{name}_v')(x)
        
        # Row-wise attention (each row i attends to all columns k)
        attn_logits = jnp.einsum('bpPf,bpkf->bpPk', q, k) / jnp.sqrt(128)
        attn_weights = jax.nn.softmax(attn_logits, axis=3)
        attn_out = jnp.einsum('bpPk,bpkf->bpPf', attn_weights, v)
        
        return hk.dropout(hk.next_rng_key(), 0.3, attn_out) if hk.running_init() else attn_out
    
    def _pair_mlp_block(self, pair_input: jnp.ndarray, name: str) -> jnp.ndarray:
        """MLP for pairwise representations."""
        x = self._rms_batch_norm(pair_input, f'{name}_norm')
        hidden_dim = pair_input.shape[-1] * 2
        
        x = hk.Linear(hidden_dim, name=f'{name}_linear1')(x)
        x = jax.nn.relu(x)
        x = hk.Linear(pair_input.shape[-1], name=f'{name}_linear2')(x)
        
        return hk.dropout(hk.next_rng_key(), 0.3, x) if hk.running_init() else x
    
    def _attention_bias_block(self, pair_activations: jnp.ndarray, name: str) -> jnp.ndarray:
        """Convert pairwise activations to attention bias."""
        x = jax.nn.gelu(self._rms_batch_norm(pair_activations, f'{name}_norm'))
        x = hk.Linear(8, with_bias=False, name=f'{name}_proj')(x)
        
        # Repeat to match sequence length (512 -> 8192, factor 16)
        x = jnp.repeat(jnp.repeat(x, 16, axis=1), 16, axis=2)
        
        # Move head dimension to position 1
        return jnp.moveaxis(x, 3, 1)
    
    # ========== Output Embedders ==========
    
    def _output_embedder(self, x: jnp.ndarray, organism_index: int, 
                        skip_x: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Generate output embeddings with organism-specific components."""
        # Project to 2x channels
        x = hk.Linear(2 * x.shape[-1], name='output_embed_proj')(x)
        
        # Add skip connection if provided (128bp embeddings for 1bp path)
        if skip_x is not None:
            skip_proj = hk.Linear(x.shape[-1], with_bias=False, name='output_embed_skip')(skip_x)
            # Repeat skip to match sequence length
            repeat_factor = x.shape[1] // skip_proj.shape[1]
            if repeat_factor > 1:
                skip_proj = jnp.repeat(skip_proj, repeat_factor, axis=1)
            x = x + skip_proj
        
        # Add organism embedding and apply activation
        organism_embed = hk.Embed(x.shape[-1], name='output_organism_embed')(organism_index)
        x = jax.nn.gelu(self._rms_batch_norm(x, 'output_embed_norm') + organism_embed[:, None, :])
        
        return x
    
    def _output_pair_embedder(self, pair_activations: jnp.ndarray, organism_index: int) -> jnp.ndarray:
        """Generate output embeddings for pairwise representations."""
        # Symmetrize pairwise matrix
        x = (pair_activations + jnp.swapaxes(pair_activations, 1, 2)) / 2.0
        
        # Add organism embedding
        organism_embed = hk.Embed(128, name='pair_organism_embed')(organism_index)
        x = jax.nn.gelu(self._rms_batch_norm(x, 'pair_embed_norm') + organism_embed[:, None, None, :])
        
        return x
    
    # ========== Prediction Heads ==========
    
    def _tracks_scaled_predictions(self, embeddings: jnp.ndarray, num_tracks: int, name: str) -> jnp.ndarray:
        """Prediction head for genomic tracks with scaling."""
        # Linear projection
        x = hk.Linear(num_tracks, name=f'{name}_linear')(embeddings)
        
        # Learnable per-track scaling
        scale = hk.get_parameter(f'{name}_scale', shape=(num_tracks,), 
                               init=hk.initializers.Constant(0.0))
        
        # Softplus for non-negative outputs with scaling
        return jax.nn.softplus(x) * jax.nn.softplus(scale)
    
    def _contact_map_head(self, embeddings_pair: jnp.ndarray, num_tracks: int) -> jnp.ndarray:
        """Contact map prediction head."""
        return hk.Linear(num_tracks, name='contact_linear')(embeddings_pair)
    
    def _splice_sites_head(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """Splice site classification head (5 classes)."""
        logits = hk.Linear(5, name='splice_sites_linear')(embeddings)
        return jax.nn.softmax(logits, axis=-1)
    
    def _splice_usage_head(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """Splice site usage head."""
        num_tissues = self.config.get('num_tissues', 49)
        num_strands = 2
        
        logits = hk.Linear(num_tissues * num_strands, name='splice_usage_linear')(embeddings)
        logits = logits.reshape(*embeddings.shape[:-1], num_strands, num_tissues)
        
        return jax.nn.sigmoid(logits)
    
    def _splice_junctions_head(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        Splice junction prediction head following paper's detailed implementation.
        
        Returns junction embeddings that will be processed by tissue_scaled_rope
        and donor-acceptor pairing in the loss/scoring functions.
        """
        # Project to intermediate dimension (512) as in paper
        junction_embeddings = hk.Linear(512, name='splice_junction_proj')(embeddings)
        return junction_embeddings
    
    def tissue_scaled_rope(self, x: jnp.ndarray, indices: jnp.ndarray, 
                          num_tissues: int, name: str) -> jnp.ndarray:
        """
        Apply tissue-specific scaling and RoPE to splice site embeddings.
        
        Args:
            x: [B, S, 512] sequence embeddings  
            indices: [P] splice site positions
            num_tissues: number of tissues
            name: name for parameters
            
        Returns:
            [B, P, num_tissues, 512] tissue-scaled embeddings with RoPE
        """
        # Extract embeddings at splice site positions
        x_sites = x[:, indices, :]  # [B, P, 512]
        
        # Tissue-specific scaling and offset
        scale = hk.get_parameter(f'{name}_scale', shape=(num_tissues, 512), 
                               init=hk.initializers.RandomNormal(0.01))
        offset = hk.get_parameter(f'{name}_offset', shape=(num_tissues, 512),
                                init=hk.initializers.Constant(0.0))
        
        # Apply tissue scaling: [B, P, num_tissues, 512]
        x_scaled = scale[None, None, :, :] * x_sites[:, :, None, :] + offset[None, None, :, :]
        
        # Apply RoPE with genomic positions
        x_rope = self._apply_rope_positions(x_scaled, indices, max_position=2**20)
        
        return x_rope
    
    def _apply_rope_positions(self, x: jnp.ndarray, positions: jnp.ndarray, max_position: int) -> jnp.ndarray:
        """Apply RoPE using actual genomic positions."""
        head_dim = x.shape[-1]
        num_freq = head_dim // 2
        
        # Modified frequency calculation as in paper
        freq = 1.0 / (jnp.arange(num_freq) + jnp.geomspace(1, max_position - num_freq + 1, num_freq))
        theta = jnp.repeat(jnp.outer(positions, freq), 2, axis=-1)
        
        # Broadcast theta to match x shape
        theta = theta[None, :, None, :]  # [1, P, 1, 512]
        theta = jnp.broadcast_to(theta, x.shape)
        
        # Apply rotation
        x_rotated = jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)
        return x * jnp.cos(theta) + x_rotated * jnp.sin(theta)


def create_alphagenome_model(config: Dict[str, Any]) -> hk.Transformed:
    """
    Create a transformed AlphaGenome model for training/inference.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Haiku transformed model
    """
    def model_fn(dna_sequence: jnp.ndarray, organism_id: jnp.ndarray):
        model = AlphaGenomeModel(config)
        return model(dna_sequence, organism_id)
    
    return hk.transform(model_fn)


# ========== Loss Functions ==========

def multinomial_loss(predictions: jnp.ndarray, targets: jnp.ndarray, 
                    multinomial_resolution: int) -> jnp.ndarray:
    """
    Multinomial + Poisson loss as described in paper.
    
    Args:
        predictions: [B, S, C] predicted tracks
        targets: [B, S, C] target tracks  
        multinomial_resolution: segment size for loss calculation
        
    Returns:
        Combined loss value
    """
    # Reshape into segments
    x = predictions.reshape((-1, multinomial_resolution, predictions.shape[-1]))
    targets_seg = targets.reshape((-1, multinomial_resolution, targets.shape[-1]))
    
    # Sum within each segment
    sum_pred = jnp.sum(x, axis=1, keepdims=True)
    sum_target = jnp.sum(targets_seg, axis=1, keepdims=True)
    
    # Poisson loss on segment totals
    poisson_loss = jnp.sum(sum_pred - sum_target * jnp.log(sum_pred + 1e-7))
    
    # Multinomial loss on within-segment distributions
    multinomial_prob = x / (sum_pred + 1e-7)
    positional_loss = jnp.sum(-targets_seg * jnp.log(multinomial_prob + 1e-7))
    
    return poisson_loss / multinomial_resolution + 5.0 * positional_loss


def junctions_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Splice junction loss combining cross-entropy and Poisson terms.
    
    Args:
        predictions: [B, D, A, num_tissues] predicted junction counts
        targets: [B, D, A, num_tissues] target junction counts
        
    Returns:
        Combined junction loss
    """
    def soft_clip(x):
        return jnp.where(x > 10.0, 2 * jnp.sqrt(x * 10.0) - 10.0, x)
    
    def multinomial_cross_entropy(pred, targ, axis):
        pred_ratios = (pred + 1e-7) / jnp.sum(pred + 1e-7, axis=axis, keepdims=True)
        target_ratios = (targ + 1e-7) / jnp.sum(targ + 1e-7, axis=axis, keepdims=True)
        return -jnp.sum(targ * jnp.log(pred_ratios))
    
    def poisson_loss_term(pred, targ, axis):
        sum_pred = jnp.sum(pred, axis=axis)
        sum_targets = soft_clip(jnp.sum(targ, axis=axis))
        return jnp.sum(sum_pred - sum_targets * jnp.log(sum_pred + 1e-7))
    
    # PSI5 and PSI3 cross-entropy terms
    ratios_loss = (multinomial_cross_entropy(predictions, targets, axis=0) +
                   multinomial_cross_entropy(predictions, targets, axis=1))
    
    # Marginal Poisson terms
    counts_loss = (poisson_loss_term(predictions, targets, axis=0) +
                   poisson_loss_term(predictions, targets, axis=1))
    
    return 0.2 * ratios_loss + 0.04 * counts_loss