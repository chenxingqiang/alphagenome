"""
AlphaGenome Distillation Training

Implements the exact distillation procedure from the AlphaGenome paper.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass

from .model import create_alphagenome_model


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""
    distillation_steps: int = 250000
    batch_size: int = 64
    peak_lr: float = 0.002
    warmup_steps: int = 5000
    constant_steps: int = 120000
    weight_decay: float = 0.04
    dropout_rate: float = 0.1
    mutation_rate: float = 0.04
    sv_lambda: float = 1.0
    sv_max_length: int = 20
    reverse_complement_prob: float = 0.5
    checkpoint_interval: int = 25000


class SequenceAugmenter:
    """Sequence augmentation for distillation training."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
    
    def augment_sequence(self, 
                        sequence: jnp.ndarray,
                        rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply all augmentations to a sequence."""
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        
        # Random mutations (4% of nucleotides)
        sequence = self._apply_random_mutations(sequence, rng1)
        
        # Structural variations
        sequence = self._apply_structural_variations(sequence, rng2)
        
        # Reverse complement (50% probability)
        sequence = self._apply_reverse_complement(sequence, rng3)
        
        return sequence
    
    def _apply_random_mutations(self,
                               sequence: jnp.ndarray,
                               rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply random point mutations to 4% of nucleotides."""
        seq_length = sequence.shape[0]
        num_mutations = int(self.config.mutation_rate * seq_length)
        
        # Sample mutation positions
        mutation_positions = jax.random.choice(
            rng, seq_length, shape=(num_mutations,), replace=False
        )
        
        # Apply mutations
        def apply_mutation(i, seq):
            pos = mutation_positions[i]
            new_base = jax.random.choice(jax.random.fold_in(rng, i), 4)
            new_onehot = jnp.zeros(4).at[new_base].set(1.0)
            return seq.at[pos].set(new_onehot)
        
        return jax.lax.fori_loop(0, num_mutations, apply_mutation, sequence)
    
    def _apply_structural_variations(self,
                                   sequence: jnp.ndarray,
                                   rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply structural variations (inversions)."""
        seq_length = sequence.shape[0]
        num_svs = jax.random.poisson(rng, self.config.sv_lambda)
        
        def apply_sv(i, seq):
            sv_rng = jax.random.fold_in(rng, i)
            sv_rng1, sv_rng2 = jax.random.split(sv_rng)
            
            sv_length = jax.random.randint(sv_rng1, (), 1, self.config.sv_max_length + 1)
            max_start = seq_length - sv_length
            sv_start = jax.random.randint(sv_rng2, (), 0, max_start)
            
            # Apply inversion (reverse and complement)
            segment = seq[sv_start:sv_start + sv_length]
            segment = jnp.flip(segment, axis=0)
            complement_map = jnp.array([3, 2, 1, 0])  # A,C,G,T -> T,G,C,A
            segment = segment[:, complement_map]
            
            return seq.at[sv_start:sv_start + sv_length].set(segment)
        
        return jax.lax.cond(
            num_svs > 0,
            lambda seq: jax.lax.fori_loop(0, num_svs, apply_sv, seq),
            lambda seq: seq,
            sequence
        )
    
    def _apply_reverse_complement(self,
                                 sequence: jnp.ndarray,
                                 rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply reverse complement with 50% probability."""
        should_reverse = jax.random.bernoulli(rng, self.config.reverse_complement_prob)
        
        def reverse_complement(seq):
            seq = jnp.flip(seq, axis=0)
            complement_map = jnp.array([3, 2, 1, 0])
            return seq[:, complement_map]
        
        return jax.lax.cond(should_reverse, reverse_complement, lambda seq: seq, sequence)


class TeacherEnsemble:
    """Ensemble of teacher models for distillation."""
    
    def __init__(self, teacher_models: List[Dict], model_fn: Callable):
        self.teachers = teacher_models
        self.model_fn = model_fn
        self.num_teachers = len(teacher_models)
    
    def get_teacher_prediction(self,
                              teacher_idx: int,
                              sequence: jnp.ndarray,
                              organism_id: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Get prediction from a specific teacher."""
        teacher_params = self.teachers[teacher_idx]
        return self.model_fn.apply(teacher_params, sequence, organism_id, train=False)
    
    def sample_teacher(self, rng: jax.random.PRNGKey) -> int:
        """Sample a random teacher index."""
        return jax.random.randint(rng, (), 0, self.num_teachers)


class AlphaGenomeDistiller:
    """AlphaGenome distillation trainer."""
    
    def __init__(self, 
                 config: DistillationConfig,
                 model_config: Dict[str, Any]):
        self.config = config
        self.model_config = model_config
        self.logger = logging.getLogger(__name__)
        
        self.model = create_alphagenome_model(model_config)
        self.augmenter = SequenceAugmenter(config)
        self.optimizer = self._create_optimizer()
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create distillation optimizer with three-phase schedule."""
        def three_phase_schedule(step):
            warmup_end = self.config.warmup_steps
            constant_end = warmup_end + self.config.constant_steps
            
            warmup_lr = self.config.peak_lr * (step / warmup_end)
            constant_lr = self.config.peak_lr
            
            decay_steps = self.config.distillation_steps - constant_end
            decay_progress = (step - constant_end) / decay_steps
            cosine_lr = self.config.peak_lr * 0.5 * (1 + jnp.cos(jnp.pi * decay_progress))
            
            return jnp.where(
                step < warmup_end,
                warmup_lr,
                jnp.where(step < constant_end, constant_lr, cosine_lr)
            )
        
        return optax.adamw(
            learning_rate=three_phase_schedule,
            b1=0.9, b2=0.999, eps=1e-8,
            weight_decay=self.config.weight_decay
        )
    
    def distill_student(self,
                       teacher_ensemble: TeacherEnsemble,
                       checkpoint_dir: Path) -> Dict:
        """Distill student model from teacher ensemble."""
        self.logger.info(f"Starting distillation with {teacher_ensemble.num_teachers} teachers")
        
        # Initialize student model
        rng = jax.random.PRNGKey(42)
        dummy_seq = jnp.ones((self.model_config['sequence_length'], 4))
        dummy_org = jnp.array([0])
        
        student_params = self.model.init(rng, dummy_seq[None], dummy_org)
        opt_state = self.optimizer.init(student_params)
        
        # Training loop
        for step in range(self.config.distillation_steps):
            step_rng = jax.random.fold_in(rng, step)
            batch = self._generate_random_batch(step_rng)
            
            student_params, opt_state, metrics = self._distillation_step(
                student_params, opt_state, teacher_ensemble, batch, step_rng
            )
            
            if step % 1000 == 0:
                self.logger.info(
                    f"Step {step}/{self.config.distillation_steps}, "
                    f"Loss: {metrics['loss']:.4f}"
                )
            
            if step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(student_params, opt_state, checkpoint_dir, step)
        
        return student_params
    
    def _distillation_step(self,
                          student_params: Dict,
                          opt_state: optax.OptState,
                          teacher_ensemble: TeacherEnsemble,
                          batch: Dict[str, jnp.ndarray],
                          rng: jax.random.PRNGKey) -> Tuple[Dict, optax.OptState, Dict]:
        """Single distillation training step."""
        
        def distill_loss_fn(student_params):
            rng1, rng2, rng3 = jax.random.split(rng, 3)
            
            # Apply augmentation
            aug_sequences = jax.vmap(self.augmenter.augment_sequence, in_axes=(0, 0))(
                batch['sequences'], jax.random.split(rng1, batch['sequences'].shape[0])
            )
            
            # Sample teacher
            teacher_idx = teacher_ensemble.sample_teacher(rng2)
            
            # Get predictions
            teacher_preds = teacher_ensemble.get_teacher_prediction(
                teacher_idx, aug_sequences, batch['organism_ids']
            )
            
            student_preds = self.model.apply(
                student_params, aug_sequences, batch['organism_ids'],
                train=True, dropout_rate=self.config.dropout_rate,
                rngs={'dropout': rng3}
            )
            
            # Compute loss
            loss = self._compute_distillation_loss(student_preds, teacher_preds)
            return loss, {'teacher_idx': teacher_idx}
        
        (loss, aux), grads = jax.value_and_grad(distill_loss_fn, has_aux=True)(student_params)
        
        updates, opt_state = self.optimizer.update(grads, opt_state, student_params)
        student_params = optax.apply_updates(student_params, updates)
        
        metrics = {
            'loss': float(loss),
            'grad_norm': float(optax.global_norm(grads)),
            'teacher_idx': int(aux['teacher_idx'])
        }
        
        return student_params, opt_state, metrics
    
    def _compute_distillation_loss(self,
                                  student_preds: Dict[str, jnp.ndarray],
                                  teacher_preds: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute distillation loss between student and teacher predictions."""
        total_loss = 0.0
        
        # Track-based losses
        track_modalities = ['rna_seq', 'atac_seq', 'dnase_seq', 'cage', 'pro_cap']
        for modality in track_modalities:
            if modality in student_preds and modality in teacher_preds:
                loss = jnp.mean((student_preds[modality] - teacher_preds[modality]) ** 2)
                total_loss += loss
        
        # ChIP-seq losses
        for chip_type in ['chip_tf', 'chip_histone']:
            if chip_type in student_preds and chip_type in teacher_preds:
                loss = jnp.mean((student_preds[chip_type] - teacher_preds[chip_type]) ** 2)
                total_loss += loss
        
        # Contact map loss
        if 'contact_maps' in student_preds and 'contact_maps' in teacher_preds:
            loss = jnp.mean((student_preds['contact_maps'] - teacher_preds['contact_maps']) ** 2)
            total_loss += 0.1 * loss
        
        # Splicing losses
        splicing_keys = ['splice_sites', 'splice_usage', 'splice_junctions']
        for key in splicing_keys:
            if key in student_preds and key in teacher_preds:
                loss = jnp.mean((student_preds[key] - teacher_preds[key]) ** 2)
                total_loss += loss
        
        return total_loss
    
    def _generate_random_batch(self, rng: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Generate random batch for distillation."""
        batch_size = self.config.batch_size
        seq_length = self.model_config['sequence_length']
        
        rng1, rng2 = jax.random.split(rng)
        
        nucleotide_indices = jax.random.randint(rng1, (batch_size, seq_length), 0, 4)
        sequences = jnp.eye(4)[nucleotide_indices]
        organism_ids = jax.random.randint(rng2, (batch_size,), 0, 2)
        
        return {'sequences': sequences, 'organism_ids': organism_ids}
    
    def _save_checkpoint(self, params: Dict, opt_state: optax.OptState, 
                        checkpoint_dir: Path, step: int):
        """Save training checkpoint."""
        checkpoint_path = checkpoint_dir / f"student_checkpoint_step_{step}.pkl"
        
        checkpoint = {
            'params': params,
            'opt_state': opt_state,
            'step': step,
            'config': self.config,
            'model_config': self.model_config
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)


def run_distillation_pipeline(model_config: Dict[str, Any],
                             distill_config: DistillationConfig = None,
                             checkpoint_dir: str = "./checkpoints") -> Dict:
    """Run complete distillation pipeline."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    
    if distill_config is None:
        distill_config = DistillationConfig()
    
    # Create dummy teachers for demo
    model = create_alphagenome_model(model_config)
    teachers = []
    for i in range(4):  # Use 4 teachers for demo instead of 64
        rng = jax.random.PRNGKey(1000 + i)
        dummy_seq = jnp.ones((model_config['sequence_length'], 4))
        dummy_org = jnp.array([0])
        params = model.init(rng, dummy_seq[None], dummy_org)
        teachers.append(params)
    
    teacher_ensemble = TeacherEnsemble(teachers, model)
    
    # Distill student
    distiller = AlphaGenomeDistiller(distill_config, model_config)
    student_params = distiller.distill_student(teacher_ensemble, checkpoint_path)
    
    # Save final model
    final_checkpoint = {
        'params': student_params,
        'config': model_config,
        'distill_config': distill_config,
        'type': 'student_final'
    }
    
    final_path = checkpoint_path / "alphagenome_student_final.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(final_checkpoint, f)
    
    return {
        'student_params': student_params,
        'num_teachers': len(teachers),
        'final_checkpoint': str(final_path),
        'distillation_steps': distill_config.distillation_steps
    } 