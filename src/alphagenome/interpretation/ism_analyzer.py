"""
In Silico Mutagenesis (ISM) Analysis Tool

Implements the exact ISM procedure described in the AlphaGenome paper for
sequence interpretation and variant effect analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

from ..core.model import AlphaGenomeModel
from ..core.variant_scorer import VariantScorer, Variant
from ..data import genome
from .ism import ism_variants, ism_matrix


@dataclass
class ISMConfig:
    """Configuration for ISM analysis."""
    window_size: int = 256  # Size of ISM window
    center_position: Optional[int] = None  # Center position for ISM
    target_modality: str = 'rna_seq'  # Target modality for analysis
    target_track: Optional[str] = None  # Specific track name
    organism: str = 'human'  # Target organism
    score_threshold: float = 0.01  # Minimum score threshold
    

class ISMAnalyzer:
    """
    In Silico Mutagenesis analyzer implementing paper methodology.
    
    Paper: "To interpret which nucleotides in a sequence of interest contribute 
    most to AlphaGenome's predictions for a specific genomic feature, we employed 
    an in silico mutagenesis (ISM) approach."
    """
    
    def __init__(self, 
                 model: AlphaGenomeModel,
                 model_params: Dict,
                 variant_scorer: VariantScorer,
                 config: ISMConfig = None):
        self.model = model
        self.model_params = model_params
        self.variant_scorer = variant_scorer
        self.config = config or ISMConfig()
        self.logger = logging.getLogger(__name__)
    
    def analyze_sequence(self,
                        sequence: str,
                        interval: genome.Interval,
                        target_feature: str = 'expression') -> Dict[str, Any]:
        """
        Perform ISM analysis on a sequence.
        
        Args:
            sequence: DNA sequence to analyze
            interval: Genomic interval for the sequence
            target_feature: Target feature to analyze
            
        Returns:
            ISM analysis results including contribution scores and visualizations
        """
        self.logger.info(f"Starting ISM analysis for {interval}")
        
        # 1. Generate all possible single nucleotide variants
        ism_interval = self._get_ism_interval(interval)
        variants = ism_variants(ism_interval, sequence[ism_interval.start - interval.start:ism_interval.end - interval.start])
        
        self.logger.info(f"Generated {len(variants)} variants for ISM")
        
        # 2. Score all variants
        variant_scores = self._score_variants(variants, sequence, interval, target_feature)
        
        # 3. Construct ISM matrix
        ism_result = ism_matrix(
            variant_scores=variant_scores,
            variants=variants,
            interval=ism_interval,
            multiply_by_sequence=True
        )
        
        # 4. Identify important motifs
        motifs = self._identify_motifs(ism_result, sequence, ism_interval)
        
        # 5. Generate sequence logo data
        logo_data = self._generate_sequence_logo_data(ism_result, sequence, ism_interval)
        
        return {
            'ism_matrix': ism_result,
            'variant_scores': variant_scores,
            'motifs': motifs,
            'logo_data': logo_data,
            'interval': ism_interval,
            'target_feature': target_feature,
            'num_variants': len(variants)
        }
    
    def comparative_ism_analysis(self,
                                ref_sequence: str,
                                alt_sequence: str,
                                variant: Variant,
                                interval: genome.Interval) -> Dict[str, Any]:
        """
        Perform comparative ISM analysis for variant interpretation.
        
        Paper: "Furthermore, to specifically investigate how a genetic variant 
        might alter local sequence motifs, this ISM procedure is applied 
        independently to both the reference (REF) sequence and the sequence 
        containing the alternative (ALT) allele."
        """
        self.logger.info(f"Starting comparative ISM for variant {variant}")
        
        # Analyze reference sequence
        ref_results = self.analyze_sequence(ref_sequence, interval, 'expression')
        
        # Analyze alternative sequence  
        alt_results = self.analyze_sequence(alt_sequence, interval, 'expression')
        
        # Compare results
        comparison = self._compare_ism_results(ref_results, alt_results, variant)
        
        return {
            'reference': ref_results,
            'alternative': alt_results,
            'comparison': comparison,
            'variant': variant
        }
    
    def _get_ism_interval(self, interval: genome.Interval) -> genome.Interval:
        """Get ISM interval centered on the main interval."""
        if self.config.center_position:
            center = self.config.center_position
        else:
            center = interval.start + interval.width // 2
        
        half_window = self.config.window_size // 2
        return genome.Interval(
            chromosome=interval.chromosome,
            start=center - half_window,
            end=center + half_window
        )
    
    def _score_variants(self,
                       variants: List[Variant],
                       sequence: str,
                       interval: genome.Interval,
                       target_feature: str) -> List[float]:
        """Score all variants using the variant scorer."""
        scores = []
        
        for i, variant in enumerate(variants):
            try:
                # Create mutated sequence
                mut_sequence = self._apply_variant_to_sequence(sequence, variant, interval)
                
                # Score variant effect
                score = self._compute_variant_effect_score(
                    sequence, mut_sequence, variant, target_feature
                )
                scores.append(score)
                
                if i % 100 == 0:
                    self.logger.debug(f"Scored {i}/{len(variants)} variants")
                    
            except Exception as e:
                self.logger.warning(f"Failed to score variant {variant}: {e}")
                scores.append(0.0)
        
        return scores
    
    def _apply_variant_to_sequence(self,
                                  sequence: str,
                                  variant: Variant,
                                  interval: genome.Interval) -> str:
        """Apply variant to sequence."""
        # Convert to relative position
        rel_pos = variant.position - interval.start - 1
        
        if rel_pos < 0 or rel_pos >= len(sequence):
            return sequence
        
        # Apply mutation
        mut_sequence = list(sequence)
        mut_sequence[rel_pos] = variant.alt_allele
        return ''.join(mut_sequence)
    
    def _compute_variant_effect_score(self,
                                     ref_sequence: str,
                                     alt_sequence: str,
                                     variant: Variant,
                                     target_feature: str) -> float:
        """Compute variant effect score for target feature."""
        try:
            # Get predictions for both sequences
            ref_pred = self._predict_sequence(ref_sequence)
            alt_pred = self._predict_sequence(alt_sequence)
            
            # Extract target feature scores
            ref_score = self._extract_feature_score(ref_pred, target_feature)
            alt_score = self._extract_feature_score(alt_pred, target_feature)
            
            # Compute difference
            return float(alt_score - ref_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to compute effect score: {e}")
            return 0.0
    
    def _predict_sequence(self, sequence: str) -> Dict[str, jnp.ndarray]:
        """Make prediction for a sequence."""
        from ..core.data_processor import DNAEncoder
        
        encoder = DNAEncoder()
        encoded_seq = encoder.encode_sequence(sequence)
        
        # Add batch dimension
        batch_seq = jnp.array(encoded_seq)[None, :, :]
        organism_id = jnp.array([0 if self.config.organism == 'human' else 1])
        
        # Make prediction
        predictions = self.model.apply(self.model_params, batch_seq, organism_id)
        
        # Remove batch dimension
        return {k: v[0] for k, v in predictions.items()}
    
    def _extract_feature_score(self,
                              predictions: Dict[str, jnp.ndarray],
                              target_feature: str) -> float:
        """Extract score for target feature."""
        if target_feature == 'expression':
            # Sum RNA-seq signal
            if 'rna_seq' in predictions:
                return float(jnp.sum(predictions['rna_seq']))
        elif target_feature == 'accessibility':
            # Sum accessibility signal
            if 'atac_seq' in predictions:
                return float(jnp.sum(predictions['atac_seq']))
        elif target_feature in predictions:
            return float(jnp.sum(predictions[target_feature]))
        
        return 0.0
    
    def _identify_motifs(self,
                        ism_matrix: np.ndarray,
                        sequence: str,
                        interval: genome.Interval) -> List[Dict]:
        """Identify significant motifs from ISM results."""
        motifs = []
        
        # Find positions with high contribution scores
        contribution_scores = np.max(np.abs(ism_matrix), axis=1)
        threshold = np.percentile(contribution_scores, 95)
        
        high_contrib_positions = np.where(contribution_scores > threshold)[0]
        
        # Group nearby positions into motifs
        motif_regions = self._group_positions(high_contrib_positions)
        
        for start, end in motif_regions:
            motif_seq = sequence[start:end+1]
            motif_score = np.sum(contribution_scores[start:end+1])
            
            motifs.append({
                'sequence': motif_seq,
                'start': interval.start + start,
                'end': interval.start + end,
                'score': float(motif_score),
                'length': end - start + 1
            })
        
        # Sort by score
        motifs.sort(key=lambda x: x['score'], reverse=True)
        
        return motifs
    
    def _group_positions(self, positions: np.ndarray, max_gap: int = 3) -> List[Tuple[int, int]]:
        """Group nearby positions into regions."""
        if len(positions) == 0:
            return []
        
        regions = []
        current_start = positions[0]
        current_end = positions[0]
        
        for pos in positions[1:]:
            if pos - current_end <= max_gap:
                current_end = pos
            else:
                regions.append((current_start, current_end))
                current_start = pos
                current_end = pos
        
        regions.append((current_start, current_end))
        return regions
    
    def _generate_sequence_logo_data(self,
                                   ism_matrix: np.ndarray,
                                   sequence: str,
                                   interval: genome.Interval) -> Dict:
        """Generate data for sequence logo visualization."""
        bases = ['A', 'C', 'G', 'T']
        logo_data = []
        
        for i in range(len(sequence)):
            position_data = {
                'position': interval.start + i,
                'reference_base': sequence[i],
                'scores': {}
            }
            
            for j, base in enumerate(bases):
                position_data['scores'][base] = float(ism_matrix[i, j])
            
            logo_data.append(position_data)
        
        return {
            'positions': logo_data,
            'interval': interval,
            'sequence': sequence
        }
    
    def _compare_ism_results(self,
                           ref_results: Dict,
                           alt_results: Dict,
                           variant: Variant) -> Dict:
        """Compare ISM results between reference and alternative."""
        comparison = {
            'variant_position': variant.position,
            'motif_changes': [],
            'score_differences': {},
            'new_motifs': [],
            'lost_motifs': []
        }
        
        # Compare motifs
        ref_motifs = ref_results['motifs']
        alt_motifs = alt_results['motifs']
        
        # Find motifs that appear/disappear
        ref_seqs = {m['sequence'] for m in ref_motifs}
        alt_seqs = {m['sequence'] for m in alt_motifs}
        
        comparison['new_motifs'] = [m for m in alt_motifs if m['sequence'] not in ref_seqs]
        comparison['lost_motifs'] = [m for m in ref_motifs if m['sequence'] not in alt_seqs]
        
        # Score differences around variant
        var_rel_pos = variant.position - ref_results['interval'].start
        window = 20  # 20bp window around variant
        
        start_idx = max(0, var_rel_pos - window)
        end_idx = min(len(ref_results['ism_matrix']), var_rel_pos + window)
        
        ref_scores = ref_results['ism_matrix'][start_idx:end_idx]
        alt_scores = alt_results['ism_matrix'][start_idx:end_idx]
        
        comparison['score_differences'] = {
            'reference_scores': ref_scores.tolist(),
            'alternative_scores': alt_scores.tolist(),
            'differences': (alt_scores - ref_scores).tolist(),
            'window_start': start_idx,
            'window_end': end_idx
        }
        
        return comparison


def run_ism_analysis(model_path: str,
                    sequence: str,
                    interval: genome.Interval,
                    config: ISMConfig = None) -> Dict[str, Any]:
    """
    Convenience function to run ISM analysis.
    
    Args:
        model_path: Path to trained model
        sequence: DNA sequence to analyze
        interval: Genomic interval
        config: ISM configuration
        
    Returns:
        ISM analysis results
    """
    import pickle
    from ..core.model import create_alphagenome_model
    from ..core.variant_scorer import VariantScorer
    
    # Load model
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model_config = checkpoint['config']
    model = create_alphagenome_model(model_config)
    model_params = checkpoint['params']
    
    # Create variant scorer
    variant_scorer = VariantScorer(model, model_params, model_config)
    
    # Create ISM analyzer
    analyzer = ISMAnalyzer(model, model_params, variant_scorer, config)
    
    # Run analysis
    return analyzer.analyze_sequence(sequence, interval) 