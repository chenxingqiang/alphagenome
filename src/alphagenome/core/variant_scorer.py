"""
AlphaGenome Variant Scoring Module

This module implements variant scoring for predicting the effects of genetic variants:
- In silico mutagenesis for single and multiple variants
- Effect scoring across different modalities (expression, chromatin, splicing)
- Quantile calibration for improved interpretability
- Pathogenicity prediction and interpretation
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import logging
from dataclasses import dataclass

from .model import AlphaGenomeModel
from .data_processor import DNAEncoder


@dataclass
class Variant:
    """Represents a genetic variant."""
    chromosome: str
    position: int  # 0-based position
    ref_allele: str
    alt_allele: str
    variant_id: Optional[str] = None


@dataclass
class VariantScore:
    """Contains scoring results for a variant."""
    variant: Variant
    scores: Dict[str, float]
    predictions: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class QuantileCalibrator:
    """Calibrates scores to quantiles for improved interpretability."""
    
    def __init__(self, modality: str):
        self.modality = modality
        self.quantiles = None
        self.bins = None
        
    def fit(self, scores: np.ndarray, n_bins: int = 1000):
        """
        Fit quantile calibrator on a dataset of scores.
        
        Args:
            scores: Array of scores to calibrate
            n_bins: Number of quantile bins
        """
        self.quantiles = np.linspace(0, 1, n_bins + 1)
        self.bins = np.quantile(scores, self.quantiles)
        
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform scores to calibrated quantiles.
        
        Args:
            scores: Scores to transform
            
        Returns:
            Calibrated quantile scores [0, 1]
        """
        if self.bins is None:
            raise ValueError("Calibrator must be fitted before transformation")
        
        # Find which quantile bin each score falls into
        bin_indices = np.searchsorted(self.bins, scores, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(self.quantiles) - 2)
        
        # Linear interpolation within bins
        lower_bounds = self.bins[bin_indices]
        upper_bounds = self.bins[bin_indices + 1]
        
        # Avoid division by zero
        range_mask = upper_bounds > lower_bounds
        interpolation_weights = np.where(
            range_mask,
            (scores - lower_bounds) / (upper_bounds - lower_bounds),
            0.0
        )
        
        quantile_scores = self.quantiles[bin_indices] + interpolation_weights * (
            self.quantiles[bin_indices + 1] - self.quantiles[bin_indices]
        )
        
        return np.clip(quantile_scores, 0.0, 1.0)


class VariantScorer:
    """
    Main class for scoring genetic variants using AlphaGenome.
    
    Implements in silico mutagenesis to predict variant effects across
    multiple genomic modalities.
    """
    
    def __init__(self, 
                 model: AlphaGenomeModel,
                 model_params: Dict,
                 config: Dict[str, Any]):
        self.model = model
        self.model_params = model_params
        self.config = config
        self.dna_encoder = DNAEncoder()
        self.logger = logging.getLogger(__name__)
        
        # Scoring configuration
        self.context_length = config.get('context_length', 524288)  # 512kb context
        self.center_crop = config.get('center_crop', 131072)  # 128kb center region
        self.organism_mapping = {'human': 0, 'mouse': 1}
        
        # Initialize calibrators
        self.calibrators = {}
        self._initialize_calibrators()
    
    def _initialize_calibrators(self):
        """Initialize quantile calibrators for different modalities."""
        modalities = [
            'rna_seq', 'atac_seq', 'dnase_seq', 'cage', 'pro_cap',
            'chip_tf', 'chip_histone', 'contact_maps',
            'splice_sites', 'splice_usage', 'splice_junctions'
        ]
        
        for modality in modalities:
            self.calibrators[modality] = QuantileCalibrator(modality)
    
    def get_sequence_context(self, 
                           reference_genome: str,
                           chromosome: str,
                           position: int,
                           organism: str = 'human') -> Tuple[str, int]:
        """
        Get genomic sequence context around a variant position.
        
        Args:
            reference_genome: Path to reference genome FASTA
            chromosome: Chromosome name
            position: Variant position (0-based)
            organism: Target organism
            
        Returns:
            Tuple of (sequence, center_position_in_sequence)
        """
        # Calculate context window
        half_context = self.context_length // 2
        start_pos = max(0, position - half_context)
        end_pos = start_pos + self.context_length
        
        # Load sequence from reference genome
        try:
            import pysam
            with pysam.FastaFile(reference_genome) as fasta:
                sequence = fasta.fetch(chromosome, start_pos, end_pos)
                
            # Calculate center position in the extracted sequence
            center_pos = position - start_pos
            
            return sequence.upper(), center_pos
            
        except Exception as e:
            self.logger.error(f"Failed to load sequence: {e}")
            # Return random sequence as fallback
            sequence = 'N' * self.context_length
            return sequence, self.context_length // 2
    
    def create_variant_sequence(self, 
                              reference_sequence: str,
                              center_position: int,
                              ref_allele: str,
                              alt_allele: str) -> str:
        """
        Create variant sequence by applying mutation to reference.
        
        Args:
            reference_sequence: Reference DNA sequence
            center_position: Position of variant in sequence
            ref_allele: Reference allele
            alt_allele: Alternative allele
            
        Returns:
            Mutated sequence
        """
        # Convert to mutable list
        seq_list = list(reference_sequence)
        
        # Verify reference allele matches
        ref_in_seq = reference_sequence[center_position:center_position + len(ref_allele)]
        if ref_in_seq.upper() != ref_allele.upper():
            self.logger.warning(
                f"Reference allele mismatch: expected {ref_allele}, "
                f"found {ref_in_seq} at position {center_position}"
            )
        
        # Apply mutation
        # Remove reference allele
        del seq_list[center_position:center_position + len(ref_allele)]
        
        # Insert alternative allele
        for i, nucleotide in enumerate(alt_allele):
            seq_list.insert(center_position + i, nucleotide.upper())
        
        return ''.join(seq_list)
    
    def predict_tracks(self, 
                      sequence: str,
                      organism: str = 'human') -> Dict[str, np.ndarray]:
        """
        Predict genomic tracks for a given sequence.
        
        Args:
            sequence: DNA sequence
            organism: Target organism
            
        Returns:
            Dictionary of predicted tracks
        """
        # Encode sequence
        encoded_seq = self.dna_encoder.encode_sequence(sequence, len(sequence))
        
        # Add batch dimension
        batch_seq = jnp.array(encoded_seq)[None, :, :]
        
        # Get organism ID
        organism_id = jnp.array([self.organism_mapping[organism]])
        
        # Run model prediction
        predictions = self.model.apply(
            self.model_params, 
            batch_seq, 
            organism_id
        )
        
        # Remove batch dimension and convert to numpy
        output = {}
        for key, pred in predictions.items():
            output[key] = np.array(pred[0])  # Remove batch dimension
        
        return output
    
    def compute_variant_effects(self, 
                               ref_predictions: Dict[str, np.ndarray],
                               alt_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute variant effect scores from reference and alternative predictions.
        
        Args:
            ref_predictions: Predictions for reference sequence
            alt_predictions: Predictions for alternative sequence
            
        Returns:
            Dictionary of effect scores
        """
        effects = {}
        
        # Helper function to crop center region
        def crop_center(array: np.ndarray) -> np.ndarray:
            if len(array.shape) == 1:
                # 1D track
                start = (len(array) - self.center_crop) // 2
                end = start + self.center_crop
                return array[start:end]
            elif len(array.shape) == 2:
                # 2D track or contact map
                start = (array.shape[0] - self.center_crop) // 2
                end = start + self.center_crop
                return array[start:end, :]
            elif len(array.shape) == 3:
                # 3D contact map
                start = (array.shape[0] - self.center_crop) // 2
                end = start + self.center_crop
                return array[start:end, start:end, :]
            else:
                return array
        
        # Compute effects for each track type
        for track_name in ref_predictions.keys():
            ref_track = crop_center(ref_predictions[track_name])
            alt_track = crop_center(alt_predictions[track_name])
            
            if 'rna_seq' in track_name or 'expression' in track_name:
                # RNA-seq effects: sum differences
                effect = np.sum(alt_track - ref_track)
                effects[f'{track_name}_effect'] = float(effect)
                
                # Relative effect
                ref_sum = np.sum(ref_track)
                if ref_sum > 0:
                    effects[f'{track_name}_relative_effect'] = float(effect / ref_sum)
                
            elif 'atac' in track_name or 'dnase' in track_name:
                # Accessibility effects: mean squared difference
                mse = np.mean((alt_track - ref_track) ** 2)
                effects[f'{track_name}_mse'] = float(mse)
                
                # Peak-wise effects (identify peaks and measure changes)
                ref_peaks = self._identify_peaks(ref_track)
                alt_peaks = self._identify_peaks(alt_track)
                peak_effect = np.sum(alt_peaks) - np.sum(ref_peaks)
                effects[f'{track_name}_peak_effect'] = float(peak_effect)
                
            elif 'chip' in track_name:
                # ChIP-seq effects: similar to accessibility
                mse = np.mean((alt_track - ref_track) ** 2)
                effects[f'{track_name}_mse'] = float(mse)
                
                # Binding site effects
                binding_effect = np.sum(alt_track) - np.sum(ref_track)
                effects[f'{track_name}_binding_effect'] = float(binding_effect)
                
            elif 'contact' in track_name:
                # 3D genome effects: contact map differences
                contact_diff = np.sum(np.abs(alt_track - ref_track))
                effects[f'{track_name}_diff'] = float(contact_diff)
                
                # Long-range interaction effects
                diagonal_mask = np.eye(alt_track.shape[0], dtype=bool)
                off_diagonal_diff = np.sum(np.abs(alt_track[~diagonal_mask] - ref_track[~diagonal_mask]))
                effects[f'{track_name}_long_range_effect'] = float(off_diagonal_diff)
                
            elif 'splice' in track_name:
                # Splicing effects
                if 'sites' in track_name:
                    # Splice site strength changes
                    site_diff = np.sum(np.abs(alt_track - ref_track))
                    effects[f'{track_name}_site_diff'] = float(site_diff)
                    
                elif 'usage' in track_name:
                    # Splice site usage changes
                    usage_diff = np.sum(alt_track - ref_track)
                    effects[f'{track_name}_usage_diff'] = float(usage_diff)
                    
                elif 'junctions' in track_name:
                    # Junction pattern changes
                    junction_diff = np.sum(np.abs(alt_track - ref_track))
                    effects[f'{track_name}_junction_diff'] = float(junction_diff)
        
        return effects
    
    def _identify_peaks(self, track: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Identify peaks in a genomic track.
        
        Args:
            track: 1D genomic track
            threshold: Peak detection threshold
            
        Returns:
            Binary array indicating peak positions
        """
        from scipy.signal import find_peaks
        
        # Find peaks above threshold
        peaks, _ = find_peaks(track, height=threshold)
        
        # Create binary peak array
        peak_array = np.zeros_like(track)
        peak_array[peaks] = track[peaks]
        
        return peak_array
    
    def score_variant(self, 
                     variant: Variant,
                     reference_genome: str,
                     organism: str = 'human') -> VariantScore:
        """
        Score a single genetic variant.
        
        Args:
            variant: Variant to score
            reference_genome: Path to reference genome
            organism: Target organism
            
        Returns:
            VariantScore object with results
        """
        self.logger.info(f"Scoring variant {variant.variant_id}: {variant.chromosome}:{variant.position} {variant.ref_allele}>{variant.alt_allele}")
        
        # Get sequence context
        ref_sequence, center_pos = self.get_sequence_context(
            reference_genome, variant.chromosome, variant.position, organism
        )
        
        # Create variant sequence
        alt_sequence = self.create_variant_sequence(
            ref_sequence, center_pos, variant.ref_allele, variant.alt_allele
        )
        
        # Make predictions for both sequences
        ref_predictions = self.predict_tracks(ref_sequence, organism)
        alt_predictions = self.predict_tracks(alt_sequence, organism)
        
        # Compute effect scores
        effect_scores = self.compute_variant_effects(ref_predictions, alt_predictions)
        
        # Apply quantile calibration if calibrators are fitted
        calibrated_scores = {}
        for score_name, score_value in effect_scores.items():
            # Extract modality from score name
            modality = score_name.split('_')[0] + '_' + score_name.split('_')[1]
            if modality in self.calibrators and self.calibrators[modality].bins is not None:
                calibrated_scores[f'{score_name}_calibrated'] = float(
                    self.calibrators[modality].transform(np.array([score_value]))[0]
                )
        
        # Combine raw and calibrated scores
        all_scores = {**effect_scores, **calibrated_scores}
        
        # Create metadata
        metadata = {
            'organism': organism,
            'context_length': self.context_length,
            'center_crop': self.center_crop,
            'reference_sequence_length': len(ref_sequence),
            'variant_sequence_length': len(alt_sequence)
        }
        
        return VariantScore(
            variant=variant,
            scores=all_scores,
            predictions={
                'reference': ref_predictions,
                'alternative': alt_predictions
            },
            metadata=metadata
        )
    
    def score_multiple_variants(self, 
                               variants: List[Variant],
                               reference_genome: str,
                               organism: str = 'human') -> List[VariantScore]:
        """
        Score multiple variants efficiently.
        
        Args:
            variants: List of variants to score
            reference_genome: Path to reference genome
            organism: Target organism
            
        Returns:
            List of VariantScore objects
        """
        results = []
        
        for variant in variants:
            try:
                score = self.score_variant(variant, reference_genome, organism)
                results.append(score)
            except Exception as e:
                self.logger.error(f"Failed to score variant {variant.variant_id}: {e}")
                # Create empty score for failed variants
                empty_score = VariantScore(
                    variant=variant,
                    scores={},
                    predictions={},
                    metadata={'error': str(e)}
                )
                results.append(empty_score)
        
        return results
    
    def create_recommended_scorers(self) -> Dict[str, 'VariantScorer']:
        """
        Create the 19 recommended variant scorers from the paper.
        
        Returns:
            Dictionary of configured scorers
        """
        scorers = {}
        
        # Expression scorers
        scorers['RNA-seq'] = self._create_expression_scorer('rna_seq')
        scorers['CAGE'] = self._create_expression_scorer('cage')
        scorers['PRO-cap'] = self._create_expression_scorer('pro_cap')
        
        # Chromatin accessibility scorers
        scorers['ATAC-seq'] = self._create_accessibility_scorer('atac_seq')
        scorers['DNase-seq'] = self._create_accessibility_scorer('dnase_seq')
        
        # Histone modification scorers
        for histone_mark in ['H3K4me3', 'H3K27ac', 'H3K4me1', 'H3K36me3']:
            scorers[f'ChIP-{histone_mark}'] = self._create_histone_scorer(histone_mark)
        
        # Transcription factor binding scorers
        scorers['TF-binding'] = self._create_tf_binding_scorer()
        
        # 3D genome scorers
        scorers['Hi-C'] = self._create_contact_scorer('hi_c')
        scorers['Micro-C'] = self._create_contact_scorer('micro_c')
        
        # Splicing scorers
        scorers['Splice-sites'] = self._create_splicing_scorer('sites')
        scorers['Splice-usage'] = self._create_splicing_scorer('usage')
        scorers['Splice-junctions'] = self._create_splicing_scorer('junctions')
        
        # Multi-modal scorers
        scorers['Multi-modal-expression'] = self._create_multimodal_scorer(['rna_seq', 'cage', 'pro_cap'])
        scorers['Multi-modal-chromatin'] = self._create_multimodal_scorer(['atac_seq', 'dnase_seq', 'chip_histone'])
        scorers['Multi-modal-all'] = self._create_multimodal_scorer(['all'])
        
        return scorers
    
    def _create_expression_scorer(self, modality: str) -> 'VariantScorer':
        """Create specialized scorer for expression data."""
        config = self.config.copy()
        config['focus_modality'] = modality
        config['center_crop'] = 131072  # 128kb for expression
        return VariantScorer(self.model, self.model_params, config)
    
    def _create_accessibility_scorer(self, modality: str) -> 'VariantScorer':
        """Create specialized scorer for chromatin accessibility."""
        config = self.config.copy()
        config['focus_modality'] = modality
        config['center_crop'] = 32768  # 32kb for accessibility peaks
        return VariantScorer(self.model, self.model_params, config)
    
    def _create_histone_scorer(self, histone_mark: str) -> 'VariantScorer':
        """Create specialized scorer for histone modifications."""
        config = self.config.copy()
        config['focus_modality'] = 'chip_histone'
        config['histone_mark'] = histone_mark
        config['center_crop'] = 65536  # 64kb for histone marks
        return VariantScorer(self.model, self.model_params, config)
    
    def _create_tf_binding_scorer(self) -> 'VariantScorer':
        """Create specialized scorer for transcription factor binding."""
        config = self.config.copy()
        config['focus_modality'] = 'chip_tf'
        config['center_crop'] = 16384  # 16kb for TF binding sites
        return VariantScorer(self.model, self.model_params, config)
    
    def _create_contact_scorer(self, contact_type: str) -> 'VariantScorer':
        """Create specialized scorer for 3D genome contacts."""
        config = self.config.copy()
        config['focus_modality'] = 'contact_maps'
        config['contact_type'] = contact_type
        config['center_crop'] = 262144  # 256kb for contact maps
        return VariantScorer(self.model, self.model_params, config)
    
    def _create_splicing_scorer(self, splice_type: str) -> 'VariantScorer':
        """Create specialized scorer for splicing predictions."""
        config = self.config.copy()
        config['focus_modality'] = f'splice_{splice_type}'
        config['center_crop'] = 8192  # 8kb for splicing sites
        return VariantScorer(self.model, self.model_params, config)
    
    def _create_multimodal_scorer(self, modalities: List[str]) -> 'VariantScorer':
        """Create multi-modal scorer combining multiple modalities."""
        config = self.config.copy()
        config['focus_modalities'] = modalities
        config['center_crop'] = 131072  # 128kb for multi-modal
        return VariantScorer(self.model, self.model_params, config)


# Utility functions for variant processing
def load_variants_from_vcf(vcf_file: str, 
                          chromosome: Optional[str] = None,
                          start: Optional[int] = None,
                          end: Optional[int] = None) -> List[Variant]:
    """
    Load variants from VCF file.
    
    Args:
        vcf_file: Path to VCF file
        chromosome: Filter by chromosome
        start: Start position filter
        end: End position filter
        
    Returns:
        List of Variant objects
    """
    import pysam
    
    variants = []
    
    with pysam.VariantFile(vcf_file) as vcf:
        if chromosome:
            region = f"{chromosome}:{start}-{end}" if start and end else chromosome
            records = vcf.fetch(region)
        else:
            records = vcf.fetch()
        
        for record in records:
            variant = Variant(
                chromosome=record.chrom,
                position=record.pos - 1,  # Convert to 0-based
                ref_allele=record.ref,
                alt_allele=str(record.alts[0]),  # Take first alternative
                variant_id=record.id
            )
            variants.append(variant)
    
    return variants


def export_variant_scores(variant_scores: List[VariantScore], 
                         output_file: str,
                         format: str = 'tsv'):
    """
    Export variant scores to file.
    
    Args:
        variant_scores: List of VariantScore objects
        output_file: Output file path
        format: Output format ('tsv', 'csv', 'json')
    """
    if format in ['tsv', 'csv']:
        # Create DataFrame
        rows = []
        for vs in variant_scores:
            row = {
                'variant_id': vs.variant.variant_id,
                'chromosome': vs.variant.chromosome,
                'position': vs.variant.position,
                'ref_allele': vs.variant.ref_allele,
                'alt_allele': vs.variant.alt_allele,
            }
            row.update(vs.scores)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if format == 'tsv':
            df.to_csv(output_file, sep='\t', index=False)
        else:
            df.to_csv(output_file, index=False)
            
    elif format == 'json':
        import json
        
        data = []
        for vs in variant_scores:
            data.append({
                'variant': {
                    'variant_id': vs.variant.variant_id,
                    'chromosome': vs.variant.chromosome,
                    'position': vs.variant.position,
                    'ref_allele': vs.variant.ref_allele,
                    'alt_allele': vs.variant.alt_allele,
                },
                'scores': vs.scores,
                'metadata': vs.metadata
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)