"""
AlphaGenome Data Processing Module

This module handles all data processing tasks for AlphaGenome:
- DNA sequence encoding and preprocessing
- Multi-modal genomic data loading and normalization
- Training data preparation and augmentation
- Cross-validation data splitting
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Iterator
import pandas as pd
import pysam
import pyBigWig
from pathlib import Path
import logging


class DNAEncoder:
    """DNA sequence encoder for converting sequences to one-hot encoding."""
    
    def __init__(self):
        self.nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.reverse_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
    
    def encode_sequence(self, sequence: str, sequence_length: int = 1048576) -> np.ndarray:
        """
        Convert DNA sequence string to one-hot encoding.
        
        Args:
            sequence: DNA sequence string
            sequence_length: Target sequence length (default 1Mb)
            
        Returns:
            One-hot encoded sequence [L, 4]
        """
        # Pad or truncate to target length
        if len(sequence) < sequence_length:
            sequence = sequence + 'N' * (sequence_length - len(sequence))
        elif len(sequence) > sequence_length:
            sequence = sequence[:sequence_length]
        
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Create one-hot encoding
        encoded = np.zeros((sequence_length, 4), dtype=np.float32)
        
        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.nucleotide_map:
                idx = self.nucleotide_map[nucleotide]
                if idx < 4:  # Skip 'N' nucleotides
                    encoded[i, idx] = 1.0
        
        return encoded
    
    def decode_sequence(self, encoded_seq: np.ndarray) -> str:
        """
        Convert one-hot encoded sequence back to string.
        
        Args:
            encoded_seq: One-hot encoded sequence [L, 4]
            
        Returns:
            DNA sequence string
        """
        sequence_indices = np.argmax(encoded_seq, axis=-1)
        sequence = ''.join([self.reverse_map.get(idx, 'N') for idx in sequence_indices])
        return sequence
    
    def reverse_complement(self, encoded_seq: np.ndarray) -> np.ndarray:
        """
        Generate reverse complement of encoded DNA sequence.
        
        Args:
            encoded_seq: One-hot encoded sequence [L, 4]
            
        Returns:
            Reverse complement encoded sequence [L, 4]
        """
        # Reverse the sequence
        reversed_seq = encoded_seq[::-1]
        
        # Complement: A<->T, C<->G
        complement_map = np.array([3, 2, 1, 0])  # A->T, C->G, G->C, T->A
        
        # Apply complement mapping
        complement_seq = reversed_seq[:, complement_map]
        
        return complement_seq


class GenomicDataLoader:
    """Loader for various genomic data formats."""
    
    def __init__(self, data_config: Dict[str, Any]):
        self.data_config = data_config
        self.logger = logging.getLogger(__name__)
    
    def load_bigwig_tracks(self, 
                          bigwig_files: List[str], 
                          chromosome: str,
                          start: int, 
                          end: int) -> np.ndarray:
        """
        Load genomic tracks from BigWig files.
        
        Args:
            bigwig_files: List of BigWig file paths
            chromosome: Chromosome name (e.g., 'chr1')
            start: Start position (0-based)
            end: End position (exclusive)
            
        Returns:
            Genomic tracks array [L, num_tracks]
        """
        num_tracks = len(bigwig_files)
        length = end - start
        tracks = np.zeros((length, num_tracks), dtype=np.float32)
        
        for i, bigwig_file in enumerate(bigwig_files):
            try:
                with pyBigWig.open(bigwig_file) as bw:
                    # Get values for the interval
                    values = bw.values(chromosome, start, end, numpy=True)
                    
                    # Handle NaN values
                    values = np.nan_to_num(values, nan=0.0)
                    
                    tracks[:, i] = values
                    
            except Exception as e:
                self.logger.warning(f"Failed to load {bigwig_file}: {e}")
                # Fill with zeros if file can't be loaded
                tracks[:, i] = 0.0
        
        return tracks
    
    def load_rna_seq_data(self, 
                         bam_files: List[str],
                         chromosome: str,
                         start: int,
                         end: int) -> Dict[str, np.ndarray]:
        """
        Load RNA-seq data from BAM files for splice junction analysis.
        
        Args:
            bam_files: List of BAM file paths  
            chromosome: Chromosome name
            start: Start position
            end: End position
            
        Returns:
            Dictionary with coverage and splice junction data
        """
        coverage_data = {}
        splice_junctions = {}
        
        for bam_file in bam_files:
            try:
                with pysam.AlignmentFile(bam_file, 'rb') as bam:
                    # Get coverage
                    coverage = np.zeros(end - start, dtype=np.float32)
                    
                    for read in bam.fetch(chromosome, start, end):
                        if read.is_unmapped or read.is_secondary:
                            continue
                            
                        # Add to coverage
                        read_start = max(read.reference_start, start) - start
                        read_end = min(read.reference_end, end) - start
                        
                        if read_start < read_end:
                            coverage[read_start:read_end] += 1
                    
                    coverage_data[bam_file] = coverage
                    
                    # Extract splice junctions
                    junctions = []
                    for read in bam.fetch(chromosome, start, end):
                        if read.is_unmapped or read.cigartuples is None:
                            continue
                            
                        ref_pos = read.reference_start
                        for op, length in read.cigartuples:
                            if op == 3:  # 'N' operation (skipped region/intron)
                                junction_start = ref_pos
                                junction_end = ref_pos + length
                                junctions.append((junction_start, junction_end))
                            if op in [0, 2, 3]:  # M, D, N operations advance reference
                                ref_pos += length
                    
                    splice_junctions[bam_file] = junctions
                    
            except Exception as e:
                self.logger.warning(f"Failed to load RNA-seq data from {bam_file}: {e}")
        
        return {
            'coverage': coverage_data,
            'splice_junctions': splice_junctions
        }
    
    def load_contact_map_data(self, 
                             cooler_file: str,
                             chromosome: str,
                             start: int,
                             end: int,
                             resolution: int = 2048) -> np.ndarray:
        """
        Load Hi-C/Micro-C contact map data from cooler format.
        
        Args:
            cooler_file: Path to cooler file
            chromosome: Chromosome name
            start: Start position
            end: End position
            resolution: Contact map resolution in bp
            
        Returns:
            Contact map matrix [bins, bins]
        """
        try:
            import cooler
            
            # Open cooler file
            c = cooler.Cooler(cooler_file)
            
            # Calculate bins
            num_bins = (end - start) // resolution
            
            # Fetch contact matrix
            region = f"{chromosome}:{start}-{end}"
            contact_matrix = c.matrix(balance=True).fetch(region)
            
            # Handle NaN values and normalize
            contact_matrix = np.nan_to_num(contact_matrix, nan=0.0)
            
            # Resize to target number of bins if needed
            if contact_matrix.shape[0] != num_bins:
                from scipy.ndimage import zoom
                scale_factor = num_bins / contact_matrix.shape[0]
                contact_matrix = zoom(contact_matrix, (scale_factor, scale_factor))
            
            return contact_matrix.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to load contact map from {cooler_file}: {e}")
            num_bins = (end - start) // resolution
            return np.zeros((num_bins, num_bins), dtype=np.float32)


class AlphaGenomeDataProcessor:
    """
    Main data processor for AlphaGenome training and inference.
    
    Handles data loading, preprocessing, augmentation, and batch generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sequence_length = config.get('sequence_length', 1048576)  # 1Mb
        self.organisms = config.get('organisms', ['human', 'mouse'])
        
        # Initialize components
        self.dna_encoder = DNAEncoder()
        self.data_loader = GenomicDataLoader(config)
        self.logger = logging.getLogger(__name__)
        
        # Data paths
        self.data_paths = self._setup_data_paths()
        
        # Cross-validation folds (following Borzoi splits)
        self.cv_folds = self._setup_cv_folds()
    
    def _setup_data_paths(self) -> Dict[str, Dict[str, Any]]:
        """Setup data paths for different organisms and modalities."""
        data_paths = {
            'human': {
                'reference_genome': self.config.get('human_reference', 'hg38.fa'),
                'rna_seq': self.config.get('human_rna_seq_files', []),
                'atac_seq': self.config.get('human_atac_files', []),
                'dnase_seq': self.config.get('human_dnase_files', []),
                'chip_seq': self.config.get('human_chip_files', []),
                'contact_maps': self.config.get('human_contact_files', [])
            },
            'mouse': {
                'reference_genome': self.config.get('mouse_reference', 'mm10.fa'),
                'rna_seq': self.config.get('mouse_rna_seq_files', []),
                'atac_seq': self.config.get('mouse_atac_files', []),
                'dnase_seq': self.config.get('mouse_dnase_files', []),
                'chip_seq': self.config.get('mouse_chip_files', []),
                'contact_maps': self.config.get('mouse_contact_files', [])
            }
        }
        return data_paths
    
    def _setup_cv_folds(self) -> Dict[str, List[str]]:
        """Setup cross-validation chromosome splits (following Borzoi)."""
        human_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX']
        mouse_chroms = [f'chr{i}' for i in range(1, 20)] + ['chrX']
        
        # 4-fold CV splits
        folds = {}
        for organism, chroms in [('human', human_chroms), ('mouse', mouse_chroms)]:
            num_folds = 4
            chrom_per_fold = len(chroms) // num_folds
            
            for fold_idx in range(num_folds):
                start_idx = fold_idx * chrom_per_fold
                end_idx = start_idx + chrom_per_fold
                if fold_idx == num_folds - 1:  # Last fold gets remaining chromosomes
                    end_idx = len(chroms)
                
                fold_name = f'{organism}_fold_{fold_idx}'
                folds[fold_name] = chroms[start_idx:end_idx]
        
        return folds
    
    def get_training_intervals(self, 
                              organism: str, 
                              fold: int,
                              split: str = 'train') -> List[Tuple[str, int, int]]:
        """
        Get genomic intervals for training/validation/test.
        
        Args:
            organism: 'human' or 'mouse'
            fold: Cross-validation fold (0-3)
            split: 'train', 'valid', or 'test'
            
        Returns:
            List of (chromosome, start, end) tuples
        """
        # Get chromosomes for this fold
        fold_name = f'{organism}_fold_{fold}'
        if split == 'train':
            # Use all folds except this one for training
            train_chroms = []
            for f in range(4):
                if f != fold:
                    train_chroms.extend(self.cv_folds[f'{organism}_fold_{f}'])
            chroms = train_chroms
        else:
            # Use this fold for validation/test
            chroms = self.cv_folds[fold_name]
        
        # Generate intervals within chromosomes
        intervals = []
        for chrom in chroms:
            # Get chromosome length (simplified - would need actual reference)
            chrom_length = self._get_chromosome_length(organism, chrom)
            
            # Generate non-overlapping 1Mb intervals
            step_size = self.sequence_length // 2  # 50% overlap
            for start in range(0, chrom_length - self.sequence_length, step_size):
                end = start + self.sequence_length
                intervals.append((chrom, start, end))
        
        return intervals
    
    def _get_chromosome_length(self, organism: str, chromosome: str) -> int:
        """Get chromosome length from reference genome."""
        # Simplified - would normally query reference genome
        # Using approximate human chromosome lengths
        human_chr_lengths = {
            'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559,
            'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
            'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
            'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
            'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
            'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
            'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
            'chr22': 50818468, 'chrX': 156040895
        }
        
        if organism == 'human':
            return human_chr_lengths.get(chromosome, 100000000)
        else:
            # Approximate mouse lengths (similar but shorter)
            return int(human_chr_lengths.get(chromosome, 100000000) * 0.9)
    
    def load_training_example(self, 
                             organism: str,
                             chromosome: str,
                             start: int,
                             end: int) -> Dict[str, np.ndarray]:
        """
        Load a complete training example for a genomic interval.
        
        Args:
            organism: 'human' or 'mouse' 
            chromosome: Chromosome name
            start: Start position
            end: End position
            
        Returns:
            Dictionary containing all model inputs and targets
        """
        example = {}
        
        # 1. Load DNA sequence
        reference_path = self.data_paths[organism]['reference_genome']
        dna_sequence = self._load_dna_sequence(reference_path, chromosome, start, end)
        example['dna_sequence'] = self.dna_encoder.encode_sequence(dna_sequence)
        example['organism_id'] = 0 if organism == 'human' else 1
        
        # 2. Load RNA-seq data
        rna_files = self.data_paths[organism]['rna_seq']
        if rna_files:
            rna_data = self.data_loader.load_rna_seq_data(rna_files, chromosome, start, end)
            example['rna_seq_coverage'] = self._aggregate_coverage_data(rna_data['coverage'])
            example['splice_junctions'] = self._process_splice_junctions(rna_data['splice_junctions'])
        
        # 3. Load ATAC-seq data
        atac_files = self.data_paths[organism]['atac_seq']
        if atac_files:
            example['atac_seq'] = self.data_loader.load_bigwig_tracks(
                atac_files, chromosome, start, end
            )
        
        # 4. Load DNase-seq data
        dnase_files = self.data_paths[organism]['dnase_seq']
        if dnase_files:
            example['dnase_seq'] = self.data_loader.load_bigwig_tracks(
                dnase_files, chromosome, start, end
            )
        
        # 5. Load ChIP-seq data
        chip_files = self.data_paths[organism]['chip_seq']
        if chip_files:
            example['chip_seq'] = self.data_loader.load_bigwig_tracks(
                chip_files, chromosome, start, end
            )
        
        # 6. Load contact maps
        contact_files = self.data_paths[organism]['contact_maps']
        if contact_files:
            # Load contact maps for this interval
            contact_maps = []
            for contact_file in contact_files:
                contact_map = self.data_loader.load_contact_map_data(
                    contact_file, chromosome, start, end
                )
                contact_maps.append(contact_map)
            
            if contact_maps:
                example['contact_maps'] = np.stack(contact_maps, axis=-1)
        
        return example
    
    def _load_dna_sequence(self, 
                          reference_path: str,
                          chromosome: str, 
                          start: int,
                          end: int) -> str:
        """Load DNA sequence from reference genome."""
        try:
            with pysam.FastaFile(reference_path) as fasta:
                sequence = fasta.fetch(chromosome, start, end)
                return sequence.upper()
        except Exception as e:
            self.logger.warning(f"Failed to load sequence from {reference_path}: {e}")
            # Return random sequence as fallback
            return 'N' * (end - start)
    
    def _aggregate_coverage_data(self, coverage_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate RNA-seq coverage across samples."""
        if not coverage_data:
            return np.zeros((self.sequence_length, 1), dtype=np.float32)
        
        # Stack all coverage tracks
        coverage_arrays = list(coverage_data.values())
        aggregated = np.stack(coverage_arrays, axis=-1)
        
        return aggregated
    
    def _process_splice_junctions(self, 
                                 junction_data: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        """Process splice junction data into matrix format."""
        # Simplified implementation - would need more sophisticated processing
        # for the full splice junction prediction head
        max_junctions = 512  # Following paper's approach
        
        junction_matrix = np.zeros((max_junctions, max_junctions), dtype=np.float32)
        
        # Aggregate junctions across samples
        all_junctions = []
        for junctions in junction_data.values():
            all_junctions.extend(junctions)
        
        # Create junction count matrix (simplified)
        # Real implementation would handle donor/acceptor site mapping
        
        return junction_matrix
    
    def apply_augmentations(self, example: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply data augmentations to training example.
        
        Args:
            example: Training example dictionary
            
        Returns:
            Augmented example
        """
        augmented = example.copy()
        
        # 1. Shift augmentation (±1024 bp)
        if np.random.random() < 0.5:
            shift = np.random.randint(-1024, 1025)
            augmented = self._apply_shift_augmentation(augmented, shift)
        
        # 2. Reverse complement augmentation
        if np.random.random() < 0.5:
            augmented = self._apply_reverse_complement(augmented)
        
        return augmented
    
    def _apply_shift_augmentation(self, 
                                 example: Dict[str, np.ndarray], 
                                 shift: int) -> Dict[str, np.ndarray]:
        """Apply shift augmentation to all tracks."""
        augmented = {}
        
        for key, data in example.items():
            if key in ['dna_sequence', 'rna_seq_coverage', 'atac_seq', 'dnase_seq', 'chip_seq']:
                # Apply shift to sequence data
                if shift > 0:
                    # Shift right - pad left, crop right
                    padded = np.pad(data, [(shift, 0)] + [(0, 0)] * (data.ndim - 1))
                    augmented[key] = padded[:data.shape[0]]
                elif shift < 0:
                    # Shift left - crop left, pad right
                    cropped = data[-shift:]
                    padded = np.pad(cropped, [(0, -shift)] + [(0, 0)] * (data.ndim - 1))
                    augmented[key] = padded
                else:
                    augmented[key] = data
            else:
                # Keep other data unchanged
                augmented[key] = data
        
        return augmented
    
    def _apply_reverse_complement(self, example: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply reverse complement augmentation."""
        augmented = {}
        
        for key, data in example.items():
            if key == 'dna_sequence':
                # Reverse complement DNA sequence
                augmented[key] = self.dna_encoder.reverse_complement(data)
            elif key in ['rna_seq_coverage', 'atac_seq', 'dnase_seq', 'chip_seq']:
                # Reverse other sequence tracks
                augmented[key] = data[::-1]
            elif key == 'contact_maps':
                # Transpose contact maps for reverse complement
                augmented[key] = np.transpose(data, (1, 0, 2))
            else:
                # Keep other data unchanged
                augmented[key] = data
        
        return augmented
    
    def create_data_iterator(self,
                           organism: str,
                           fold: int,
                           split: str = 'train',
                           batch_size: int = 1,
                           shuffle: bool = True) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Create data iterator for training/validation.
        
        Args:
            organism: 'human' or 'mouse'
            fold: Cross-validation fold
            split: 'train', 'valid', or 'test'
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of training examples
        """
        # Get intervals for this split
        intervals = self.get_training_intervals(organism, fold, split)
        
        if shuffle:
            np.random.shuffle(intervals)
        
        batch = []
        for chromosome, start, end in intervals:
            try:
                # Load training example
                example = self.load_training_example(organism, chromosome, start, end)
                
                # Apply augmentations for training
                if split == 'train':
                    example = self.apply_augmentations(example)
                
                batch.append(example)
                
                if len(batch) >= batch_size:
                    # Convert to JAX arrays and yield batch
                    yield self._collate_batch(batch)
                    batch = []
                    
            except Exception as e:
                self.logger.warning(f"Failed to load example {chromosome}:{start}-{end}: {e}")
                continue
        
        # Yield remaining examples
        if batch:
            yield self._collate_batch(batch)
    
    def _collate_batch(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, jnp.ndarray]:
        """Collate list of examples into batch."""
        if not batch:
            return {}
        
        collated = {}
        for key in batch[0].keys():
            # Stack examples along batch dimension
            arrays = [example[key] for example in batch]
            collated[key] = jnp.stack(arrays, axis=0)
        
        return collated


# Utility functions for data processing
def normalize_track_data(tracks: np.ndarray, 
                        method: str = 'rpm',
                        target_sum: float = 1e6) -> np.ndarray:
    """
    Normalize genomic tracks.
    
    Args:
        tracks: Track data [L, num_tracks]
        method: Normalization method ('rpm', 'zscore', 'quantile')
        target_sum: Target sum for RPM normalization
        
    Returns:
        Normalized tracks
    """
    if method == 'rpm':
        # Reads per million normalization
        track_sums = np.sum(tracks, axis=0, keepdims=True)
        track_sums = np.where(track_sums == 0, 1, track_sums)  # Avoid division by zero
        normalized = tracks * (target_sum / track_sums)
        
    elif method == 'zscore':
        # Z-score normalization
        means = np.mean(tracks, axis=0, keepdims=True)
        stds = np.std(tracks, axis=0, keepdims=True)
        stds = np.where(stds == 0, 1, stds)  # Avoid division by zero
        normalized = (tracks - means) / stds
        
    elif method == 'quantile':
        # Quantile normalization
        from scipy.stats import rankdata
        ranked = rankdata(tracks, axis=0, method='average')
        target_quantiles = np.linspace(0, 1, tracks.shape[0])
        normalized = np.interp(ranked, np.arange(1, tracks.shape[0] + 1), target_quantiles)
        
    else:
        normalized = tracks
    
    return normalized.astype(np.float32)