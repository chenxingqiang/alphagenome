"""
Contact Maps Prediction and Processing

Implements contact map prediction, processing, and evaluation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import scipy.ndimage

from ..core.model import AlphaGenomeModel
from ..data import genome


@dataclass
class ContactMapConfig:
    """Configuration for contact map prediction."""
    resolution: int = 2048
    distance_normalization: bool = True
    matrix_balancing: bool = True
    log_transform: bool = True
    epsilon: float = 1e-7


class ContactMapProcessor:
    """Contact map processing following Orca/AlphaGenome methodology."""
    
    def __init__(self, config: ContactMapConfig = None):
        self.config = config or ContactMapConfig()
        self.logger = logging.getLogger(__name__)
    
    def preprocess_contact_map(self, 
                              contact_matrix: np.ndarray,
                              source_resolution: int = 1000) -> np.ndarray:
        """Preprocess contact map following Orca protocol."""
        # Matrix balancing
        if self.config.matrix_balancing:
            contact_matrix = self._balance_matrix(contact_matrix)
        
        # Distance-based normalization
        if self.config.distance_normalization:
            contact_matrix = self._distance_normalize(contact_matrix)
        
        return contact_matrix
    
    def _balance_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Apply matrix balancing to scale values."""
        matrix = matrix.copy().astype(np.float64)
        matrix[np.isnan(matrix)] = 0
        matrix[matrix < 0] = 0
        
        # Iterative balancing
        for _ in range(10):
            row_sums = np.sum(matrix, axis=1)
            row_sums[row_sums == 0] = 1
            matrix = matrix / row_sums[:, None]
            
            col_sums = np.sum(matrix, axis=0)
            col_sums[col_sums == 0] = 1
            matrix = matrix / col_sums[None, :]
            
            total_sum = np.sum(matrix)
            if total_sum > 0:
                matrix = matrix * (matrix.shape[0] * matrix.shape[1]) / total_sum
        
        return matrix
    
    def _distance_normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Apply distance-based normalization."""
        n = matrix.shape[0]
        normalized = np.zeros_like(matrix)
        
        # Compute distance-dependent means
        distance_means = {}
        for distance in range(n):
            if distance == 0:
                values = np.diag(matrix)
            else:
                values = []
                for i in range(n - distance):
                    values.append(matrix[i, i + distance])
                    if i + distance < n:
                        values.append(matrix[i + distance, i])
                values = np.array(values)
            
            non_zero_values = values[values > 0]
            if len(non_zero_values) > 0:
                distance_means[distance] = np.mean(non_zero_values)
            else:
                distance_means[distance] = self.config.epsilon
        
        # Apply normalization
        for i in range(n):
            for j in range(n):
                distance = abs(i - j)
                mean_val = distance_means[distance]
                
                if self.config.log_transform:
                    normalized[i, j] = np.log(matrix[i, j] + self.config.epsilon) - \
                                     np.log(mean_val + self.config.epsilon)
                else:
                    normalized[i, j] = (matrix[i, j] + self.config.epsilon) / \
                                     (mean_val + self.config.epsilon)
        
        return normalized


class ContactMapPredictor:
    """Contact map prediction using AlphaGenome model."""
    
    def __init__(self, 
                 model: AlphaGenomeModel,
                 model_params: Dict,
                 config: ContactMapConfig = None):
        self.model = model
        self.model_params = model_params
        self.config = config or ContactMapConfig()
        self.processor = ContactMapProcessor(config)
        self.logger = logging.getLogger(__name__)
    
    def predict_contact_map(self,
                           sequence: str,
                           organism: str = 'human') -> np.ndarray:
        """Predict contact map for a DNA sequence."""
        # Encode sequence
        from ..core.data_processor import DNAEncoder
        encoder = DNAEncoder()
        encoded_seq = encoder.encode_sequence(sequence)
        
        # Add batch dimension
        batch_seq = jnp.array(encoded_seq)[None, :, :]
        organism_id = jnp.array([0 if organism == 'human' else 1])
        
        # Make prediction
        predictions = self.model.apply(self.model_params, batch_seq, organism_id)
        
        # Extract contact maps
        if 'contact_maps' in predictions:
            contact_maps = predictions['contact_maps'][0]  # Remove batch dimension
            
            # If multiple contact map tracks, average them
            if len(contact_maps.shape) == 3:
                contact_map = np.mean(contact_maps, axis=-1)
            else:
                contact_map = contact_maps
            
            # Ensure symmetry
            contact_map = (contact_map + contact_map.T) / 2
            
            return np.array(contact_map)
        else:
            raise ValueError("Model does not predict contact maps")
    
    def predict_contact_difference(self,
                                  ref_sequence: str,
                                  alt_sequence: str,
                                  organism: str = 'human') -> np.ndarray:
        """Predict contact map difference between reference and alternative sequences."""
        ref_map = self.predict_contact_map(ref_sequence, organism)
        alt_map = self.predict_contact_map(alt_sequence, organism)
        
        return alt_map - ref_map
    
    def evaluate_against_experimental(self,
                                    predicted: np.ndarray,
                                    experimental: np.ndarray,
                                    metric: str = 'pearson') -> float:
        """Evaluate predicted contact map against experimental data."""
        # Ensure same shape
        if predicted.shape != experimental.shape:
            scale_factor = experimental.shape[0] / predicted.shape[0]
            predicted = scipy.ndimage.zoom(predicted, scale_factor, order=1)
        
        # Flatten matrices for correlation calculation
        pred_flat = predicted.flatten()
        exp_flat = experimental.flatten()
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(pred_flat) & np.isfinite(exp_flat)
        pred_flat = pred_flat[valid_mask]
        exp_flat = exp_flat[valid_mask]
        
        if len(pred_flat) == 0:
            return 0.0
        
        if metric == 'pearson':
            correlation = np.corrcoef(pred_flat, exp_flat)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        elif metric == 'mse':
            return float(np.mean((pred_flat - exp_flat) ** 2))
        elif metric == 'mae':
            return float(np.mean(np.abs(pred_flat - exp_flat)))
        else:
            raise ValueError(f"Unknown metric: {metric}")


def load_experimental_contact_map(cooler_file: str,
                                 interval: genome.Interval,
                                 resolution: int = 1000) -> np.ndarray:
    """Load experimental contact map from cooler file."""
    try:
        import cooler
        
        c = cooler.Cooler(cooler_file)
        region = f"{interval.chromosome}:{interval.start}-{interval.end}"
        contact_matrix = c.matrix(balance=True).fetch(region)
        contact_matrix = np.nan_to_num(contact_matrix, nan=0.0)
        
        return contact_matrix.astype(np.float32)
        
    except ImportError:
        raise ImportError("cooler package required for loading experimental contact maps")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load contact map: {e}")
        num_bins = interval.width // resolution
        return np.zeros((num_bins, num_bins), dtype=np.float32)


def compare_contact_maps(predicted: np.ndarray,
                        experimental: np.ndarray,
                        metrics: List[str] = None) -> Dict[str, float]:
    """Compare predicted and experimental contact maps."""
    if metrics is None:
        metrics = ['pearson', 'mse', 'mae']
    
    results = {}
    
    # Ensure same shape
    if predicted.shape != experimental.shape:
        scale_factor = experimental.shape[0] / predicted.shape[0]
        predicted = scipy.ndimage.zoom(predicted, scale_factor, order=1)
    
    # Flatten matrices
    pred_flat = predicted.flatten()
    exp_flat = experimental.flatten()
    
    # Remove invalid values
    valid_mask = np.isfinite(pred_flat) & np.isfinite(exp_flat)
    pred_flat = pred_flat[valid_mask]
    exp_flat = exp_flat[valid_mask]
    
    for metric in metrics:
        try:
            if metric == 'pearson':
                if len(pred_flat) > 1:
                    correlation = np.corrcoef(pred_flat, exp_flat)[0, 1]
                    results[metric] = float(correlation) if not np.isnan(correlation) else 0.0
                else:
                    results[metric] = 0.0
            elif metric == 'mse':
                results[metric] = float(np.mean((pred_flat - exp_flat) ** 2))
            elif metric == 'mae':
                results[metric] = float(np.mean(np.abs(pred_flat - exp_flat)))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to compute {metric}: {e}")
            results[metric] = 0.0
    
    return results 