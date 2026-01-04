"""
Similarity metrics for ground-truth comparison and reward calculation.

This module provides configurable similarity metrics for comparing generated
sequences against ground-truth references, supporting both feature-space and
attribute-level comparisons.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.spatial.distance import cosine, euclidean

from src.utils.logging.logging_manager import get_logger

logger = get_logger(__name__)


class SimilarityCalculator:
    """
    Configurable similarity calculator supporting multiple metrics and modes.
    
    Supports:
    - Feature-space similarity (cosine, euclidean)
    - Attribute-level similarity (key, tempo, rhythmic density)
    - Token equality fallback
    """
    
    SUPPORTED_MODES = ["cosine", "euclidean", "attribute", "token_equality"]
    
    def __init__(self, mode: str = "cosine"):
        """
        Initialize similarity calculator.
        
        Args:
            mode: Similarity calculation mode ('cosine', 'euclidean', 'attribute', 'token_equality')
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported similarity mode: {mode}. Supported modes: {self.SUPPORTED_MODES}")
        
        self.mode = mode
    
    def calculate_similarity(
        self,
        generated_sequence: Union[List[int], np.ndarray],
        reference_data: Union[List[int], np.ndarray, Dict[str, Any]],
        generated_features: Optional[np.ndarray] = None,
        reference_features: Optional[np.ndarray] = None,
        **kwargs
    ) -> float:
        """
        Calculate similarity between generated sequence and reference.
        
        Args:
            generated_sequence: Generated sequence of cluster IDs or tokens
            reference_data: Reference sequence or metadata dict
            generated_features: Optional feature vector for generated sequence
            reference_features: Optional feature vector for reference
            **kwargs: Additional parameters for specific similarity modes
            
        Returns:
            Similarity score normalized to [0, 1] range
        """
        try:
            if self.mode == "cosine":
                return self._cosine_similarity(generated_features, reference_features, 
                                             generated_sequence, reference_data)
            elif self.mode == "euclidean":
                return self._euclidean_similarity(generated_features, reference_features,
                                                generated_sequence, reference_data)
            elif self.mode == "attribute":
                if isinstance(reference_data, dict):
                    return self._attribute_similarity(generated_sequence, reference_data, **kwargs)
                else:
                    logger.warning("Attribute similarity requires reference_data to be a dict. Falling back to token equality.")
                    return self._token_equality(generated_sequence, reference_data)
            elif self.mode == "token_equality":
                return self._token_equality(generated_sequence, reference_data)
            else:
                logger.warning(f"Unknown similarity mode: {self.mode}. Falling back to token equality.")
                return self._token_equality(generated_sequence, reference_data)
                
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}. Returning 0.0")
            return 0.0
    
    def _cosine_similarity(
        self,
        generated_features: Optional[np.ndarray],
        reference_features: Optional[np.ndarray],
        generated_sequence: Optional[Union[List[int], np.ndarray]] = None,
        reference_data: Optional[Union[List[int], np.ndarray, Dict[str, Any]]] = None
    ) -> float:
        """Calculate cosine similarity between feature vectors."""
        if generated_features is None or reference_features is None:
            logger.info("Features not available, falling back to token equality")
            if generated_sequence is not None and reference_data is not None:
                return self._token_equality(generated_sequence, reference_data)
            return 0.0
        
        # Ensure feature vectors are numpy arrays
        gen_feat = np.asarray(generated_features).flatten()
        ref_feat = np.asarray(reference_features).flatten()
        
        # Handle dimension mismatch by padding or truncating
        min_len = min(len(gen_feat), len(ref_feat))
        if min_len == 0:
            return 0.0
            
        gen_feat = gen_feat[:min_len]
        ref_feat = ref_feat[:min_len]
        
        # Check for zero vectors before computing cosine similarity
        gen_norm = np.linalg.norm(gen_feat)
        ref_norm = np.linalg.norm(ref_feat)
        
        if gen_norm == 0.0 or ref_norm == 0.0:
            logger.info("Zero vector detected, falling back to token equality")
            if generated_sequence is not None and reference_data is not None:
                return self._token_equality(generated_sequence, reference_data)
            return 0.0
        
        # Calculate cosine similarity (scipy returns distance, so we convert to similarity)
        try:
            similarity = 1 - cosine(gen_feat, ref_feat)
            # Check for NaN result
            if np.isnan(similarity):
                logger.info("NaN similarity detected, falling back to token equality")
                if generated_sequence is not None and reference_data is not None:
                    return self._token_equality(generated_sequence, reference_data)
                return 0.0
            # Normalize to [0, 1] range
            return float(max(0.0, min(1.0, (similarity + 1) / 2)))
        except ValueError:
            # Handle other issues
            logger.info("Cosine similarity failed, falling back to token equality")
            if generated_sequence is not None and reference_data is not None:
                return self._token_equality(generated_sequence, reference_data)
            return 0.0
    
    def _euclidean_similarity(
        self,
        generated_features: Optional[np.ndarray],
        reference_features: Optional[np.ndarray],
        generated_sequence: Optional[Union[List[int], np.ndarray]] = None,
        reference_data: Optional[Union[List[int], np.ndarray, Dict[str, Any]]] = None
    ) -> float:
        """Calculate normalized euclidean similarity between feature vectors."""
        if generated_features is None or reference_features is None:
            logger.info("Features not available, falling back to token equality")
            if generated_sequence is not None and reference_data is not None:
                return self._token_equality(generated_sequence, reference_data)
            return 0.0
        
        # Ensure feature vectors are numpy arrays
        gen_feat = np.asarray(generated_features).flatten()
        ref_feat = np.asarray(reference_features).flatten()
        
        # Handle dimension mismatch
        min_len = min(len(gen_feat), len(ref_feat))
        if min_len == 0:
            return 0.0
            
        gen_feat = gen_feat[:min_len]
        ref_feat = ref_feat[:min_len]
        
        try:
            # Calculate euclidean distance
            distance = euclidean(gen_feat, ref_feat)
            # Convert to similarity using exponential decay
            max_distance = np.sqrt(len(gen_feat)) * 2  # Heuristic for normalization
            similarity = np.exp(-distance / max_distance)
            return max(0.0, min(1.0, similarity))
        except ValueError:
            return 0.0
    
    def _attribute_similarity(
        self,
        generated_sequence: Union[List[int], np.ndarray],
        reference_data: Dict[str, Any],
        generated_attributes: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate attribute-level similarity (key, tempo, rhythmic density, etc.).
        
        This is a placeholder implementation that can be extended with actual
        musical attribute extraction and comparison logic.
        """
        if not isinstance(reference_data, dict):
            logger.warning("Reference data must be a dictionary for attribute similarity")
            return 0.0
        
        if generated_attributes is None:
            # Placeholder: extract attributes from generated sequence
            generated_attributes = self._extract_attributes_from_sequence(generated_sequence)
        
        # Compare key attributes
        similarities = []
        
        # Key signature similarity (if available)
        if "key" in reference_data and "key" in generated_attributes:
            key_sim = 1.0 if reference_data["key"] == generated_attributes["key"] else 0.0
            similarities.append(key_sim)
        
        # Tempo similarity (if available)
        if "tempo" in reference_data and "tempo" in generated_attributes:
            ref_tempo = float(reference_data["tempo"])
            gen_tempo = float(generated_attributes["tempo"])
            # Normalize tempo difference to [0, 1]
            tempo_diff = abs(ref_tempo - gen_tempo) / max(ref_tempo, gen_tempo, 1.0)
            tempo_sim = max(0.0, 1.0 - tempo_diff)
            similarities.append(tempo_sim)
        
        # Rhythmic density similarity (placeholder)
        if "rhythmic_density" in reference_data and "rhythmic_density" in generated_attributes:
            ref_density = float(reference_data["rhythmic_density"])
            gen_density = float(generated_attributes["rhythmic_density"])
            density_diff = abs(ref_density - gen_density) / max(ref_density, gen_density, 1.0)
            density_sim = max(0.0, 1.0 - density_diff)
            similarities.append(density_sim)
        
        # Return average similarity if any attributes were compared
        if similarities:
            return float(np.mean(similarities))
        else:
            logger.info("No matching attributes found, returning default similarity")
            return 0.5  # Neutral similarity when no attributes can be compared
    
    def _token_equality(
        self,
        generated_sequence: Union[List[int], np.ndarray],
        reference_data: Union[List[int], np.ndarray, Dict[str, Any]]
    ) -> float:
        """Calculate similarity based on token equality."""
        # Handle case where reference_data is a dict (extract sequence if available)
        if isinstance(reference_data, dict):
            if "sequence" in reference_data:
                reference_sequence = reference_data["sequence"]
            else:
                logger.warning("No sequence found in reference data for token equality")
                return 0.0
        else:
            reference_sequence = reference_data
        
        # Convert to lists for comparison
        gen_seq = list(np.asarray(generated_sequence).flatten())
        ref_seq = list(np.asarray(reference_sequence).flatten())
        
        if not gen_seq or not ref_seq:
            return 0.0
        
        # Compare sequences token by token
        min_len = min(len(gen_seq), len(ref_seq))
        matches = sum(1 for i in range(min_len) if gen_seq[i] == ref_seq[i])
        
        # Penalty for different lengths
        max_len = max(len(gen_seq), len(ref_seq))
        similarity = matches / max_len if max_len > 0 else 0.0
        
        return similarity
    
    def _extract_attributes_from_sequence(
        self,
        sequence: Union[List[int], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Extract musical attributes from a sequence of cluster IDs.
        
        This is a placeholder implementation that should be replaced with
        actual musical analysis based on the cluster mappings and MIDI data.
        """
        seq_array = np.asarray(sequence)
        
        # Placeholder attribute extraction
        attributes = {
            "key": "C",  # Default key
            "tempo": 120.0,  # Default tempo
            "rhythmic_density": len(seq_array) / 16.0,  # Simple density measure
            "pitch_range": len(np.unique(seq_array)) if len(seq_array) > 0 else 1,
        }
        
        return attributes


def normalize_reward(reward: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize reward to specified range.
    
    Args:
        reward: Raw reward value
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
        
    Returns:
        Normalized reward in [min_val, max_val] range
    """
    if np.isnan(reward) or np.isinf(reward):
        return min_val
    
    # Clip to reasonable bounds first
    clipped = np.clip(reward, -10.0, 10.0)
    
    # Normalize to [0, 1] using sigmoid-like transformation
    normalized = 1 / (1 + np.exp(-clipped))
    
    # Scale to desired range
    scaled = min_val + (max_val - min_val) * normalized
    
    return float(scaled)


def create_similarity_calculator(config: Dict[str, Any]) -> SimilarityCalculator:
    """
    Factory function to create similarity calculator from configuration.
    
    Args:
        config: Configuration dictionary containing similarity_mode
        
    Returns:
        Configured SimilarityCalculator instance
    """
    similarity_mode = config.get("similarity_mode", "cosine")
    return SimilarityCalculator(mode=similarity_mode)