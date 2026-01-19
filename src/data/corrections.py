"""
Correction Generation for HG-DAgger style training.

Provides corrective actions for failed episodes using:
- Oracle policy (LIBERO expert demonstrations)
- Scripted perturbations (add noise to failed actions, record "correction")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .episode_buffer import Episode, FailureCategory
from .failure_mining import FailureAnalysis


@dataclass
class Correction:
    """A single correction action."""
    timestep: int
    original_action: np.ndarray
    corrected_action: np.ndarray
    correction_type: str  # "oracle", "scripted", "human"
    confidence: float = 1.0


@dataclass
class CorrectedEpisode:
    """Episode with corrections applied."""
    original: Episode
    corrections: List[Correction]
    corrected_actions: np.ndarray
    failure_analysis: FailureAnalysis


class CorrectionGenerator(ABC):
    """Base class for correction generators."""
    
    @abstractmethod
    def generate_corrections(
        self,
        episode: Episode,
        analysis: FailureAnalysis,
    ) -> List[Correction]:
        """Generate corrections for a failed episode."""
        pass
    
    def apply_corrections(
        self,
        episode: Episode,
        corrections: List[Correction],
    ) -> np.ndarray:
        """Apply corrections to get corrected action sequence."""
        corrected = episode.actions.copy()
        
        for corr in corrections:
            if 0 <= corr.timestep < len(corrected):
                corrected[corr.timestep] = corr.corrected_action
        
        return corrected


class OracleCorrector(CorrectionGenerator):
    """
    Generate corrections using LIBERO's oracle/expert policy.
    
    The oracle policy is the scripted expert that created the demonstrations.
    We can query it for the "correct" action at any state.
    """
    
    def __init__(
        self,
        env,
        oracle_policy=None,
        noise_std: float = 0.01,
    ):
        """
        Initialize oracle corrector.
        
        Args:
            env: LIBERO environment instance
            oracle_policy: Oracle policy (if None, use env's built-in)
            noise_std: Small noise to add for robustness
        """
        self.env = env
        self.oracle_policy = oracle_policy
        self.noise_std = noise_std
        
    def generate_corrections(
        self,
        episode: Episode,
        analysis: FailureAnalysis,
    ) -> List[Correction]:
        """
        Generate corrections around the failure point.
        
        Args:
            episode: Failed episode
            analysis: Failure analysis with timestep
            
        Returns:
            List of corrections around failure point
        """
        corrections = []
        
        # Get correction window around failure timestep
        failure_t = analysis.timestep
        window_start = max(0, failure_t - 5)
        window_end = min(len(episode.actions), failure_t + 10)
        
        for t in range(window_start, window_end):
            # Get state at timestep t
            image = episode.images[t]
            proprio = episode.proprioceptions[t]
            
            # Query oracle for correct action
            oracle_action = self._get_oracle_action(image, proprio)
            
            if oracle_action is not None:
                # Add small noise for robustness
                oracle_action = oracle_action + np.random.normal(
                    0, self.noise_std, size=oracle_action.shape
                )
                
                corrections.append(Correction(
                    timestep=t,
                    original_action=episode.actions[t],
                    corrected_action=oracle_action,
                    correction_type="oracle",
                    confidence=0.95,
                ))
        
        return corrections
    
    def _get_oracle_action(
        self,
        image: np.ndarray,
        proprio: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Query oracle policy for action."""
        if self.oracle_policy is not None:
            # Use provided oracle
            return self.oracle_policy.get_action(image, proprio)
        
        # Try to use LIBERO's built-in expert
        try:
            if hasattr(self.env, 'get_expert_action'):
                return self.env.get_expert_action()
            elif hasattr(self.env, '_get_optimal_action'):
                return self.env._get_optimal_action()
        except Exception:
            pass
        
        return None


class ScriptedCorrector(CorrectionGenerator):
    """
    Generate corrections using scripted perturbations.
    
    For cases where oracle isn't available, we use simple heuristics:
    - Grasp slip: Close gripper more firmly
    - Reach miss: Move towards target
    - Collision: Move away from obstacle
    """
    
    def __init__(
        self,
        action_noise_std: float = 0.1,
        position_correction_gain: float = 0.5,
        gripper_correction: float = -0.5,  # Negative = close more
    ):
        """
        Initialize scripted corrector.
        
        Args:
            action_noise_std: Noise to add to scripted corrections
            position_correction_gain: Gain for position corrections
            gripper_correction: Gripper adjustment for grasp issues
        """
        self.action_noise_std = action_noise_std
        self.position_correction_gain = position_correction_gain
        self.gripper_correction = gripper_correction
    
    def generate_corrections(
        self,
        episode: Episode,
        analysis: FailureAnalysis,
    ) -> List[Correction]:
        """Generate scripted corrections based on failure category."""
        if analysis.category == FailureCategory.GRASP_SLIP:
            return self._correct_grasp_slip(episode, analysis)
        elif analysis.category == FailureCategory.REACH_MISS:
            return self._correct_reach_miss(episode, analysis)
        elif analysis.category == FailureCategory.COLLISION:
            return self._correct_collision(episode, analysis)
        else:
            return self._default_correction(episode, analysis)
    
    def _correct_grasp_slip(
        self,
        episode: Episode,
        analysis: FailureAnalysis,
    ) -> List[Correction]:
        """Correct grasp slip by closing gripper more firmly."""
        corrections = []
        failure_t = analysis.timestep
        
        # Apply gripper correction around failure point
        for t in range(max(0, failure_t - 3), min(len(episode.actions), failure_t + 3)):
            corrected = episode.actions[t].copy()
            
            # Adjust gripper (last dimension)
            corrected[-1] = np.clip(corrected[-1] + self.gripper_correction, -1, 1)
            
            # Add noise
            corrected[:6] += np.random.normal(0, self.action_noise_std, size=6)
            
            corrections.append(Correction(
                timestep=t,
                original_action=episode.actions[t],
                corrected_action=corrected,
                correction_type="scripted_grasp",
                confidence=0.7,
            ))
        
        return corrections
    
    def _correct_reach_miss(
        self,
        episode: Episode,
        analysis: FailureAnalysis,
    ) -> List[Correction]:
        """Correct reach miss by amplifying movement towards goal."""
        corrections = []
        failure_t = analysis.timestep
        
        for t in range(max(0, failure_t - 5), len(episode.actions)):
            original = episode.actions[t]
            corrected = original.copy()
            
            # Amplify position actions
            corrected[:3] = original[:3] * (1 + self.position_correction_gain)
            
            # Add noise
            corrected[:6] += np.random.normal(0, self.action_noise_std, size=6)
            
            corrections.append(Correction(
                timestep=t,
                original_action=original,
                corrected_action=corrected,
                correction_type="scripted_reach",
                confidence=0.6,
            ))
        
        return corrections
    
    def _correct_collision(
        self,
        episode: Episode,
        analysis: FailureAnalysis,
    ) -> List[Correction]:
        """Correct collision by moving away from obstacle."""
        corrections = []
        failure_t = analysis.timestep
        
        # Reverse direction briefly, then continue
        for t in range(max(0, failure_t), min(len(episode.actions), failure_t + 5)):
            original = episode.actions[t]
            corrected = original.copy()
            
            # Reverse position actions
            corrected[:3] = -original[:3] * 0.5
            
            corrections.append(Correction(
                timestep=t,
                original_action=original,
                corrected_action=corrected,
                correction_type="scripted_collision",
                confidence=0.7,
            ))
        
        return corrections
    
    def _default_correction(
        self,
        episode: Episode,
        analysis: FailureAnalysis,
    ) -> List[Correction]:
        """Default correction: add noise around failure point."""
        corrections = []
        failure_t = analysis.timestep
        
        for t in range(max(0, failure_t - 3), min(len(episode.actions), failure_t + 5)):
            original = episode.actions[t]
            corrected = original + np.random.normal(0, self.action_noise_std, size=original.shape)
            
            corrections.append(Correction(
                timestep=t,
                original_action=original,
                corrected_action=corrected,
                correction_type="scripted_default",
                confidence=0.5,
            ))
        
        return corrections


class HybridCorrector(CorrectionGenerator):
    """
    Combines oracle and scripted corrections.
    
    Uses oracle when available, falls back to scripted.
    """
    
    def __init__(
        self,
        env=None,
        oracle_policy=None,
        **kwargs,
    ):
        """Initialize hybrid corrector."""
        self.oracle = OracleCorrector(env, oracle_policy) if env else None
        self.scripted = ScriptedCorrector(**kwargs)
    
    def generate_corrections(
        self,
        episode: Episode,
        analysis: FailureAnalysis,
    ) -> List[Correction]:
        """Generate corrections, preferring oracle."""
        if self.oracle:
            try:
                corrections = self.oracle.generate_corrections(episode, analysis)
                if corrections:
                    return corrections
            except Exception:
                pass
        
        # Fall back to scripted
        return self.scripted.generate_corrections(episode, analysis)


def create_corrected_dataset(
    episodes: List[Episode],
    analyses: List[FailureAnalysis],
    corrector: CorrectionGenerator,
) -> List[CorrectedEpisode]:
    """
    Create a dataset of corrected episodes for training.
    
    Args:
        episodes: Failed episodes
        analyses: Corresponding failure analyses
        corrector: Correction generator
        
    Returns:
        List of corrected episodes
    """
    corrected_dataset = []
    
    for ep, analysis in zip(episodes, analyses):
        if not ep.success:
            corrections = corrector.generate_corrections(ep, analysis)
            corrected_actions = corrector.apply_corrections(ep, corrections)
            
            corrected_dataset.append(CorrectedEpisode(
                original=ep,
                corrections=corrections,
                corrected_actions=corrected_actions,
                failure_analysis=analysis,
            ))
    
    return corrected_dataset
