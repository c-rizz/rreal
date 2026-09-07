"""Hyperparameter dataclasses for the RND novelty estimator and its scaler.

These live in their own module, separate from :mod:`rreal.utils.RNDNoveltyEstimator`, because
SAC needs them at import time to decide the width of its reward vector, while
RNDNoveltyEstimator imports from rreal.algorithms.sac. Keeping the dataclasses here breaks
that cycle. They are re-exported from RNDNoveltyEstimator, so existing imports keep working.
"""

import torch as th
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RNDEstimatorHyperparams:
    vec_encoder_arch : list[int] | Literal['identity'] = field(default_factory=lambda: [256,256])
    vec_encoding_size : int = 64
    vec_input_size : int = 0
    img_encoder_arch : str = "conv_extrasmall"
    img_encoding_size : int = 0
    img_input_size_chw : tuple[int, int, int] = (0,0,0)
    combiner_arch : list[int] | Literal['identity'] = field(default_factory=list)
    feature_size : int = 64
    learning_rate : float = 1e-4
    ensemble_size : int = 3
    th_device : th.device = th.device("cuda")
    dict_obs_image_key : str | int = "image"
    dict_obs_vector_key : str | int = "vector"
    use_torch_compile : bool = False
    weight_decay : float = 0.001


@dataclass
class RNDScalerHyperparams:
    """Hyperparameters for :class:`rreal.utils.RNDNoveltyEstimator.NoveltyScaler`.

    Attributes
    ----------
    avg_alpha : float
        Alpha for the exponential moving averages of the novelty statistics
    th_device : th.device
        Torch device to use for the internal tensors
    reward_bonus_weight : float
        Final weight of the novelty-based reward bonus
    reward_novelty_interest_std_threshold : float
        Threshold, in novelty standard deviations, above which a sample counts as interesting
    reward_novelty_std_squash : float
        Squash factor for normalized novelty, in standard deviations, to reduce outlier impact
    reward_target_bland_ratio : float
        Target ratio of reward for bland (neither boring nor interesting) samples
    reward_target_interesting_ratio : float
        Target ratio of reward for interesting samples
    reward_max_ratio : float
        The reward ratio is clamped to +-this value, so that outlying novelties cannot
        produce an arbitrarily large bonus or penalty
    reward_range_increment : float
        Increment added to the reward standard deviation when scaling bonuses, so that a
        constant reward, whose standard deviation is zero, still yields a bonus
    kurtosis_min : float
        Kurtosis at or below which the bonuses are zeroed out
    kurtosis_max : float
        Kurtosis at or above which the bonuses are fully applied
    novelty_weight_squash : float
        Squash factor for the novelty weights to reduce outlier impact, used when computing weights
    rewards_num : int
        Number of *environment* reward channels; the reward statistics are tracked per channel.
        When separate_reward_channel is set this does not count the novelty channel, which is
        appended by the scaler and carries no environment reward to track.
    use_gate : bool
        Whether to shrink the bonuses using the tail-heaviness gate
    separate_reward_channel : bool
        If True the bonus is returned as an extra reward channel appended to the environment
        rewards, instead of being added onto every existing channel. The novelty then gets its
        own q value and its own discount, and the task q values keep estimating the task return.
        SAC reads this to decide whether to widen its reward space, so it must be set before the
        agent is built.
    """
    avg_alpha : float = 0.99
    th_device : th.device = th.device("cuda")
    reward_bonus_weight : float = 0.5
    reward_novelty_interest_std_threshold : float = 1.5
    reward_novelty_std_squash : float = 3.0
    reward_target_bland_ratio : float = 0.25
    reward_target_interesting_ratio : float = 0.9
    reward_max_ratio : float = 1.5
    reward_range_increment : float = 0.01
    kurtosis_min : float = 3.0
    kurtosis_max : float = 10.0
    novelty_weight_squash : float = 10.0
    rewards_num : int = 1
    use_gate : bool = False
    separate_reward_channel : bool = True


@dataclass
class SAC_RND_reward_hyperparams:
    estimator_hyperparams : RNDEstimatorHyperparams = field(default_factory=RNDEstimatorHyperparams)
    scaler_hyperparams : RNDScalerHyperparams = field(default_factory=RNDScalerHyperparams)
    use_actor_encoding : bool = False
    """Compute novelty over the actor's encoding of the next observation instead of the critic's.
    Needed when the critic does not encode through the representation novelty should be measured in
    (e.g. a privileged critic, whose encoding is the raw privileged observation)."""
    reward_channel_name : str = "rnd_novelty"
    """Label of the novelty reward channel, used for the q names and the per-channel logging.
    Only used when scaler_hyperparams.separate_reward_channel is set."""
    gamma : float | None = None
    """Discount factor of the novelty reward channel. Exploration usually wants a shorter horizon
    than the task. If None the novelty channel gets the same gamma as the other reward channels.
    Only used when scaler_hyperparams.separate_reward_channel is set."""
