import torch as th
from rreal.utils.utils import build_mlp_net, Parallel
from dataclasses import dataclass, field, fields
from typing import Literal
from adarl.utils.dbg import ggLog
from adarl.utils.dbg.dbg_checks import dbg_check
from rreal.utils.FixedAdamW import AdamW
import copy

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
    learning_rate : float = 1e-3
    ensemble_size : int = 3
    th_device : th.device = th.device("cuda")
    dict_obs_image_key : str | int = "image"
    dict_obs_vector_key : str | int = "vector"
    use_torch_compile : bool = False

@dataclass
class RNDScalerHyperparams:
    """Hyperparameters for :class:`NoveltyScaler`.

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
        Number of reward channels; the statistics are tracked per channel
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

@dataclass
class RNDHyperparams:
    estimator_hyperparams : RNDEstimatorHyperparams = field(default_factory=RNDEstimatorHyperparams)
    scaler_hyperparams : RNDScalerHyperparams = field(default_factory=RNDScalerHyperparams)


def _shallow_copy_dataclass(dc):
    return type(dc)(**{field.name: getattr(dc, field.name) for field in fields(dc)})

class RNDNoveltyEstimator(th.nn.Module):

    def __init__(self,  hyperparams : RNDEstimatorHyperparams):
        super().__init__()
        self._hyperparams = _shallow_copy_dataclass(hyperparams)
        
        self.build_models()
        if self._hyperparams.use_torch_compile:
            self._compute_loss = th.compile(self._compute_loss)
            self._optimizer_step = th.compile(self._optimizer_step)

    def _build_net(self):
        if self._hyperparams.img_encoding_size == 0:
            return build_mlp_net(arch = self._hyperparams.vec_encoder_arch, 
                                input_size = self._hyperparams.vec_input_size,
                                output_size = self._hyperparams.feature_size,
                                ensemble_size=self._hyperparams.ensemble_size,
                                last_activation_class=th.nn.Tanh,
                                return_ensemble_mean=False,
                                hidden_activations=th.nn.LeakyReLU,
                                return_ensemble_std=False).to(device=self._hyperparams.th_device)
        else:
            from rreal.nets.Mixed_encoder import DictMixedEncoder
            def build_mixed_encoder():
                return DictMixedEncoder(image_channels = self._hyperparams.img_input_size_chw[0],
                                        image_width = self._hyperparams.img_input_size_chw[1],
                                        image_height = self._hyperparams.img_input_size_chw[2],
                                        img_ensemble_size = 1,
                                        img_encoding_size = self._hyperparams.img_encoding_size,
                                        backbone = self._hyperparams.img_encoder_arch,
                                        checkDimensions = False,
                                        torchDevice = self._hyperparams.th_device,
                                        use_coord_conv = True,
                                        dropout_prob = 0,
                                        vec_encoder_arch = self._hyperparams.vec_encoder_arch,
                                        vec_part_size = self._hyperparams.vec_input_size,
                                        vec_encoding_size = self._hyperparams.vec_encoding_size,
                                        vec_ensemble_size = 1,
                                        output_size = self._hyperparams.feature_size,
                                        combiner_arch = self._hyperparams.combiner_arch,
                                        encoders_activation = th.nn.LeakyReLU,
                                        use_batchnorm = False,
                                        use_weightnorm = False,
                                        image_dict_key=self._hyperparams.dict_obs_image_key,
                                        vector_dict_key=self._hyperparams.dict_obs_vector_key)
            mixed_encoder = Parallel([build_mixed_encoder() for _ in range(self._hyperparams.ensemble_size)],
                                     return_mean=False).to(device=self._hyperparams.th_device)
            return mixed_encoder

    def build_models(self):
        self._target_net = self._build_net()
        for p in self._target_net.parameters(): # apparently requires_grad_ doesn't work on scriptmodules
            p.requires_grad_(False)
        self._predictor_net = self._build_net()

        self._optimizer = AdamW(self._predictor_net.parameters(), lr = self._hyperparams.learning_rate)
        self._optimizer.zero_grad()

    def _compute_error(self, dict_obs_batch : dict[str|int,th.Tensor] | None = None, vector_obs_batch : th.Tensor | None = None, img_obs_batch : th.Tensor | None = None) -> th.Tensor:
        """ Returns a signed novelty estimate, where the sign is determined by whether the predictor underestimates or overestimates the target features.
            This can be useful when using novelty estimates as loss weights, as it allows to upweight or downweight samples.

        Parameters
        ----------
        dict_obs_batch : dict[str|int,th.Tensor] | None, optional
            _description_, by default None
        vector_obs_batch : th.Tensor | None, optional
            _description_, by default None
        img_obs_batch : th.Tensor | None, optional
            _description_, by default None

        Returns
        -------
        th.Tensor
            Signed novelty estimates for each sample in the batch.
        """
        if self._hyperparams.img_encoding_size == 0:
            if isinstance(dict_obs_batch, dict):
                vector_obs_batch = dict_obs_batch[self._hyperparams.dict_obs_vector_key]
            with th.no_grad():
                target_features = self._target_net(vector_obs_batch)
            predicted_features = self._predictor_net(vector_obs_batch)
        else:
            if not isinstance(dict_obs_batch, dict):
                if vector_obs_batch is None or img_obs_batch is None:
                    raise ValueError("If dict_obs_batch is not provided, both vector_obs_batch and img_obs_batch must be provided.")
                dict_obs_batch = {  self._hyperparams.dict_obs_image_key : img_obs_batch,
                                    self._hyperparams.dict_obs_vector_key : vector_obs_batch}
            # ggLog.info(f"RNDNoveltyEstimator.forward(): dict_obs_batch[{self._hyperparams.dict_obs_image_key}].size() = {dict_obs_batch[self._hyperparams.dict_obs_image_key].size()}")
            # ggLog.info(f"RNDNoveltyEstimator.forward(): dict_obs_batch[{self._hyperparams.dict_obs_vector_key}].size() = {dict_obs_batch[self._hyperparams.dict_obs_vector_key].size()}")
            with th.no_grad():
                target_features = self._target_net(dict_obs_batch)
            predicted_features = self._predictor_net(dict_obs_batch)
        return target_features - predicted_features

    def forward(self, dict_obs_batch : dict[str|int,th.Tensor] | None = None, vector_obs_batch : th.Tensor | None = None, img_obs_batch : th.Tensor | None = None) -> th.Tensor:
        error = self._compute_error(dict_obs_batch=dict_obs_batch, vector_obs_batch=vector_obs_batch, img_obs_batch=img_obs_batch)
        # we now have two [batch_size, ensemble_size, feature_size] tensors
        # we do the mean across both feature_size and ensemble_size.
        #     as the diffs are squared ensembles cannot compensate each other
        # we return a [batch_size] tensor. i.e. we return the novelty for each sample
        return th.mean(th.square(error),dim=(1,2)) 

    def _compute_loss(self, dict_obs_batch : dict[str|int,th.Tensor] | None = None, vector_obs_batch : th.Tensor | None = None, img_obs_batch : th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor]:
        square_errors = self(dict_obs_batch=dict_obs_batch, vector_obs_batch=vector_obs_batch, img_obs_batch=img_obs_batch)
        loss = th.mean(square_errors)
        return loss, square_errors

    def _optimizer_step(self):
        self._optimizer.step()

    def train_model(self, dict_obs_batch : dict[str|int,th.Tensor] | None = None, vector_obs_batch : th.Tensor | None = None, img_obs_batch : th.Tensor | None = None):
        # ggLog.info("RNDNoveltyEstimator.train_model(): Training RND predictor")
        # ggLog.info(f"RNDNoveltyEstimator.train_model(): dict_obs_batch[{self._hyperparams.dict_obs_image_key}].size() = {dict_obs_batch[self._hyperparams.dict_obs_image_key].size()}")
        # ggLog.info(f"RNDNoveltyEstimator.train_model(): dict_obs_batch[{self._hyperparams.dict_obs_vector_key}].size() = {dict_obs_batch[self._hyperparams.dict_obs_vector_key].size()}")          
        self.train() # Put module in train mode
        self._optimizer.zero_grad(set_to_none=True)
        loss, square_errors = self._compute_loss(dict_obs_batch=dict_obs_batch, vector_obs_batch=vector_obs_batch, img_obs_batch=img_obs_batch)
        loss.backward()
        with th.no_grad():
            self._optimizer_step()
        return loss.detach(), square_errors.detach()


class NoveltyScaler():
    """ To apply RND estimates to enrich rewards or losses, these estimates need to be rescaled appropriately.
        In the case of rewards, we want to generate an additive bonus that is neither too small to be irrelevant nor
         too large to dominate the reward signal.
        Also, once the novelty becomes uniform across the data distribution the bonus should go to zero or at least become uniform,
        most importantly, it should not amplify noise.
        Similar reasoning applies to loss weighting, we want to upweight novel samples but not too much, and once novelty is uniform we
         want uniform weights.
        To achieve this we:
        - keep running averages of the raw novelty estimates, their square, and their fourth and second residuals
        - we use these to compute a normalized novelty estimate for each sample in a batch
        - we estimated tail-heaviness (e.g. kurtosis) of the novelty distribution to scale down the normalized novelty when it becomes uniform
        - finally we scale the normalized novelty to the reward (or loss) using some heuristics
        Tail-heaviness is used as a proxy for how much novelty or imbalance is left in the distribution, as novelty evens out the distribution
         becomes more Gaussian and less tail-heavy.
        We can estimate tail-heaviness using different metrics
         - Kurtosis
         - Quantile ratios (Moors 1987 - A quantile alternative to kurtosis), 
         - L-kurtosis (Hosking 1990 - L-moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics, 
           Vogel 2023 - When Heavy Tails Disrupt Statistical Inference).
        Normalization and tail-heaviness is computed on exponentially moving averages of the relevant statistics.
    """
    def __init__(self,  hyperparams: RNDScalerHyperparams):
        """See :class:`RNDScalerHyperparams` for the meaning of each hyperparameter."""
        self._n_updates = 0
        self._n_reward_updates = 0
        self._hp = copy.deepcopy(hyperparams)
        self._stats_initialized = False
        self._avg_novelty = th.as_tensor(0.0, device=hyperparams.th_device)
        self._avg_novelty_m2 = th.as_tensor(0.0, device=hyperparams.th_device) # 2nd central moment, about the running mean
        self._avg_novelty_m4 = th.as_tensor(0.0, device=hyperparams.th_device) # 4th central moment, about the running mean
        self._current_kurtosis = th.as_tensor(float("nan"), device=hyperparams.th_device)

        self._avg_raw_reward : th.Tensor = th.zeros(size=(hyperparams.rewards_num,), device=hyperparams.th_device)
        self._avg_raw_reward_var = th.zeros(size=(hyperparams.rewards_num,), device=hyperparams.th_device)
        self._raw_reward_means_var = th.zeros(size=(hyperparams.rewards_num,), device=hyperparams.th_device)
        self._current_reward_variance = th.zeros(size=(hyperparams.rewards_num,), device=hyperparams.th_device)

        # HYPERPARAMETERS
        self._reward_bonus_weight = th.as_tensor(hyperparams.reward_bonus_weight, device=hyperparams.th_device) # weight of the novelty-based reward bonus
        self._epsilon = 1e-14 # to avoid numerical issues, carefule here, don't set it too big, losses easily get close to 1e-8


    def process_bonuses(self, raw_bonus_batch : th.Tensor, raw_reward_batch : th.Tensor,
                              return_avg_raw_exp_bonus : th.Tensor | None,
                              return_avg_proc_exp_bonus : th.Tensor | None,
                              return_all_proc_exp_bonus : th.Tensor | None,
                              return_all_norm_exp_bonus : th.Tensor | None,
                              return_all_raw_exp_bonus : th.Tensor | None):
        extra_returns = [  return_avg_raw_exp_bonus,
                            return_avg_proc_exp_bonus,
                            return_all_proc_exp_bonus,
                            return_all_norm_exp_bonus,
                            return_all_raw_exp_bonus] 
        if all([ret is None for ret in extra_returns]):
            extra_returns = None
        elif any([ret is None for ret in extra_returns]):
            for ret in extra_returns:
                if ret is None:
                    raise ValueError("If any of the extra returns is requested, all must be requested.")
        return self.novelty_to_reward_bonuses(raw_bonus_batch, raw_reward_batch,
                                              extra_returns = extra_returns #type: ignore
                                              )
        
    def update_stats(self,  raw_novelty_batch : th.Tensor, 
                            raw_reward_batch : th.Tensor | None):
        """Update the running statistics with a new batch.

        The exponential moving averages use a step-dependent alpha: while fewer than
        1/(1-avg_alpha) updates have been seen, alpha is 1-1/t, which makes each average an
        exact arithmetic mean over every batch seen so far. Afterwards alpha settles at
        avg_alpha and the averages become the usual exponential window. This gives the
        minimum-variance estimate early on, when the bonuses are already being applied, and
        seeds the accumulators exactly on the first update (where alpha is zero).
        """
        with th.no_grad():
            if raw_reward_batch is not None:
                self._n_reward_updates += 1
                a = min(self._hp.avg_alpha, 1.0 - 1.0/self._n_reward_updates)
                raw_reward_mean = th.mean(raw_reward_batch, dim=0)
                raw_reward_batch_var = ((raw_reward_batch - raw_reward_mean)**2).mean(dim=0)
                if self._n_reward_updates > 1:
                    # Drift of the batch mean with respect to the running mean. There is no
                    # running mean to drift from on the first update.
                    raw_reward_mean_drift = (raw_reward_mean - self._avg_raw_reward)**2
                else:
                    raw_reward_mean_drift = th.zeros_like(raw_reward_batch_var)
                # mean of the batch means (i.e. the mean)
                self._avg_raw_reward.copy_(a*self._avg_raw_reward + (1-a)*raw_reward_mean)
                # mean of the batch variances (not the variance)
                self._avg_raw_reward_var.copy_(a*self._avg_raw_reward_var + (1-a)*raw_reward_batch_var)
                # mean of the batch mean drifts (i.e. the variance of the means)
                self._raw_reward_means_var.copy_(a*self._raw_reward_means_var + (1-a)*raw_reward_mean_drift)
                # Law of total variance
                self._current_reward_variance = self._avg_raw_reward_var + self._raw_reward_means_var

            self._n_updates += 1
            a = min(self._hp.avg_alpha, 1.0 - 1.0/self._n_updates)
            novelty_batch_mean = th.mean(raw_novelty_batch)
            # Central moments are taken about the *running* mean rather than the batch mean, so
            # that they cover the drift of the batch means and not just the within-batch spread.
            # mean((x-c)^2) expands to the law of total variance, and mean((x-c)^4) to its
            # equivalent for the fourth moment, so no cross terms need to be tracked separately.
            # On the first update there is no running mean to center on yet.
            center = self._avg_novelty if self._n_updates > 1 else novelty_batch_mean
            novelty_batch_m2 = th.mean((raw_novelty_batch - center)**2.0)
            novelty_batch_m4 = th.mean((raw_novelty_batch - center)**4.0)
            self._avg_novelty.copy_(a*self._avg_novelty + (1-a)*novelty_batch_mean)
            self._avg_novelty_m2.copy_(a*self._avg_novelty_m2 + (1-a)*novelty_batch_m2)
            self._avg_novelty_m4.copy_(a*self._avg_novelty_m4 + (1-a)*novelty_batch_m4)
            self._current_kurtosis = self._avg_novelty_m4/th.square(self._avg_novelty_m2)
            self._stats_initialized = True

    def current_kurtosis_estimate(self) -> th.Tensor:
        return self._current_kurtosis
    
    def current_avg_novelty(self) -> th.Tensor:
        return self._avg_novelty

    def novelty_to_reward_bonuses(self,     raw_novelty_batch : th.Tensor, 
                                            raw_reward_batch : th.Tensor,
                                            extra_returns : list[th.Tensor] | None = None,
                                            update_stats : bool = True):
        """
        This function converts raw novelty estimates into reward bonuses, to be added to the environment rewards.
        The bonuses are scaled such that:
         - they are neither too small nor too large compared to the average reward
         - they go to zero as the novelty distribution evens out
         - they are normalized and squashed to avoid outliers having too much impact

        Parameters
        ----------
        raw_novelty_batch : th.Tensor
            _description_
        raw_reward_batch : th.Tensor
            _description_
        extra_returns : list[th.Tensor] | None, optional
            _description_, by default None
        update_stats : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """

        # We do as if the novelty is gaussian, but it is more of a Chi-squared distribution, as it 
        # is the mean of squared errors.
        # it would make more sense to use chi-square quantiles as std thresholds and normalization factors.

        if update_stats:
            self.update_stats(raw_novelty_batch, raw_reward_batch)

        if not self._stats_initialized:
            return raw_reward_batch # if we don't have stats yet, we just return the raw reward batch
        # novelty_mean = th.mean(raw_novelty_batch)
        # novelty_std = th.std(raw_novelty_batch)
        novelty_mean = self._avg_novelty
        novelty_std = th.sqrt(self._avg_novelty_m2) # 2nd central moment about the running mean, so no cancellation
        novelty_kurtosis = self._current_kurtosis

        # squash and normalize the bonuses 
        norm_novelty = th.tanh((raw_novelty_batch - novelty_mean)/(self._hp.reward_novelty_std_squash*novelty_std)) # squash at _novelty_std_squash
        norm_novelty = norm_novelty*self._hp.reward_novelty_std_squash/self._hp.reward_novelty_interest_std_threshold # normalize at interest_threshold*sigma
        # now interest_threshold*sigma is at 1

        # now:
        # Interesting stuff ends up being beyond +1
        # Normal stuff is around zero
        # Boring stuff is below -1
        # The min and max should be at ±_novelty_std_squash/_novelty_interest_std_threshold (i.e. ±2 in the default case)

        # shrink using kurtosis, assuming kurtosis gets low when exploration is done
        norm_novelty = norm_novelty*th.clamp((novelty_kurtosis - self._hp.kurtosis_min)/(self._hp.kurtosis_max-self._hp.kurtosis_min), min=0, max=1)

        # Scale the squashed/normalized bonuses to the reward
        novelty_reward_ratio = self._hp.reward_target_bland_ratio + norm_novelty*self._hp.reward_target_interesting_ratio # This can lead to negative ratios for boring samples!
        novelty_reward_ratio = th.clamp(novelty_reward_ratio, min=-self._hp.reward_max_ratio, max=self._hp.reward_max_ratio) # clamp to avoid excessive bonuses
        bonus_range = th.sqrt(self._current_reward_variance) + self._hp.reward_range_increment # this way even if the reward average is zero we still get an exploration bonus
        novelty_reward = self._reward_bonus_weight*bonus_range*novelty_reward_ratio.unsqueeze(1) # (batch_size, rewards_num)
        
        # scaled_exp_bonus = th.clamp(scaled_exp_bonus, min=0)
        rewards = raw_reward_batch + novelty_reward
        
        if extra_returns is not None:
            extra_returns[0][:] = novelty_mean
            extra_returns[1][:] = th.mean(novelty_reward)
            extra_returns[2][:] = novelty_reward
            extra_returns[3][:] = norm_novelty
            extra_returns[4][:] = raw_novelty_batch
        return rewards
    

    def novelty_to_weights(self,    raw_novelty_batch : th.Tensor, 
                                    update_stats : bool = True):
        """
        This function is meant to be used to reweigh losses, to compensate for data imbalances.
        It follows the reasoning that if a certain class has half the data samples as it should, then it should have double the RND
        prediction error, all else being equal.
        So we want to convert the novelty into a weight multiplier, following this logic we do it by using 'sample_novelty/avg_novelty'
         as the base weight multiplier.
        However, as the novelty distribution evens out, we want to reduce the impact of the novelty. So we use the same kurtosis-based
         scaling as for rewards.

        Parameters
        ----------
        raw_novelty_batch : th.Tensor
            novelty estimates produced by RND (MSE between target and predictor)
        raw_reward_batch : th.Tensor
            _description_
        update_stats : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        dbg_check(lambda: raw_novelty_batch.size()[0] >1,
                  lambda: "NoveltyScaler.novelty_to_weights(): raw_novelty_batch should have more than 1 sample to compute meaningful statistics.")
        if update_stats:
            self.update_stats(raw_novelty_batch, None)
        novelty_mean = th.mean(raw_novelty_batch)
        # novelty_std = th.std(raw_novelty_batch)
        # novelty_mean = self._avg_novelty
        novelty_kurtosis = self._current_kurtosis

        #TODO: Could actually perform the weighting with better metrics, like trying to see if it is actually a proper chi squared with some metric

        # print(f"novelty batch size = {raw_novelty_batch.size()}")
        # print(f"novelty_mean = {novelty_mean.item()}, novelty_kurtosis = {novelty_kurtosis.item()}")
        # print(f"novelty_batch minmax = {raw_novelty_batch.min().item()}-{raw_novelty_batch.max().item()}")
        novelty_weights = raw_novelty_batch / (novelty_mean + self._epsilon) # base weight multiplier
        novelty_weights = th.tanh((novelty_weights - 1.0)/self._hp.novelty_weight_squash)*self._hp.novelty_weight_squash + 1.0 # squash at _novelty_std_squash
        kurtosis_factor = th.clamp((novelty_kurtosis - self._hp.kurtosis_min)/(self._hp.kurtosis_max-self._hp.kurtosis_min), min=0, max=1) # scale from 0 to 1 based on kurtosis
        novelty_weights = 1.0 + (novelty_weights - 1.0)*kurtosis_factor # scale towards 1.0 as kurtosis goes down

        return novelty_weights

from rreal.algorithms.sac import DictTransitionBatch

class SAC_RND_reward_augmentor():
    """ This class is meant to be used as a reward augmentor for SAC, using RND novelty estimates to augment the rewards.
        It uses a RNDNoveltyEstimator and a NoveltyScaler to compute the novelty-based reward bonuses.
        It can be used as a reward augmentor for SAC by passing it to the SAC constructor.
    """
    def __init__(self, rnd_novelty_estimator : RNDNoveltyEstimator, novelty_scaler : NoveltyScaler):
        self._rnd_novelty_estimator = rnd_novelty_estimator
        self._novelty_scaler = novelty_scaler

    def get_augmented_rewards(self,
                            transitions : DictTransitionBatch,
                            critic_enc_obss : th.Tensor,
                            critic_enc_next_obss : th.Tensor,
                            actor_next_enc_obss : th.Tensor) -> th.Tensor:
        """ Computes the novelty-based reward bonuses for the given transitions.

        Parameters
        ----------
        transitions : DictTransitionBatch
            The transitions from which to compute the novelty-based reward bonuses.

        Returns
        -------
        th.Tensor
            The rewards augmented with novelty-based reward bonuses for the given transitions.
        """
        with th.no_grad():
            raw_reward_batch = transitions.rewards
            raw_novelty_batch = self._rnd_novelty_estimator(vector_obs_batch=critic_enc_next_obss)
            augmented_rewards = self._novelty_scaler.novelty_to_reward_bonuses(raw_novelty_batch, raw_reward_batch, update_stats=False)
            return augmented_rewards

    def train_postupdate_hook(self,
                              transitions : DictTransitionBatch,
                              encoded_obss : tuple[th.Tensor | None, th.Tensor, th.Tensor, th.Tensor],
                              losses : tuple[th.Tensor, th.Tensor, th.Tensor]):
        """ This function is meant to be used as a post-update hook for SAC, to train the RND novelty estimator after each SAC update.

        Parameters
        ----------
        transitions : DictTransitionBatch
            The transitions used by the SAC update
        encoded_obss : tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]
            The encoded observations produced by SAC for the transitions
        losses : tuple[th.Tensor, th.Tensor, th.Tensor]
            The losses computed during the SAC update (q_loss, actor_loss, alpha_loss)
        """
        raw_reward_batch = transitions.rewards
        critic_enc_next_obss = encoded_obss[3]
        loss, square_errors = self._rnd_novelty_estimator.train_model(vector_obs_batch=critic_enc_next_obss)
        self._novelty_scaler.update_stats(raw_novelty_batch=square_errors,
                                          raw_reward_batch=raw_reward_batch)
        augmented_rewards = self._novelty_scaler.novelty_to_reward_bonuses(square_errors, raw_reward_batch, update_stats=False)
        logs = {
            "rnd/bonus_ratio_mean" : th.mean(th.abs(augmented_rewards - raw_reward_batch)/(th.abs(raw_reward_batch) + 1e-8)),
            "rnd/bonus_ratio_std" :  th.std(th.abs(augmented_rewards - raw_reward_batch)/(th.abs(raw_reward_batch) + 1e-8)),
            "rnd/loss" : loss,
            "rnd/avg_novelty" : self._novelty_scaler.current_avg_novelty(),
            "rnd/kurtosis" : self._novelty_scaler.current_kurtosis_estimate(),
            "rnd/square_errors" : square_errors
        }
        return logs