import torch as th
from rreal.utils.utils import build_mlp_net, Parallel
from dataclasses import dataclass
from typing import Literal
from adarl.utils.dbg import ggLog

class RNDNoveltyEstimator(th.nn.Module):

    @dataclass
    class Hyperparams:
        vec_encoder_arch : list[int] | Literal['identity']
        vec_input_size : int
        vec_encoding_size : int
        img_input_size_chw : tuple[int, int, int]
        img_encoding_size : int
        img_encoder_arch : str
        combiner_arch : list[int]
        feature_size : int
        learning_rate : float
        ensemble_size : int
        th_device : th.device
        dict_obs_image_key : str | int
        dict_obs_vector_key : str | int

    def __init__(self,  vec_encoder_arch : list[int] | Literal['identity'],
                        vec_encoding_size : int,
                        vec_input_size : int,
                        img_encoder_arch : str = "conv_extrasmall",
                        img_encoding_size : int = 0,
                        img_input_size_chw : tuple[int, int, int] = (0,0,0),
                        combiner_arch : list[int] | Literal['identity'] = [],
                        feature_size : int = 64,
                        learning_rate : float = 1e-3,
                        ensemble_size : int = 3,
                        th_device : th.device = th.device("cuda"),
                        dict_obs_image_key : str | int = "image",
                        dict_obs_vector_key : str | int = "vector"):
        super().__init__()
        self._hyperparams = self.Hyperparams(   vec_encoder_arch=vec_encoder_arch,
                                                vec_encoding_size=vec_encoding_size,
                                                vec_input_size=vec_input_size,
                                                img_encoder_arch=img_encoder_arch,
                                                img_encoding_size=img_encoding_size,
                                                img_input_size_chw=img_input_size_chw,
                                                combiner_arch=combiner_arch,
                                                feature_size=feature_size,
                                                learning_rate=learning_rate,
                                                ensemble_size=ensemble_size,
                                                th_device=th_device,
                                                dict_obs_image_key=dict_obs_image_key,
                                                dict_obs_vector_key=dict_obs_vector_key)
        self.build_models()

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

        self._optimizer = th.optim.Adam(self._predictor_net.parameters(), lr = self._hyperparams.learning_rate)
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


    def train_model(self, dict_obs_batch : dict[str|int,th.Tensor] | None = None, vector_obs_batch : th.Tensor | None = None, img_obs_batch : th.Tensor | None = None):
        # ggLog.info("RNDNoveltyEstimator.train_model(): Training RND predictor")
        # ggLog.info(f"RNDNoveltyEstimator.train_model(): dict_obs_batch[{self._hyperparams.dict_obs_image_key}].size() = {dict_obs_batch[self._hyperparams.dict_obs_image_key].size()}")
        # ggLog.info(f"RNDNoveltyEstimator.train_model(): dict_obs_batch[{self._hyperparams.dict_obs_vector_key}].size() = {dict_obs_batch[self._hyperparams.dict_obs_vector_key].size()}")          
        self.train() # Put module in train mode
        self._optimizer.zero_grad(set_to_none=True)
        square_errors = self(dict_obs_batch=dict_obs_batch, vector_obs_batch=vector_obs_batch, img_obs_batch=img_obs_batch)
        loss = th.mean(square_errors)
        loss.backward()
        self._optimizer.step()
        return loss
        

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
    def __init__(self,  avg_alpha : float, 
                        th_device : th.device,
                        reward_bonus_weight : float = 0.5,
                        reward_novelty_interest_std_threshold : float = 1.5,
                        reward_novelty_std_squash : float = 3.0,
                        reward_target_bland_ratio : float = 0.25,
                        reward_target_interesting_ratio : float = 0.9,
                        reward_increment : float = 0.01,
                        kurtosis_min : float = 10.0,
                        kurtosis_max : float = 20.0,
                        novelty_weight_squash : float = 10.0):
        """

        Parameters
        ----------
        avg_alpha : float
            Alpha for the exponential moving averages of the novelty statistics
        bonus_weight : float
            Final weight of the novelty-based reward bonus
        th_device : th.device
            Torch device to use for the internal tensors
        novelty_interest_std_threshold : float, optional
            Threshold for considering novelty as interesting, by default 1.5
        novelty_std_squash : float, optional
            Squash factor for normalized novelty to reduce outlier impact, by default 3.0
        reward_target_bland_ratio : float, optional
            Target ratio of reward for bland (neither boring nor interesting) samples, by default 0.25
        reward_target_interesting_ratio : float, optional
            Target ratio of reward for interesting samples, by default 0.9
        reward_increment : float, optional
            Increment added to the average reward when scaling bonuses, by default 0.01
        kurtosis_min : float, optional
            Minimum kurtosis value for scaling bonuses, by default 10.0
        kurtosis_max : float, optional
            Maximum kurtosis value for scaling bonuses, by default 20.0
        novelty_weight_squash : float, optional
            Squash factor for the novelty weights to reduce outlier impact, used when computing weights, by default 10.0
        """
        self._n_updates = 0
        self._avg_novelty : th.Tensor
        self._avg_novelty_mean_of_square : th.Tensor
        self._avg_novelty_mean_of_fourth_residual : th.Tensor
        self._avg_novelty_mean_of_second_residual : th.Tensor
        self._avg_raw_reward : th.Tensor
        self._current_kurtosis = th.as_tensor(float("nan"), device=th_device)

        # HYPERPARAMETERS
        self._avgs_alpha_th = th.as_tensor(avg_alpha, device=th_device) # stats exponential moving average alpha
        self._reward_bonus_weight = th.as_tensor(reward_bonus_weight, device=th_device) # weight of the novelty-based reward bonus
        self._epsilon = 1e-14 # to avoid numerical issues, carefule here, don't set it too big, losses easily get close to 1e-8
        # Normalization and scaling hyperparameters:
        self._novelty_interest_std_threshold = reward_novelty_interest_std_threshold # We consider 'interesting' novelties that are at this multiple of std in the novelty distribution..
        self._novelty_std_squash = reward_novelty_std_squash # We squash the normalized novelty at this multiple of std (sigma), to reduce the impact of outliers
        # We compute reward bonuses as ratios of the average reward, following these target ratios:
        self._reward_target_bland_ratio = reward_target_bland_ratio # Stuff that is neither boring nor interesting shoud end up accounting for this amount of reward
        self._reward_target_interesting_ratio = reward_target_interesting_ratio # Stuff that is interesting shoud end up accounting for this amount of reward
        self._reward_increment = reward_increment # We add this to the average reward when scaling the bonuses to ensure that even if the average reward is zero we still get some bonus
        # We linearly scale the bonuses between these two kurtosis values:
        self._kurtosis_min = kurtosis_min # when the novelty kurtosis reaches this value the bonuses gets zeroed out
        self._kurtosis_max = kurtosis_max # when the novelty kurtosis is at or above this value the bonuses are fully applied

        # When using novelty to compute loss weights for data imbalance:
        self._novelty_weight_squash = novelty_weight_squash # we squash the novelty-based weights at this value to avoid extreme weights

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
        novelty_batch_mean = th.mean(raw_novelty_batch)
        novelty_batch_mean_of_square = th.mean(th.square(raw_novelty_batch))
        novelty_batch_mean_of_fourth_residual = th.mean(th.pow(raw_novelty_batch - novelty_batch_mean, 4.0))
        novelty_batch_mean_of_second_residual = th.mean(th.pow(raw_novelty_batch - novelty_batch_mean, 2.0))
        if self._n_updates == 0:
            self._avg_novelty = novelty_batch_mean
            self._avg_novelty_mean_of_square = novelty_batch_mean_of_square
            self._avg_novelty_mean_of_fourth_residual = novelty_batch_mean_of_fourth_residual
            self._avg_novelty_mean_of_second_residual = novelty_batch_mean_of_second_residual
            if raw_reward_batch is not None:
                self._avg_raw_reward = th.mean(raw_reward_batch)
        else:
            alpha = self._avgs_alpha_th
            self._avg_novelty =                         alpha * self._avg_novelty +                         (1-alpha)*novelty_batch_mean
            self._avg_novelty_mean_of_square =          alpha * self._avg_novelty_mean_of_square +          (1-alpha)*novelty_batch_mean_of_square
            self._avg_novelty_mean_of_fourth_residual = alpha * self._avg_novelty_mean_of_fourth_residual + (1-alpha)*novelty_batch_mean_of_fourth_residual
            self._avg_novelty_mean_of_second_residual = alpha * self._avg_novelty_mean_of_second_residual + (1-alpha)*novelty_batch_mean_of_second_residual
            if raw_reward_batch is not None:
                self._avg_raw_reward =                      alpha * self._avg_raw_reward +                      (1-alpha)*th.mean(raw_reward_batch)
        self._current_kurtosis = th.mean(self._avg_novelty_mean_of_fourth_residual)/th.square(th.mean(self._avg_novelty_mean_of_second_residual))
        self._n_updates += 1

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
        # novelty_mean = th.mean(raw_novelty_batch)
        # novelty_std = th.std(raw_novelty_batch)
        novelty_mean = self._avg_novelty
        novelty_std = th.sqrt(self._avg_novelty_mean_of_square - th.square(self._avg_novelty)) # maybe use bias correction?
        novelty_kurtosis = self._current_kurtosis

        # squash and normalize the bonuses 
        norm_novelty = th.tanh((raw_novelty_batch - novelty_mean)/(self._novelty_std_squash*novelty_std)) # squash at _novelty_std_squash
        norm_novelty = norm_novelty*self._novelty_std_squash/self._novelty_interest_std_threshold # normalize at interest_threshold*sigma
        # now interest_threshold*sigma is at 1

        # now:
        # Interesting stuff ends up being beyond +1
        # Normal stuff is around zero
        # Boring stuff is below -1
        # The min and max should be at ±_novelty_std_squash/_novelty_interest_std_threshold (i.e. ±2 in the default case)

        # shrink using kurtosis, assuming kurtosis gets low when exploration is done
        norm_novelty = norm_novelty*th.clamp((novelty_kurtosis - self._kurtosis_min)/(self._kurtosis_max-self._kurtosis_min), min=0, max=1)

        # Scale the squashed/normalized bonuses to the reward
        novelty_reward_ratio = self._reward_target_bland_ratio + norm_novelty*self._reward_target_interesting_ratio # This can lead to negative ratios for boring samples!
        inc_avg_reward = self._avg_raw_reward + self._reward_increment # this way even if the reward average is zero we still get an exploration bonus
        novelty_reward = self._reward_bonus_weight*inc_avg_reward*novelty_reward_ratio
        
        # scaled_exp_bonus = th.clamp(scaled_exp_bonus, min=0)
        rewards = raw_reward_batch + novelty_reward.unsqueeze(dim=1)
        
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
        if update_stats:
            self.update_stats(raw_novelty_batch, None)
        novelty_mean = th.mean(raw_novelty_batch)
        # novelty_std = th.std(raw_novelty_batch)
        # novelty_mean = self._avg_novelty
        novelty_kurtosis = self._current_kurtosis

        #TODO: Could actually perform the weighting with better metrics, like trying to see if it is actually a proper chi squared with some metric


        novelty_weights = raw_novelty_batch / (novelty_mean + self._epsilon) # base weight multiplier
        novelty_weights = th.tanh((novelty_weights - 1.0)/self._novelty_weight_squash)*self._novelty_weight_squash + 1.0 # squash at _novelty_std_squash
        kurtosis_factor = th.clamp((novelty_kurtosis - self._kurtosis_min)/(self._kurtosis_max-self._kurtosis_min), min=0, max=1) # scale from 0 to 1 based on kurtosis
        novelty_weights = 1.0 + (novelty_weights - 1.0)*kurtosis_factor # scale towards 1.0 as kurtosis goes down

        return novelty_weights
