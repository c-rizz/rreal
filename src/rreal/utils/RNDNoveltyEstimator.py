import torch as th
from autoencoding_rl.utils import build_mlp_net
from dataclasses import dataclass

class RNDNoveltyEstimator(th.nn.Module):

    @dataclass
    class Hyperparams:
        nn_arch : list[int] = None
        input_size : int = None
        feature_vec_size : int = None
        learning_rate : float = None
        ensemble_size : int = None

    def __init__(self, nn_arch : list[int], input_size : int, feature_vec_size : int, learning_rate):
        super().__init__()
        self._hyperparams = self.Hyperparams()
        self.set_hyperparams(nn_arch, input_size, feature_vec_size, learning_rate, 5)
        self.build_models()

    def set_hyperparams(self, nn_arch : list[int], input_size : int, feature_vec_size : int, learning_rate : float, ensemble_size : int):
        self._hyperparams.nn_arch = nn_arch
        self._hyperparams.input_size = input_size
        self._hyperparams.feature_vec_size = feature_vec_size
        self._hyperparams.learning_rate = learning_rate
        self._hyperparams.ensemble_size = ensemble_size

    def _build_net(self):
        return build_mlp_net(arch = self._hyperparams.nn_arch, 
                            input_size = self._hyperparams.input_size,
                            output_size = self._hyperparams.feature_vec_size,
                            ensemble_size=self._hyperparams.ensemble_size,
                            last_activation_class=th.nn.Tanh,
                            return_ensemble_mean=False,
                            hidden_activations=th.nn.LeakyReLU,
                            return_ensemble_std=False)

    def build_models(self):
        self._target_net = self._build_net()
        for p in self._target_net.parameters(): # apparently requires_grad_ doesn't work on scriptmodules
            p.requires_grad_(False)
        self._predictor_net = self._build_net()

        self._optimizer = th.optim.Adam(self._predictor_net.parameters(), lr = self._hyperparams.learning_rate)
        self._optimizer.zero_grad()

    def forward(self, batch : th.Tensor):
        with th.no_grad():
            target_features = self._target_net(batch)
        predicted_features = self._predictor_net(batch)
        # we now have two [batch_size, ensemble_size, feature_size] tensors
        # we do the mean across both feature_size and ensemble_size.
        #     as the diffs are squared ensembles cannot compensate each other
        # we return a [batch_size] tensor. i.e. we return the novelty for each sample
        return th.mean(th.square(target_features-predicted_features),dim=(1,2)) 


    def train_model(self, batch : th.Tensor):

        self.train() # Put module in train mode
        self._optimizer.zero_grad(set_to_none=True)
        square_errors = self(batch)
        loss = th.mean(square_errors)
        loss.backward()
        self._optimizer.step()
        return loss
        

class NoveltyScaler():
    def __init__(self,  avg_alpha : float, 
                        bonus_weight : float,
                        th_device : th.device):
        self._n_updates = 0
        self._avgs_alpha_th = th.as_tensor(avg_alpha, device=th_device)
        self._bonus_weight = th.as_tensor(bonus_weight, device=th_device)
        self._avg_raw_exploration_bonus : th.Tensor
        self._avg_squared_raw_exploration_bonus : th.Tensor
        self._avg_fourth_raw_eb_residual : th.Tensor
        self._avg_second_raw_eb_residual : th.Tensor
        self._avg_raw_reward : th.Tensor
        
    def process_bonuses(self, raw_bonus_batch : th.Tensor, raw_reward_batch : th.Tensor,
                              return_avg_raw_exp_bonus : th.Tensor | None,
                              return_avg_proc_exp_bonus : th.Tensor | None,
                              return_all_proc_exp_bonus : th.Tensor | None,
                              return_all_norm_exp_bonus : th.Tensor | None,
                              return_all_raw_exp_bonus : th.Tensor | None):
        raw_batch_eb_mean = th.mean(raw_bonus_batch)
        raw_batch_square_eb_mean = th.mean(th.square(raw_bonus_batch))
        raw_batch_fourth_eb_residual_mean = th.mean(th.pow(raw_bonus_batch - raw_batch_eb_mean, 4.0))
        raw_batch_second_eb_residual_mean = th.mean(th.pow(raw_bonus_batch - raw_batch_eb_mean, 2.0))
        
        if self._n_updates == 0:
            self._avg_raw_exploration_bonus = raw_batch_eb_mean
            self._avg_squared_raw_exploration_bonus = raw_batch_square_eb_mean
            self._avg_fourth_raw_eb_residual = raw_batch_fourth_eb_residual_mean
            self._avg_second_raw_eb_residual = raw_batch_second_eb_residual_mean
            self._avg_raw_reward = th.mean(raw_reward_batch)
        else:
            alpha = self._avgs_alpha_th
            self._avg_raw_exploration_bonus =         alpha * self._avg_raw_exploration_bonus +         (1-alpha)*raw_batch_eb_mean
            self._avg_squared_raw_exploration_bonus = alpha * self._avg_squared_raw_exploration_bonus + (1-alpha)*raw_batch_square_eb_mean
            self._avg_fourth_raw_eb_residual = alpha * self._avg_fourth_raw_eb_residual + (1-alpha)*raw_batch_fourth_eb_residual_mean
            self._avg_second_raw_eb_residual = alpha * self._avg_second_raw_eb_residual + (1-alpha)*raw_batch_second_eb_residual_mean
            self._avg_raw_reward = self._avg_raw_reward * alpha + (1-alpha)*raw_reward_batch

        # eb_min = th.min(exp_bonuses)
        # eb_max = th.max(exp_bonuses)
        interest_threshold = 1.5 # We put 'interesting' at this multiple of std (sigma). 1sigma = 68%, 1.5sigma=85%, 2sigma=95%, 3 sigma = 99.7%
        sigma_squash = 3
        target_bland_ratio = 0.25 # Stuff that is neither boring nor interesting shoud end up accounting for this amount of reward
        target_interesting_ratio = 0.9 # Stuff that is interesting shoud end up accounting for this amount of reward
        rew_increment = 0.01
        kurtosis_min = 10
        kurtosis_max = 20

        # raw_eb_mean = avg_raw_expl_bonus
        # raw_eb_std = th.sqrt(avg_squared_raw_expl_bonus- th.square(avg_raw_expl_bonus))
        raw_eb_kurtosis = th.mean(self._avg_fourth_raw_eb_residual)/th.square(th.mean(self._avg_second_raw_eb_residual))
        raw_batch_eb_std = th.std(raw_bonus_batch)

        # squash and normalize the bonuses 
        norm_exp_bonus = th.tanh((raw_bonus_batch - raw_batch_eb_mean)/(sigma_squash*raw_batch_eb_std))*sigma_squash*raw_batch_eb_std # squash at sigma_squash*sigma
        norm_exp_bonus = norm_exp_bonus/(interest_threshold*raw_batch_eb_std) # normalize at interest_threshold*sigma
        # now interest_threshold*sigma is at 1

        # now:
        # Interesting stuff ends up being beyond +1
        # Normal stuff is at zero
        # Boring stuff is below -1
        # The min and max should be at ±sigma_squash/interesting_threshold

        # shrink using kurtosis, assuming kurtosis gets low when exploration is done
        norm_exp_bonus = norm_exp_bonus*th.clamp((raw_eb_kurtosis - kurtosis_min)/(kurtosis_max-kurtosis_min), min=0, max=1)

        # Scale the squashed/normalized bonuses to the reward
        inc_avg_reward = self._avg_raw_reward + rew_increment # this way even if the reward average is zero we still get an exploration bonus
        scaled_exp_bonus = self._bonus_weight*(inc_avg_reward*target_bland_ratio + 
                                                norm_exp_bonus*inc_avg_reward*target_interesting_ratio)
        
        # scaled_exp_bonus = th.clamp(scaled_exp_bonus, min=0)
        rewards = raw_reward_batch + scaled_exp_bonus.unsqueeze(dim=1)
        
        if return_avg_raw_exp_bonus is not None:
            return_avg_raw_exp_bonus[:] = raw_batch_eb_mean
        if return_avg_proc_exp_bonus is not None:
            return_avg_proc_exp_bonus[:] = th.mean(scaled_exp_bonus)
        if return_all_proc_exp_bonus is not None:
            return_all_proc_exp_bonus[:] = scaled_exp_bonus
        if return_all_norm_exp_bonus is not None:
            return_all_norm_exp_bonus[:] = norm_exp_bonus
        if return_all_raw_exp_bonus is not None:
            return_all_raw_exp_bonus[:] = raw_bonus_batch
        return rewards
