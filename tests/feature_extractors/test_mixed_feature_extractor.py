"""Tests for MixedFeatureExtractor: dict observations mixing vectors and one image."""
from __future__ import annotations

import zipfile
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from rreal.feature_extractors import get_feature_extractor
from rreal.feature_extractors.mixed_feature_extractor import (MixedFeatureExtractor,
                                                              MixedFeatureExtractorInitArgs)

VEC_SIZE = 7 + 7  # joint_pos + joint_vel
IMG_CHW = (3, 64, 64)
ENCODING_SIZE = 32
CPU = th.device("cpu")


def mixed_obs_space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "joint_pos": gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32),
        "joint_vel": gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32),
        "camera":    gym.spaces.Box(low=0, high=255, shape=IMG_CHW, dtype=np.uint8),
    })


def make_hp(**overrides) -> MixedFeatureExtractorInitArgs:
    args = dict(device=CPU, encoding_size=ENCODING_SIZE, img_encoding_size=16,
                vec_encoding_size=16, img_backbone="conv", use_batchnorm=False)
    args.update(overrides)
    return MixedFeatureExtractorInitArgs(**args)


def make_obs(*batch_dims: int) -> dict[str, th.Tensor]:
    return {"joint_pos": th.randn(*batch_dims, 7),
            "joint_vel": th.randn(*batch_dims, 7),
            "camera":    th.randint(0, 256, tuple(batch_dims) + IMG_CHW, dtype=th.uint8)}


@pytest.fixture(scope="module")
def extractor() -> MixedFeatureExtractor:
    """Built once for the module: building it is the expensive part of these tests."""
    th.manual_seed(0)
    return MixedFeatureExtractor(observation_space=mixed_obs_space(), hp=make_hp())


def test_is_registered():
    assert get_feature_extractor("MixedFeatureExtractor") is MixedFeatureExtractor


def test_encodes_a_batch(extractor):
    enc = extractor.extract_features(make_obs(5))
    assert enc.size() == (5, ENCODING_SIZE)
    assert enc.size()[-1] == extractor.encoding_size()


def test_encodes_a_batch_of_trajectories(extractor):
    """Observations may carry a trajectory dimension; it must survive the encoder round trip."""
    enc = extractor.extract_features(make_obs(4, 3))
    assert enc.size() == (4, 3, ENCODING_SIZE)


def test_vector_normalization_is_not_confused_by_the_trajectory_dim(extractor):
    """Regression: normalizing before flattening made RunningMeanStd build (traj, VEC_SIZE)
    statistics instead of (VEC_SIZE,), and blow up on the running-stat assignment."""
    extractor.train()
    try:
        enc = extractor.extract_features(make_obs(4, 3))
    finally:
        extractor.eval()
    assert th.isfinite(enc).all()


def test_images_are_scaled_to_unit_range(extractor):
    """uint8 [0,255] pixels must reach the encoder as floats in [0,1]."""
    _, img, batch_dims = extractor._preprocess(make_obs(5))
    assert img.dtype == th.float32
    assert batch_dims == (5,)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_gradients_reach_the_encoder(extractor):
    """Unlike StackVectorsFeatureExtractor this one is trainable end-to-end, so
    extract_features must stay differentiable."""
    extractor.zero_grad(set_to_none=True)
    enc = extractor.extract_features(make_obs(5))
    assert enc.requires_grad
    enc.sum().backward()
    with_grad = [n for n, p in extractor.named_parameters()
                 if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(with_grad) == len(list(extractor.parameters()))
    extractor.zero_grad(set_to_none=True)


def test_save_load_round_trip(extractor, tmp_path):
    extractor.eval()
    obs = make_obs(2)
    before = extractor.extract_features(obs).detach()

    path = tmp_path/"model.zip"
    with zipfile.ZipFile(path, "w") as archive:
        extractor.save_to_archive(archive)
    with zipfile.ZipFile(path, "r") as archive:
        reloaded = MixedFeatureExtractor.load(archive)

    assert reloaded is not None, "load() must return the extractor"
    reloaded.eval()
    after = reloaded.extract_features(obs).detach()
    assert th.allclose(before, after)
    reloaded.check_init_args_match("MixedFeatureExtractor", extractor.get_init_args())


def test_load_can_retarget_the_device(extractor, tmp_path):
    """Regression: load() used to inject a top-level 'device' init arg, which the hp-based
    constructor rejects with a TypeError."""
    path = tmp_path/"model.zip"
    with zipfile.ZipFile(path, "w") as archive:
        extractor.save_to_archive(archive)
    with zipfile.ZipFile(path, "r") as archive:
        reloaded = MixedFeatureExtractor.load(archive, device=CPU)
    assert reloaded.extract_features(make_obs(2)).size() == (2, ENCODING_SIZE)


def test_check_init_args_match_ignores_the_device(extractor):
    """A checkpoint trained on GPU must be loadable into a CPU run and vice versa."""
    other_device_args = {"observation_space": mixed_obs_space(),
                         "hp": replace(make_hp(), device=th.device("cuda:0"))}
    extractor.check_init_args_match("MixedFeatureExtractor", other_device_args)


def test_check_init_args_match_does_not_mutate_the_init_args(extractor):
    """Regression: the device was neutralized by assigning to hp.device on a shallow dict copy,
    which permanently blanked the device on the live extractor (and on the caller's args)."""
    loaded_args = {"observation_space": mixed_obs_space(), "hp": make_hp()}
    extractor.check_init_args_match("MixedFeatureExtractor", loaded_args)
    assert extractor.get_init_args()["hp"].device == CPU
    assert loaded_args["hp"].device == CPU


def test_vector_only_observation_disables_the_image_branch():
    space = gym.spaces.Dict({"joint_pos": gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)})
    fe = MixedFeatureExtractor(observation_space=space,
                               hp=make_hp(encoding_size=8, img_encoding_size=16, vec_encoding_size=8))
    assert fe.extract_features({"joint_pos": th.randn(5, 7)}).size() == (5, 8)


def test_image_only_observation_disables_the_vector_branch():
    space = gym.spaces.Dict({"camera": gym.spaces.Box(low=0, high=255, shape=IMG_CHW, dtype=np.uint8)})
    fe = MixedFeatureExtractor(observation_space=space,
                               hp=make_hp(encoding_size=8, img_encoding_size=8, vec_encoding_size=16))
    obs = {"camera": th.randint(0, 256, (5,)+IMG_CHW, dtype=th.uint8)}
    assert fe.extract_features(obs).size() == (5, 8)


def test_identity_combiner_requires_matching_encoding_size():
    with pytest.raises(AttributeError):
        MixedFeatureExtractor(observation_space=mixed_obs_space(),
                              hp=make_hp(encoding_size=999, img_encoding_size=16,
                                         vec_encoding_size=16, combiner_arch="identity"))
