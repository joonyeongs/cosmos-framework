# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch
import torchvision.transforms.v2 as T

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    build_episode_spans,
    parse_source_intervals,
    replay_start_ranges,
)
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

_IMAGE_FEATURES = {"wrist": "wrist_key", "left": "left_key", "right": "right_key"}


@pytest.mark.L0
@pytest.mark.parametrize(
    ("stride", "expected"),
    [(1, [(1, 23, 3), (1, 34, 1)]), (2, [(1, 24, 1), (1, 34, 1)])],
)
def test_training_start_ranges_limit_only_the_initial_observation(stride, expected):
    episodes = {
        "dataset_from_index": [0, 20],
        "dataset_to_index": [20, 40],
        "length": [20, 20],
        "training_start_ranges": [[], [[3, 6], [14, 20]]],
    }
    spans, count, raw_frames = build_episode_spans(episodes, [0, 1], chunk_length=5, sample_stride=stride)
    assert spans == expected
    assert count == sum(span[2] for span in expected)
    assert raw_frames == 40


@pytest.mark.L0
def test_parse_source_intervals_accepts_python_literal_strings():
    literal = "[{'node': None, 'source': 'cosmos3', 'start': 3, 'stop': 9}]"
    assert parse_source_intervals(literal) == [{"node": None, "source": "cosmos3", "start": 3, "stop": 9}]
    assert parse_source_intervals('[{"source": "cosmos3", "start": 1, "stop": 2}]')[0]["stop"] == 2
    with pytest.raises(ValueError):
        parse_source_intervals([{"source": "cosmos3", "start": 1}])


@pytest.mark.L0
def test_replay_ranges_take_cosmos3_steps_outside_primary_starts():
    episodes = {
        "dataset_from_index": [0, 70],
        "dataset_to_index": [70, 110],
        "length": [70, 40],
        # Privileged interval [20, 25) padded by ten frames on each side.
        "training_start_ranges": [[[10, 35]], []],
        "source_intervals": [
            str(
                [
                    {"source": "code_without_privileged", "start": 0, "stop": 5},
                    {"source": "cosmos3", "start": 5, "stop": 20},
                    {"source": "code_with_privileged", "start": 20, "stop": 25},
                    {"source": "cosmos3", "start": 25, "stop": 70},
                ]
            ),
            [{"source": "cosmos3", "start": 0, "stop": 40}],
        ],
    }
    ranges = replay_start_ranges(episodes, [0, 1], ["cosmos3"])
    assert ranges == {0: [(5, 10), (35, 70)], 1: [(0, 40)]}
    spans, count, _ = build_episode_spans(episodes, [0, 1], chunk_length=32, start_ranges=ranges)
    # Episode 0 can start before 70 - 32 = 38; episode 1 before 8.
    assert spans == [(0, 5, 5), (0, 35, 3), (1, 70, 8)]
    assert count == 16
    primary, _, _ = build_episode_spans(episodes, [0, 1], chunk_length=32)
    assert primary == [(0, 10, 25)]


@pytest.mark.L0
def test_replay_padding_widens_intervals_but_keeps_primary_starts_out():
    episodes = {
        "length": [100],
        "training_start_ranges": [[[0, 20]]],
        "source_intervals": [
            [
                {"source": "cosmos3", "start": 25, "stop": 30},
                {"source": "code_without_privileged", "start": 30, "stop": 45},
                {"source": "cosmos3", "start": 45, "stop": 55},
                {"source": "cosmos3", "start": 95, "stop": 100},
            ]
        ],
    }
    assert replay_start_ranges(episodes, [0], ["cosmos3"]) == {0: [(25, 30), (45, 55), (95, 100)]}
    # Padded by 10: (15, 40) and (35, 65) merge; (85, 100) is clipped; [0, 20) stays primary.
    assert replay_start_ranges(episodes, [0], ["cosmos3"], padding=10) == {0: [(20, 65), (85, 100)]}


class _PooledIndices:
    """Minimal pooled dataset: returns the flat index it was asked for."""

    def __init__(self, primary, replay):
        self._pools = (primary, replay)

    def __len__(self):
        return sum(length for _, length in self._pools[0] + self._pools[1])

    def __getitem__(self, idx):
        return idx

    def get_shuffle_blocks(self):
        return self._pools[0] + self._pools[1]

    def get_shuffle_block_pools(self):
        return self._pools


def _take(stream, n):
    iterator = iter(stream)
    return [next(iterator) for _ in range(n)]


@pytest.mark.L0
def test_replay_fraction_interleaves_one_replay_window_per_four_primary():
    primary = [(0, 12), (12, 9), (21, 30)]
    replay = [(51, 6), (57, 4)]
    samples = _take(ActionIterableShuffleDataset(_PooledIndices(primary, replay), seed=7, replay_fraction=0.2), 500)
    is_replay = [idx >= 51 for idx in samples]
    assert sum(is_replay) == 100
    assert is_replay[:10] == [False, False, False, False, True] * 2
    # Within each pool the order is the replay-free block stream of that pool.
    primary_only = _take(ActionIterableShuffleDataset(_PooledIndices(primary, []), seed=7), 400)
    assert [idx for idx in samples if idx < 51] == primary_only


@pytest.mark.L0
def test_zero_replay_fraction_streams_primary_blocks_only():
    primary = [(0, 5), (5, 7)]
    replay = [(12, 9)]
    samples = _take(ActionIterableShuffleDataset(_PooledIndices(primary, replay), seed=3), 60)
    assert max(samples) < 12
    with pytest.raises(ValueError):
        _take(ActionIterableShuffleDataset(_PooledIndices(primary, []), seed=3, replay_fraction=0.2), 1)
    with pytest.raises(ValueError):
        ActionIterableShuffleDataset(_PooledIndices(primary, replay), replay_fraction=1.0)


def _dataset(use_image_augmentation: bool, jitter_after_compose: bool = False) -> DROIDLeRobotDataset:
    """Build a bare instance exercising only ``_compose_multi_view``.

    ``__init__`` opens LeRobot datasets from disk, which this test does not need:
    the method under test reads three tensors out of ``sample`` and the
    augmentation attributes, and touches nothing else.
    """
    dataset = object.__new__(DROIDLeRobotDataset)
    dataset._image_features = _IMAGE_FEATURES
    dataset._use_image_augmentation = use_image_augmentation
    dataset._jitter_after_compose = jitter_after_compose
    dataset._image_augmentor = None
    dataset._color_augmentor = None
    dataset._use_color_augmentation = True
    return dataset


def _sample(wrist: float, left: float, right: float, t: int = 2, h: int = 16, w: int = 32) -> dict:
    return {
        "wrist_key": torch.full((t, 3, h, w), wrist),
        "left_key": torch.full((t, 3, h, w), left),
        "right_key": torch.full((t, 3, h, w), right),
    }


@pytest.mark.L0
def test_compose_multi_view_tiles_wrist_over_left_and_right() -> None:
    """Wrist spans the top, left and right tile the bottom at half scale."""
    composite = _dataset(use_image_augmentation=False)._compose_multi_view(_sample(0.1, 0.5, 0.9))

    assert composite.shape == (2, 3, 24, 32)  # [T,C,3H/2,W]
    torch.testing.assert_close(composite[:, :, :16, :], torch.full((2, 3, 16, 32), 0.1))
    torch.testing.assert_close(composite[:, :, 16:, :16], torch.full((2, 3, 8, 16), 0.5))
    torch.testing.assert_close(composite[:, :, 16:, 16:], torch.full((2, 3, 8, 16), 0.9))


@pytest.mark.L0
@pytest.mark.parametrize("jitter_after_compose", [False, True])
def test_compose_multi_view_keeps_layout_under_augmentation(jitter_after_compose: bool) -> None:
    """Augmentation must not change the composite's shape or tiling."""
    torch.manual_seed(0)
    dataset = _dataset(use_image_augmentation=True, jitter_after_compose=jitter_after_compose)
    composite = dataset._compose_multi_view(_sample(0.1, 0.5, 0.9))

    assert composite.shape == (2, 3, 24, 32)
    # Each region is still uniform, so the tiles did not bleed into each other.
    for region in (composite[:, :, :16, :], composite[:, :, 16:, :16], composite[:, :, 16:, 16:]):
        assert torch.allclose(region, region.flatten()[0].expand_as(region), atol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("jitter_after_compose", [False, True])
def test_color_jitter_is_shared_across_the_three_views(jitter_after_compose: bool) -> None:
    """One colour draw covers the whole composite.

    Jittering the views independently would cut colour diversity across the
    frame and make the three cameras disagree on lighting, so identical input
    views have to stay identical after augmentation.
    """
    torch.manual_seed(0)
    dataset = _dataset(use_image_augmentation=True, jitter_after_compose=jitter_after_compose)
    composite = dataset._compose_multi_view(_sample(0.4, 0.4, 0.4))

    wrist_px = composite[:, :, :16, :].flatten()[0]
    torch.testing.assert_close(composite[:, :, 16:, :16], torch.full((2, 3, 8, 16), wrist_px))
    torch.testing.assert_close(composite[:, :, 16:, 16:], torch.full((2, 3, 8, 16), wrist_px))


@pytest.mark.L0
@pytest.mark.parametrize("jitter_after_compose", [False, True])
def test_augmentation_actually_changes_the_composite(jitter_after_compose: bool) -> None:
    """Guards against the augmentor silently becoming a no-op."""
    sample = _sample(0.4, 0.6, 0.8)
    plain = _dataset(use_image_augmentation=False)._compose_multi_view(sample)
    torch.manual_seed(0)
    dataset = _dataset(use_image_augmentation=True, jitter_after_compose=jitter_after_compose)
    augmented = dataset._compose_multi_view(sample)

    assert not torch.allclose(plain, augmented)


@pytest.mark.L0
def test_default_keeps_jitter_inside_the_per_view_pipeline() -> None:
    """``jitter_after_compose`` defaults to the original ordering.

    Off: ColorJitter stays bundled in the per-view Compose and nothing runs on
    the composite, byte-identical to the pre-flag pipeline. On: ColorJitter
    leaves the per-view Compose and runs on the composite instead.
    """
    default = _dataset(use_image_augmentation=True)
    default._compose_multi_view(_sample(0.1, 0.5, 0.9))
    assert default._color_augmentor is None
    assert any(isinstance(t, T.ColorJitter) for t in default._image_augmentor.transforms)

    reordered = _dataset(use_image_augmentation=True, jitter_after_compose=True)
    reordered._compose_multi_view(_sample(0.1, 0.5, 0.9))
    assert isinstance(reordered._color_augmentor, T.ColorJitter)
    assert not any(isinstance(t, T.ColorJitter) for t in reordered._image_augmentor.transforms)


@pytest.mark.parametrize("jitter_after_compose", [False, True])
def test_disable_color_preserves_crop_resize_and_colors(jitter_after_compose):
    dataset = _dataset(use_image_augmentation=True, jitter_after_compose=jitter_after_compose)
    dataset._use_color_augmentation = False
    result = dataset._compose_multi_view(_sample(0.1, 0.5, 0.9))
    assert dataset._color_augmentor is None
    assert [type(t) for t in dataset._image_augmentor.transforms] == [T.RandomCrop, T.Resize]
    torch.testing.assert_close(result[:, :, :16], torch.full((2, 3, 16, 32), 0.1))
    torch.testing.assert_close(result[:, :, 16:, :16], torch.full((2, 3, 8, 16), 0.5))
    torch.testing.assert_close(result[:, :, 16:, 16:], torch.full((2, 3, 8, 16), 0.9))
