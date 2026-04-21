"""Тесты для модуля src/dataset.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import build_split_lists, load_all_labels


def test_build_split_lists_counts(tmp_path):
    """Проверяет корректное количество файлов при случайной разбивке 70/20/10."""
    img_dir = tmp_path / "img"
    img_dir.mkdir()

    for i in range(100):
        (img_dir / f"img_{i:03d}.jpg").touch()

    splits = build_split_lists(tmp_path, seed=42)

    assert len(splits["train"]) == 70
    assert len(splits["valid"]) == 20
    assert len(splits["test"]) == 10


def test_build_split_lists_no_overlap(tmp_path):
    """Проверяет, что сплиты не пересекаются."""
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    for i in range(100):
        (img_dir / f"img_{i:03d}.jpg").touch()

    splits = build_split_lists(tmp_path, seed=42)

    train_names = {p.name for p in splits["train"]}
    valid_names = {p.name for p in splits["valid"]}
    test_names = {p.name for p in splits["test"]}

    assert len(train_names & valid_names) == 0
    assert len(train_names & test_names) == 0
    assert len(valid_names & test_names) == 0


def test_build_split_lists_reproducible(tmp_path):
    """Проверяет воспроизводимость разбивки при одинаковом seed."""
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    for i in range(50):
        (img_dir / f"img_{i:03d}.jpg").touch()

    splits1 = build_split_lists(tmp_path, seed=42)
    splits2 = build_split_lists(tmp_path, seed=42)

    assert [p.name for p in splits1["train"]] == [p.name for p in splits2["train"]]


def test_build_split_lists_different_seeds(tmp_path):
    """Проверяет, что разные seed дают разные разбивки."""
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    for i in range(50):
        (img_dir / f"img_{i:03d}.jpg").touch()

    splits1 = build_split_lists(tmp_path, seed=42)
    splits2 = build_split_lists(tmp_path, seed=123)

    # С большой вероятностью разбивки должны отличаться
    assert [p.name for p in splits1["train"]] != [p.name for p in splits2["train"]]


def test_load_all_labels_empty_dir(tmp_path):
    """Проверяет обработку пустой директории меток."""
    lbl_dir = tmp_path / "test" / "labels"
    lbl_dir.mkdir(parents=True)

    boxes, counts = load_all_labels(tmp_path, "test")

    assert boxes.shape == (0, 4)
    assert counts == []


def test_load_all_labels_with_annotations(tmp_path):
    """Проверяет корректное чтение аннотаций."""
    lbl_dir = tmp_path / "train" / "labels"
    lbl_dir.mkdir(parents=True)

    # Создаём файл с 3 аннотациями
    ann_file = lbl_dir / "img_001.txt"
    ann_file.write_text(
        "0 0.5 0.5 0.1 0.1\n"
        "0 0.3 0.3 0.05 0.08\n"
        "0 0.7 0.6 0.12 0.09\n"
    )

    boxes, counts = load_all_labels(tmp_path, "train")

    assert boxes.shape == (3, 4)
    assert counts == [3]
    assert np.allclose(boxes[0], [0.5, 0.5, 0.1, 0.1])
