"""Тесты для модуля src/validator.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validator import extract_metrics, print_comparison


def make_mock_metrics(precision: float, recall: float, map50: float, map_val: float):
    """Создаёт mock-объект метрик Ultralytics."""
    mock = MagicMock()
    mock.box.mp = precision
    mock.box.mr = recall
    mock.box.map50 = map50
    mock.box.map = map_val
    return mock


def test_extract_metrics_values():
    """Проверяет корректное извлечение метрик."""
    mock = make_mock_metrics(0.99, 0.98, 0.993, 0.885)
    result = extract_metrics(mock)

    assert result["Precision"] == 0.99
    assert result["Recall"] == 0.98
    assert result["mAP50"] == 0.993
    assert result["mAP50-95"] == 0.885
    assert "F1" in result


def test_extract_metrics_f1_formula():
    """Проверяет правильность формулы F1."""
    p, r = 0.9, 0.8
    mock = make_mock_metrics(p, r, 0.9, 0.8)
    result = extract_metrics(mock)

    expected_f1 = 2 * p * r / (p + r)
    assert abs(result["F1"] - round(expected_f1, 4)) < 1e-4


def test_extract_metrics_perfect():
    """Проверяет F1=1.0 при Precision=Recall=1.0."""
    mock = make_mock_metrics(1.0, 1.0, 1.0, 1.0)
    result = extract_metrics(mock)

    assert result["F1"] == 1.0


def test_print_comparison_runs(capsys):
    """Проверяет, что print_comparison не падает и выводит текст."""
    results_df = pd.DataFrame([
        {
            "Split": "Test",
            "Precision": 0.9932,
            "Recall": 0.9879,
            "F1": 0.9906,
            "mAP50": 0.9943,
            "mAP50-95": 0.8897,
        }
    ])
    print_comparison(results_df)
    captured = capsys.readouterr()
    assert "СРАВНЕНИЕ" in captured.out
    assert "mAP50" in captured.out


def test_print_comparison_no_test(capsys):
    """Проверяет корректную обработку отсутствия Test сплита."""
    results_df = pd.DataFrame([
        {"Split": "Val", "Precision": 0.99, "Recall": 0.99,
         "F1": 0.99, "mAP50": 0.99, "mAP50-95": 0.88}
    ])
    print_comparison(results_df)
    captured = capsys.readouterr()
    assert "не найдены" in captured.out
