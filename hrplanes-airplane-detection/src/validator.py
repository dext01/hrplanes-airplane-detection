"""Логика валидации и вычисления метрик модели."""

from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def extract_metrics(metrics_obj) -> dict:
    """
    Извлекает метрики из объекта результатов Ultralytics.

    Args:
        metrics_obj: Объект результатов model.val().

    Returns:
        Словарь с Precision, Recall, F1, mAP50, mAP50-95.
    """
    p = float(metrics_obj.box.mp)
    r = float(metrics_obj.box.mr)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1": round(f1, 4),
        "mAP50": round(float(metrics_obj.box.map50), 4),
        "mAP50-95": round(float(metrics_obj.box.map), 4),
    }


def validate_splits(
    model: YOLO,
    yaml_path: str | Path,
    imgsz: int = 960,
    splits: list[str] | None = None,
) -> pd.DataFrame:
    """
    Запускает валидацию на указанных сплитах датасета.

    Args:
        model: Загруженная модель YOLO.
        yaml_path: Путь к YAML конфигу датасета.
        imgsz: Размер изображения для инференса.
        splits: Список сплитов. По умолчанию ["val", "test"].

    Returns:
        DataFrame с метриками: Split | Precision | Recall | F1 | mAP50 | mAP50-95.
    """
    if splits is None:
        splits = ["val", "test"]

    rows = []
    for split in splits:
        print(f"Оцениваем на {split}...")
        metrics = model.val(
            data=str(yaml_path),
            imgsz=imgsz,
            split=split,
            verbose=False,
        )
        row = {"Split": split.capitalize(), **extract_metrics(metrics)}
        rows.append(row)
        print(
            f"  {split}: F1={row['F1']:.4f} | "
            f"P={row['Precision']:.4f} | R={row['Recall']:.4f} | "
            f"mAP50={row['mAP50']:.4f} | mAP50-95={row['mAP50-95']:.4f}"
        )

    return pd.DataFrame(rows)


def print_comparison(results_df: pd.DataFrame) -> None:
    """
    Выводит таблицу сравнения наших результатов с результатами статьи.

    Args:
        results_df: DataFrame с нашими метриками (должна содержать Split=Test).
    """
    paper = {
        "F1": 0.9932,
        "Precision": 0.9915,
        "Recall": 0.9950,
        "mAP50": 0.9939,
        "mAP50-95": 0.8990,
    }

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ С РЕЗУЛЬТАТАМИ СТАТЬИ (Exp12, YOLOv8x SGD)")
    print("=" * 60)
    print(f"{'Метрика':<12} {'Статья':>10} {'Наш Test':>10} {'Δ':>8}")
    print("-" * 44)

    test_row = results_df[results_df["Split"] == "Test"]
    if test_row.empty:
        print("Test метрики не найдены.")
        return

    for m in ["F1", "Precision", "Recall", "mAP50", "mAP50-95"]:
        p_val = paper[m]
        o_val = float(test_row[m].values[0])
        delta = o_val - p_val
        sign = "+" if delta >= 0 else ""
        print(f"  {m:<10} {p_val:>10.4f} {o_val:>10.4f} {sign}{delta:>7.4f}")
