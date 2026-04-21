"""Логика обучения модели YOLOv8."""

from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: str | Path) -> dict:
    """
    Загружает конфигурацию обучения из YAML файла.

    Args:
        config_path: Путь к YAML файлу конфигурации.

    Returns:
        Словарь с параметрами.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(
    model: YOLO,
    yaml_path: str | Path,
    config: dict,
    results_dir: Path,
) -> Path:
    """
    Запускает обучение модели YOLOv8.

    Args:
        model: Модель YOLO (предобученная или с кастомными весами).
        yaml_path: Путь к YAML конфигу датасета.
        config: Словарь с гиперпараметрами (из train_config.yaml).
        results_dir: Директория для сохранения результатов обучения.

    Returns:
        Путь к лучшим весам (best.pt).
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config["training"]
    aug_cfg = config["augmentation"]
    exp_cfg = config["experiment"]

    model.train(
        data=str(yaml_path),
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        lr0=train_cfg["lr0"],
        optimizer=train_cfg["optimizer"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        name=exp_cfg["name"],
        project=str(results_dir),
        exist_ok=True,
        workers=train_cfg["workers"],
        device=train_cfg["device"],
        seed=exp_cfg["seed"],
        save=True,
        save_period=train_cfg["save_period"],
        verbose=True,
        # Аугментация из статьи
        hsv_h=aug_cfg["hsv_h"],
        hsv_s=aug_cfg["hsv_s"],
        hsv_v=aug_cfg["hsv_v"],
        mosaic=aug_cfg["mosaic"],
        flipud=aug_cfg["flipud"],
        fliplr=aug_cfg["fliplr"],
    )

    best_pt = results_dir / exp_cfg["name"] / "weights" / "best.pt"
    print(f"\n✅ Обучение завершено. Лучшие веса: {best_pt}")
    return best_pt
