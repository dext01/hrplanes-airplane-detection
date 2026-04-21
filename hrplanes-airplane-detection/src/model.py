"""Инициализация и управление моделью YOLOv8."""

from pathlib import Path

from ultralytics import YOLO


def load_model(weights: str | Path = "yolov8x.pt") -> YOLO:
    """
    Загружает модель YOLOv8.

    Args:
        weights: Путь к весам или название предобученной модели
                 (например, "yolov8x.pt").

    Returns:
        Загруженная модель YOLO.
    """
    model = YOLO(str(weights))
    info = get_model_info(model)
    print(f"Модель загружена: {weights}")
    print(f"Параметров: {info['parameters_M']}M")
    return model


def get_model_info(model: YOLO) -> dict:
    """
    Возвращает базовую информацию о модели.

    Args:
        model: Модель YOLO.

    Returns:
        Словарь с ключами architecture, parameters, parameters_M.
    """
    params = sum(p.numel() for p in model.model.parameters())
    return {
        "architecture": model.model.__class__.__name__,
        "parameters": params,
        "parameters_M": round(params / 1e6, 2),
    }
