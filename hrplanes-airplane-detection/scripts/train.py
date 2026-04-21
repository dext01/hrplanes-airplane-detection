"""
Скрипт обучения YOLOv8x на датасете HRPlanes.

Воспроизводит Experiment 12 из статьи:
"Exploring YOLOv8 and YOLOv9 for Efficient Airplane Detection
in VHR Remote Sensing Imagery" (Ilmak et al., 2025)

Использование:
    python scripts/train.py
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --epochs 50 --imgsz 640
    python scripts/train.py --weights artifacts/weights/best.pt --epochs 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model
from src.trainer import load_config, train
from src.utils import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение YOLOv8x на датасете HRPlanes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Путь к конфигурационному файлу",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Путь к весам для дообучения (по умолчанию из конфига)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Количество эпох (переопределяет значение из конфига)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Размер изображения (переопределяет значение из конфига)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("runs"),
        help="Директория для сохранения результатов обучения",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/plots"),
        help="Директория для сохранения графиков",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 55)
    print("ОБУЧЕНИЕ YOLOv8x — HRPlanes Dataset")
    print("=" * 55)

    # Загружаем конфиг
    config = load_config(args.config)

    # Переопределяем параметры из CLI если указаны
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.imgsz is not None:
        config["training"]["imgsz"] = args.imgsz

    weights = args.weights or config["model"]["architecture"] + ".pt"
    yaml_path = Path(config["data"]["yaml_path"])

    if not yaml_path.exists():
        print(f"\n❌ YAML конфиг не найден: {yaml_path}")
        print("Сначала подготовьте датасет:")
        print("  python scripts/prepare_dataset.py")
        sys.exit(1)

    print(f"\nКонфиг:    {args.config}")
    print(f"Веса:      {weights}")
    print(f"Датасет:   {yaml_path}")
    print(f"Эпох:      {config['training']['epochs']}")
    print(f"imgsz:     {config['training']['imgsz']}")
    print(f"batch:     {config['training']['batch']}")
    print(f"optimizer: {config['training']['optimizer']}")
    print(f"seed:      {config['experiment']['seed']}")

    # Обучение
    model = load_model(weights)
    best_pt = train(model, yaml_path, config, args.results_dir)

    # Строим кривые обучения
    exp_name = config["experiment"]["name"]
    curves_csv = args.results_dir / exp_name / "results.csv"
    plot_training_curves(curves_csv, args.artifacts_dir)

    print(f"\n🎉 Обучение завершено!")
    print(f"   Лучшие веса: {best_pt}")
    print(f"   Запуск валидации: python scripts/val.py --weights {best_pt}")


if __name__ == "__main__":
    main()
