"""
Скрипт валидации обученной модели YOLOv8x.

Вычисляет метрики (F1, Precision, Recall, mAP50, mAP50-95)
на val и test сплитах, сохраняет результаты и графики.

Использование:
    python scripts/val.py --weights artifacts/weights/best.pt
    python scripts/val.py --weights runs/exp_v8x_960_sgd_aug/weights/best.pt
    python scripts/val.py --weights artifacts/weights/best.pt --splits val test
    python scripts/val.py --weights artifacts/weights/best.pt --imgsz 640
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model
from src.utils import plot_comparison_vs_paper, plot_val_test_metrics, save_metrics
from src.validator import print_comparison, validate_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Валидация модели YOLOv8x на датасете HRPlanes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Путь к файлу весов модели (.pt)",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("data/hrplanes.yaml"),
        help="Путь к YAML конфигу датасета",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Размер изображения для инференса",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Сплиты для оценки",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("artifacts"),
        help="Директория для сохранения метрик и графиков",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 55)
    print("ВАЛИДАЦИЯ YOLOv8x — HRPlanes Dataset")
    print("=" * 55)
    print(f"\nВеса:   {args.weights}")
    print(f"YAML:   {args.yaml}")
    print(f"imgsz:  {args.imgsz}")
    print(f"Сплиты: {args.splits}")

    if not args.weights.exists():
        print(f"\n❌ Файл весов не найден: {args.weights}")
        print("Скачайте веса с Google Drive или обучите модель:")
        print("  python scripts/train.py")
        sys.exit(1)

    if not args.yaml.exists():
        print(f"\n❌ YAML конфиг не найден: {args.yaml}")
        print("Подготовьте датасет:")
        print("  python scripts/prepare_dataset.py")
        sys.exit(1)

    # Загрузка модели и валидация
    model = load_model(args.weights)
    results_df = validate_splits(model, args.yaml, args.imgsz, args.splits)

    # Вывод результатов
    print("\n" + "=" * 55)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 55)
    print(results_df.to_string(index=False))

    print_comparison(results_df)

    # Сохранение
    metrics_dir = args.save_dir / "metrics"
    plots_dir = args.save_dir / "plots"

    save_metrics(results_df, metrics_dir)
    plot_val_test_metrics(results_df, plots_dir)
    plot_comparison_vs_paper(results_df, plots_dir)

    print(f"\n✅ Все результаты сохранены в {args.save_dir}/")
    print(f"   Метрики: {metrics_dir / 'results_final.csv'}")
    print(f"   Графики: {plots_dir}/")


if __name__ == "__main__":
    main()
