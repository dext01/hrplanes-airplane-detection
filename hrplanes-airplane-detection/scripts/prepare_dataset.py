"""
Скрипт подготовки датасета HRPlanes.

Скачивает датасет с Zenodo, распаковывает архив и организует
файлы в YOLO-структуру (train/valid/test).

Использование:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --data-dir data/HRPlanes --seed 42
    python scripts/prepare_dataset.py --skip-download  # если уже скачан
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для импорта src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    build_split_lists,
    copy_to_yolo_structure,
    create_yaml,
    download_dataset,
    extract_dataset,
    load_all_labels,
    verify_structure,
)
from src.utils import plot_eda, visualize_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка датасета HRPlanes для обучения YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/HRPlanes"),
        help="Директория для хранения датасета",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=Path("data/hrplanes.yaml"),
        help="Путь для сохранения YAML конфига",
    )
    parser.add_argument(
        "--zenodo-id",
        type=int,
        default=14546832,
        help="ID датасета на Zenodo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed для воспроизводимости разбивки",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Пропустить скачивание (если датасет уже скачан)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/plots"),
        help="Директория для сохранения EDA графиков",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 55)
    print("ПОДГОТОВКА ДАТАСЕТА HRPlanes")
    print("=" * 55)

    # Шаг 1: Скачивание и распаковка
    if not args.skip_download:
        download_dataset(args.data_dir, args.zenodo_id)
        extract_dataset(args.data_dir)
    else:
        print("Скачивание пропущено (--skip-download).")

    # Шаг 2: Построение сплитов
    print("\nСтроим сплиты (seed={})...".format(args.seed))
    splits = build_split_lists(args.data_dir, seed=args.seed)
    for name, lst in splits.items():
        print(f"  {name}: {len(lst)} изображений")

    # Шаг 3: Копирование в YOLO-структуру
    print("\nКопируем в YOLO-структуру...")
    copy_to_yolo_structure(args.data_dir, splits)

    # Шаг 4: YAML конфиг
    print("\nСоздаём YAML конфиг...")
    create_yaml(args.data_dir, args.yaml_path)

    # Шаг 5: Верификация
    print("\nПроверяем структуру:")
    ok = verify_structure(args.data_dir)

    if not ok:
        print("\n❌ Ошибка: некоторые сплиты пустые.")
        sys.exit(1)

    # Шаг 6: EDA
    print("\nСтроим EDA графики...")
    train_boxes, train_counts = load_all_labels(args.data_dir, "train")
    val_boxes, val_counts = load_all_labels(args.data_dir, "valid")
    test_boxes, test_counts = load_all_labels(args.data_dir, "test")
    all_counts = train_counts + val_counts + test_counts

    print(f"  Всего аннотаций: {len(train_boxes) + len(val_boxes) + len(test_boxes):,}")

    plot_eda(train_boxes, val_boxes, test_boxes, all_counts, args.artifacts_dir)
    visualize_samples(args.data_dir, "train", args.artifacts_dir, n=4)
    visualize_samples(args.data_dir, "test", args.artifacts_dir, n=4)

    print("\n✅ Датасет готов к обучению!")
    print(f"   YAML: {args.yaml_path}")
    print(f"   Запуск обучения: python scripts/train.py")


if __name__ == "__main__":
    main()
