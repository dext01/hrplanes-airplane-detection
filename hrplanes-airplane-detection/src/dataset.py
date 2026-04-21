"""Утилиты для загрузки и подготовки датасета HRPlanes."""

import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml


def download_dataset(data_dir: Path, zenodo_id: int = 14546832) -> None:
    """Скачивает датасет HRPlanes с Zenodo."""
    data_dir.mkdir(parents=True, exist_ok=True)

    if any(data_dir.glob("*.7z.*")) or (data_dir / "img").exists():
        print("Датасет уже скачан, пропускаем.")
        return

    print(f"Скачиваем HRPlanes с Zenodo {zenodo_id} (~10 GB)...")
    subprocess.run(["zenodo_get", str(zenodo_id), "-o", str(data_dir)], check=True)


def extract_dataset(data_dir: Path) -> None:
    """Распаковывает 7z архив датасета."""
    img_dir = data_dir / "img"
    if img_dir.exists():
        print("Датасет уже распакован, пропускаем.")
        return

    z_parts = sorted(data_dir.glob("*.7z.*"))
    if not z_parts:
        raise FileNotFoundError(f"Архив .7z не найден в {data_dir}")

    print(f"Распаковываем {z_parts[0].name}...")
    subprocess.run(["7z", "x", str(z_parts[0]), f"-o{data_dir}", "-y"], check=True)


def build_split_lists(
    data_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
) -> dict[str, list[Path]]:
    """
    Строит списки файлов для train/valid/test сплитов.

    Использует файлы разбивки из датасета если они есть,
    иначе делает случайную разбивку по заданным пропорциям.

    Args:
        data_dir: Директория с датасетом.
        seed: Random seed для воспроизводимости.
        train_ratio: Доля тренировочных данных.
        val_ratio: Доля валидационных данных.

    Returns:
        Словарь {split_name: [list of image paths]}.
    """
    img_dir = data_dir / "img"
    all_imgs = sorted(img_dir.glob("*.jpg"))

    # Ищем файлы разбивки из датасета
    split_files: dict[str, Path | None] = {"train": None, "valid": None, "test": None}
    for f in data_dir.glob("*.txt"):
        name = f.name.lower()
        if "train" in name and "valid" not in name:
            split_files["train"] = f
        elif "valid" in name or "val" in name:
            split_files["valid"] = f
        elif "test" in name:
            split_files["test"] = f

    if all(v is None for v in split_files.values()):
        print("Файлы разбивки не найдены — делаем случайную разбивку...")
        random.seed(seed)
        shuffled = list(all_imgs)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return {
            "train": shuffled[:n_train],
            "valid": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }

    splits: dict[str, list[Path]] = {}
    for split_name, txt_path in split_files.items():
        if txt_path is None:
            splits[split_name] = []
            continue
        names: set[str] = set()
        for line in txt_path.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                names.add(Path(line).stem)
        splits[split_name] = [
            img_dir / (s + ".jpg") for s in names if (img_dir / (s + ".jpg")).exists()
        ]

    return splits


def copy_to_yolo_structure(
    data_dir: Path,
    splits: dict[str, list[Path]],
) -> None:
    """
    Копирует файлы изображений и аннотаций в YOLO-структуру.

    Создаёт папки split/images/ и split/labels/ для каждого сплита.

    Args:
        data_dir: Корневая директория датасета.
        splits: Словарь {split_name: [list of image paths]}.
    """
    img_dir = data_dir / "img"

    for split_name, img_list in splits.items():
        img_out = data_dir / split_name / "images"
        lbl_out = data_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            lbl_path = img_dir / (img_path.stem + ".txt")
            if not (img_out / img_path.name).exists():
                shutil.copy2(img_path, img_out / img_path.name)
            if lbl_path.exists() and not (lbl_out / lbl_path.name).exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)


def create_yaml(data_dir: Path, output_path: Path) -> None:
    """Создаёт YAML конфиг для Ultralytics YOLO."""
    config = {
        "path": str(data_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["airplane"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"YAML создан: {output_path}")


def verify_structure(data_dir: Path) -> bool:
    """
    Проверяет корректность YOLO-структуры датасета.

    Returns:
        True если все сплиты непустые, иначе False.
    """
    all_ok = True
    for split in ["train", "valid", "test"]:
        imgs = list((data_dir / split / "images").glob("*.jpg"))
        lbls = list((data_dir / split / "labels").glob("*.txt"))
        status = "✅" if imgs else "❌"
        print(f"  {status} {split:6s}: {len(imgs):4d} images, {len(lbls):4d} labels")
        if not imgs:
            all_ok = False
    return all_ok


def load_all_labels(data_dir: Path, split: str) -> tuple[np.ndarray, list[int]]:
    """
    Загружает все YOLO-аннотации из директории сплита.

    Args:
        data_dir: Корневая директория датасета.
        split: Название сплита (train / valid / test).

    Returns:
        Кортеж (boxes array [N, 4], counts per image list).
    """
    boxes: list[list[float]] = []
    counts: list[int] = []
    lbl_dir = data_dir / split / "labels"

    for lf in sorted(lbl_dir.glob("*.txt")):
        lines = [ln.strip() for ln in lf.read_text().strip().split("\n") if ln.strip()]
        counts.append(len(lines))
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                boxes.append([float(p) for p in parts[1:5]])

    return np.array(boxes) if boxes else np.zeros((0, 4)), counts
