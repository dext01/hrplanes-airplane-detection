"""Вспомогательные утилиты: визуализация, сохранение, EDA."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def save_metrics(results_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Сохраняет DataFrame с метриками в CSV.

    Args:
        results_df: DataFrame с метриками.
        save_dir: Директория для сохранения.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "results_final.csv"
    results_df.to_csv(path, index=False)
    print(f"✅ Метрики сохранены: {path}")


def plot_eda(
    train_boxes: np.ndarray,
    val_boxes: np.ndarray,
    test_boxes: np.ndarray,
    all_counts: list[int],
    save_dir: Path,
) -> None:
    """
    Строит EDA-графики датасета.

    Args:
        train_boxes: Bounding boxes тренировочного сплита.
        val_boxes: Bounding boxes валидационного сплита.
        test_boxes: Bounding boxes тестового сплита.
        all_counts: Список количества объектов на изображение.
        save_dir: Директория для сохранения.
    """
    all_boxes = np.vstack([train_boxes, val_boxes, test_boxes])
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("HRPlanes Dataset — EDA", fontsize=16, fontweight="bold")

    # Pie chart распределения
    split_sizes = [len(train_boxes), len(val_boxes), len(test_boxes)]
    split_labels = [
        f"Train\n{len(train_boxes):,}",
        f"Val\n{len(val_boxes):,}",
        f"Test\n{len(test_boxes):,}",
    ]
    axes[0, 0].pie(split_sizes, labels=split_labels, autopct="%1.1f%%",
                   colors=colors[:3], startangle=90)
    axes[0, 0].set_title("Распределение аннотаций")

    # Гистограмма ширины BBox
    axes[0, 1].hist(all_boxes[:, 2], bins=60, color=colors[0], alpha=0.85, edgecolor="white")
    axes[0, 1].axvline(np.mean(all_boxes[:, 2]), color="red", linestyle="--",
                       label=f"mean={np.mean(all_boxes[:, 2]):.3f}")
    axes[0, 1].set_title("Ширина BBox")
    axes[0, 1].set_xlabel("width (норм.)")
    axes[0, 1].set_ylabel("Кол-во")
    axes[0, 1].legend()

    # Гистограмма высоты BBox
    axes[0, 2].hist(all_boxes[:, 3], bins=60, color=colors[1], alpha=0.85, edgecolor="white")
    axes[0, 2].axvline(np.mean(all_boxes[:, 3]), color="red", linestyle="--",
                       label=f"mean={np.mean(all_boxes[:, 3]):.3f}")
    axes[0, 2].set_title("Высота BBox")
    axes[0, 2].set_xlabel("height (норм.)")
    axes[0, 2].set_ylabel("Кол-во")
    axes[0, 2].legend()

    # Scatter центров BBox
    idx = np.random.choice(len(all_boxes), min(4000, len(all_boxes)), replace=False)
    axes[1, 0].scatter(all_boxes[idx, 0], all_boxes[idx, 1], alpha=0.15, s=3, c=colors[4])
    axes[1, 0].set_title("Распределение центров BBox")
    axes[1, 0].set_xlabel("cx")
    axes[1, 0].set_ylabel("cy")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)

    # Самолётов на изображение
    axes[1, 1].hist(all_counts, bins=40, color=colors[2], alpha=0.85, edgecolor="white")
    axes[1, 1].axvline(np.mean(all_counts), color="red", linestyle="--",
                       label=f"mean={np.mean(all_counts):.1f}")
    axes[1, 1].set_title("Самолётов на изображение")
    axes[1, 1].set_xlabel("Кол-во самолётов")
    axes[1, 1].set_ylabel("Кол-во изображений")
    axes[1, 1].legend()

    # Площадь BBox
    areas = all_boxes[:, 2] * all_boxes[:, 3]
    axes[1, 2].hist(areas, bins=60, color=colors[3], alpha=0.85, edgecolor="white")
    axes[1, 2].axvline(np.mean(areas), color="navy", linestyle="--",
                       label=f"mean={np.mean(areas):.5f}")
    axes[1, 2].set_title("Площадь BBox")
    axes[1, 2].set_xlabel("area (w×h)")
    axes[1, 2].legend()

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "eda_hrplanes.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ EDA сохранён: {path}")


def plot_val_test_metrics(results_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Строит bar-chart метрик Val vs Test.

    Args:
        results_df: DataFrame с колонками Split | F1 | Precision | Recall | mAP50 | mAP50-95.
        save_dir: Директория для сохранения.
    """
    colors = ["#2196F3", "#4CAF50"]
    metrics_cols = ["F1", "Precision", "Recall", "mAP50", "mAP50-95"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle("YOLOv8x SGD Aug 960px — Val / Test метрики",
                 fontsize=14, fontweight="bold")

    for ax, col in zip(axes, metrics_cols):
        bars = ax.bar(
            results_df["Split"],
            results_df[col],
            color=colors,
            alpha=0.85,
            edgecolor="white",
            width=0.5,
        )
        ax.bar_label(bars, fmt="%.4f", fontsize=10, padding=3)
        ax.set_title(col, fontweight="bold")
        ax.set_ylim(max(0, results_df[col].min() - 0.03), 1.01)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "metrics_val_test.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ График метрик сохранён: {path}")


def plot_comparison_vs_paper(results_df: pd.DataFrame, save_dir: Path) -> None:
    """
    Строит сравнительный bar-chart наших результатов vs результаты статьи.

    Args:
        results_df: DataFrame с метриками (Split=Test).
        save_dir: Директория для сохранения.
    """
    paper = {"F1": 0.9932, "mAP50": 0.9939, "mAP50-95": 0.8990}
    test_row = results_df[results_df["Split"] == "Test"].iloc[0]
    our = {m: float(test_row[m]) for m in ["F1", "mAP50", "mAP50-95"]}

    metrics_show = ["F1", "mAP50", "mAP50-95"]
    x = np.arange(len(metrics_show))
    w = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, [paper[m] for m in metrics_show], w,
           label="Статья (Exp12, YOLOv8x SGD)", color="#FF5722", alpha=0.85)
    ax.bar(x + w / 2, [our[m] for m in metrics_show], w,
           label="Наш результат (Test)", color="#2196F3", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_show, fontsize=12)
    ax.set_ylim(0.85, 1.005)
    ax.set_title("YOLOv8x SGD Aug 960px — наши результаты vs статья",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, m in enumerate(metrics_show):
        ax.text(i - w / 2, paper[m] + 0.002, f"{paper[m]:.4f}", ha="center", fontsize=9)
        ax.text(i + w / 2, our[m] + 0.002, f"{our[m]:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "comparison_vs_paper.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ График сравнения сохранён: {path}")


def plot_training_curves(curves_csv: Path, save_dir: Path) -> None:
    """
    Строит кривые обучения из файла results.csv Ultralytics.

    Args:
        curves_csv: Путь к results.csv из директории обучения.
        save_dir: Директория для сохранения графика.
    """
    if not curves_csv.exists():
        print(f"⚠️  {curves_csv} не найден, пропускаем кривые обучения.")
        return

    curves = pd.read_csv(curves_csv)
    curves.columns = curves.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Кривые обучения — YOLOv8x SGD Aug 960px",
                 fontsize=13, fontweight="bold")

    def safe_plot(ax, col_train: str, col_val: str, title: str) -> None:
        if col_train in curves.columns:
            ax.plot(curves["epoch"], curves[col_train],
                    label="Train", color="#2196F3", linewidth=1.5)
        if col_val in curves.columns:
            ax.plot(curves["epoch"], curves[col_val],
                    label="Val", color="#FF5722", linewidth=1.5, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    safe_plot(axes[0, 0], "train/box_loss", "val/box_loss", "Box Loss")
    safe_plot(axes[0, 1], "train/cls_loss", "val/cls_loss", "Class Loss")
    safe_plot(axes[0, 2], "train/dfl_loss", "val/dfl_loss", "DFL Loss")

    for col, label, color in [
        ("metrics/precision(B)", "Precision", "#4CAF50"),
        ("metrics/recall(B)", "Recall", "#FF9800"),
    ]:
        if col in curves.columns:
            axes[1, 0].plot(curves["epoch"], curves[col],
                            label=label, color=color, linewidth=1.5)
    axes[1, 0].set_title("Precision & Recall", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    for col, label, color in [
        ("metrics/mAP50(B)", "mAP50", "#2196F3"),
        ("metrics/mAP50-95(B)", "mAP50-95", "#9C27B0"),
    ]:
        if col in curves.columns:
            axes[1, 1].plot(curves["epoch"], curves[col],
                            label=label, color=color, linewidth=1.5)
    axes[1, 1].set_title("mAP", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    if "metrics/precision(B)" in curves.columns and "metrics/recall(B)" in curves.columns:
        p = curves["metrics/precision(B)"]
        r = curves["metrics/recall(B)"]
        f1 = 2 * p * r / (p + r + 1e-9)
        axes[1, 2].plot(curves["epoch"], f1, color="#F44336", linewidth=1.5)
        axes[1, 2].set_title("F1 Score", fontweight="bold")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].grid(alpha=0.3)
    else:
        axes[1, 2].axis("off")

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "training_curves.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✅ Кривые обучения сохранены: {path}")


def visualize_samples(data_dir: Path, split: str, save_dir: Path, n: int = 4) -> None:
    """
    Визуализирует примеры изображений с аннотациями bounding box.

    Args:
        data_dir: Корневая директория датасета.
        split: Название сплита (train / valid / test).
        save_dir: Директория для сохранения.
        n: Количество примеров для отображения.
    """
    img_dir = data_dir / split / "images"
    lbl_dir = data_dir / split / "labels"
    img_paths = sorted(img_dir.glob("*.jpg"))[:n]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle(f"Примеры: {split}", fontsize=14, fontweight="bold")

    for ax, ip in zip(axes, img_paths):
        img = np.array(Image.open(ip))
        h, w = img.shape[:2]
        lp = lbl_dir / (ip.stem + ".txt")
        ax.imshow(img)

        if lp.exists():
            for line in lp.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                rect = mpatches.Rectangle(
                    (x1, y1),
                    bw * w,
                    bh * h,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                )
                ax.add_patch(rect)

        n_planes = (
            len([ln for ln in lp.read_text().strip().split("\n") if ln.strip()])
            if lp.exists()
            else 0
        )
        ax.set_title(f"{ip.stem[:18]}\n({n_planes} самолётов)", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"samples_{split}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✅ Примеры сохранены: {path}")
