# Веса модели

Обученные веса модели YOLOv8x хранятся на Google Drive
из-за большого размера файла (~270 MB).

## Скачать веса

| Файл | Описание | Ссылка |
|------|----------|--------|
| `best.pt` | Лучшие веса по mAP50 за всё обучение (~70 эпох) | [Google Drive](https://drive.google.com/drive/folders/ВАША_ССЫЛКА) |
| `last.pt` | Веса последней эпохи | [Google Drive](https://drive.google.com/drive/folders/ВАША_ССЫЛКА) |

После скачивания положите файлы в эту папку:

```
artifacts/weights/best.pt
artifacts/weights/last.pt
```

## Использование весов

```python
from ultralytics import YOLO

# Загрузка модели
model = YOLO("artifacts/weights/best.pt")

# Инференс на одном изображении
results = model.predict("your_image.jpg", imgsz=960, conf=0.25)
results[0].show()

# Инференс с сохранением результатов
results = model.predict("your_image.jpg", imgsz=960, conf=0.25, save=True)
```

## Параметры обучения

| Параметр | Значение |
|----------|----------|
| Архитектура | YOLOv8x |
| Датасет | HRPlanes (18,477 аннотаций, 3,101 изображений) |
| Эпох | ~70 (прерванное обучение: ~40 + 30) |
| imgsz | 960×960 |
| batch | 8 (T4 GPU, OOM при 16) |
| Оптимизатор | SGD (lr=0.001 → 0.0005) |
| Аугментация | HSV(h=0.015, s=0.7, v=0.4) + Mosaic + Flip |

## Финальные метрики

| Split | F1 | Precision | Recall | mAP50 | mAP50-95 |
|-------|----|-----------|--------|-------|----------|
| Val | 0.9924 | 0.9912 | 0.9936 | 0.9931 | 0.8834 |
| **Test** | **0.9906** | **0.9932** | **0.9879** | **0.9943** | **0.8897** |
| Статья | 0.9932 | 0.9915 | 0.9950 | 0.9939 | 0.8990 |
