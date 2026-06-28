from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11n.pt")

    model.train(
        data="data/fruit_yolo11/data.yaml",
        imgsz=640,
        epochs=120,
        patience=25,
        batch=32,
        device=0,
        workers=4,
        cache=False,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        project="runs/detect",
        name="fruit_yolo11n",
    )


if __name__ == "__main__":
    main()
