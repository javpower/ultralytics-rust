# -*- coding: utf-8 -*-
# 官方 ultralytics 基线：同数据同配方（与 Rust y26s_coco120 一致），用于逐 epoch 对齐对照
from ultralytics import YOLO


def main():
    m = YOLO("E:/ultralytics-rust/weights/yolo26s.pt")
    m.train(data="E:/ultralytics-rust/wire-defect.yaml", epochs=40, imgsz=1280, batch=8,
            optimizer="AdamW", lr0=0.01, lrf=0.01, cos_lr=True, momentum=0.937,
            weight_decay=0.0005, warmup_epochs=3, degrees=10, translate=0.1, scale=0.5,
            fliplr=0.5, close_mosaic=15, seed=42, amp=True, cache="ram", workers=4,
            val=True, project="E:/ultralytics-rust/runs/official_baseline", name="baseline", exist_ok=True)


if __name__ == "__main__":
    main()
