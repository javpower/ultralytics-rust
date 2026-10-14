//! ul-data — 数据层：图像 IO、YOLO 数据集（detect/seg/pose/obb）、增强管线。

pub mod augment;
pub mod dataset;
pub mod image;

pub use dataset::{load_dataset_yaml, DetInstance, Sample, YoloDataset};
