//! # ultralytics-rust
//!
//! 纯 Rust 复刻的 Ultralytics YOLO 全栈：**训练 / 验证 / 推理 / 导出**，
//! 零 Python 依赖（深度学习底座为 libtorch 的 Rust 绑定 `tch`）。
//!
//! 支持家族：v3 / v5 / v8 / v9 / v10 / v11 / v12 / v26（detect、segment、pose、obb、classify），
//! 官方 48 个模型 YAML 内嵌于二进制，开箱即用；官方 `.pt` 权重可直接加载。
//!
//! ## 快速开始（库 API）
//!
//! ```no_run
//! use ultralytics_rust::prelude::*;
//!
//! fn main() -> anyhow::Result<()> {
//!     // 官方 COCO 预训练权重直接加载（需 libtorch，见 README）
//!     let mut model = YoloModel::build("yolo26n", Some(80), Device::Cpu, true)?;
//!
//!     // 训练（detect 示例；数据集为 YOLO 目录/YAML 格式）
//!     let cfg = TrainCfg { epochs: 100, imgsz: 640, ..Default::default() };
//!     let report = ul_engine::train(&mut model, std::path::Path::new("coco8.yaml"), &cfg)?;
//!     println!("mAP50-95 = {:.4}", report.final_metrics.map50_95);
//!     Ok(())
//! }
//! ```
//!
//! ## CLI
//!
//! ```text
//! cargo install ultralytics-rust
//! yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
//! yolo detect predict model=yolo26n.pt source=image.jpg
//! ```
//!
//! ## 依赖要求
//!
//! 构建与运行需要 libtorch 2.11（CPU 或 CUDA 12.8），通过环境变量 `LIBTORCH`
//! 指定路径；详见项目 README。

pub use ul_core;
pub use ul_data;
pub use ul_engine;
pub use ul_nn;

/// 常用类型与入口函数的一站式导入。
pub mod prelude {
    pub use ul_core::cfg;
    pub use ul_data::dataset::YoloDataset;
    pub use ul_core::metrics::DetMetrics;
    pub use ul_engine::{
        load_checkpoint, predict, save_checkpoint, train, validate_model, DetBatch, TrainCfg, TrainReport,
    };
    pub use ul_nn::model::YoloModel;
    pub use ul_nn::parse::Task;
}
