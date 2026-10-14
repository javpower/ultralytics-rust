//! ul-core — ultralytics-rust 的纯逻辑核心层。
//!
//! 与 tch 无依赖：错误类型、几何/框运算、模型 YAML 解析（嵌入官方 cfg）、
//! COCO 协议评测（ap_per_class / match_predictions 忠实移植）、纯 CPU NMS。

pub mod bbox;
pub mod cfg;
pub mod error;
pub mod metrics;
pub mod nms;

pub use error::{Result, UlError};

/// COCO 评测 IoU 阈值：0.50:0.05:0.95 共 10 档（与 ultralytics `iouv` 一致）。
pub const IOUV: [f32; 10] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95];
