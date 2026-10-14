//! ul-nn — 张量层：模块动物园、parse_model、.pt 加载、TAL、ONNX 导出。

pub mod attention;
pub mod block;
pub mod conv;
pub mod head;
pub mod model;
pub mod module;
pub mod ops;
pub mod parse;
pub mod pt;
pub mod special;
pub mod tal;

pub use model::{ModelOutput, YoloModel};
pub use parse::Task;

use ul_core::UlError;

pub type Result<T> = std::result::Result<T, UlError>;
