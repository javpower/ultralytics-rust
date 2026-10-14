//! ul-engine — 引擎层：损失、训练、验证、预测、导出。

pub mod engine;
pub mod loss;
pub mod nms;

pub use engine::{
    load_checkpoint, predict, save_checkpoint, train, validate_model, ModelEma, PredictOut,
    TrainCfg, TrainReport,
};
pub use loss::{classification_loss, v8_detection_loss, DetBatch, LossW};
pub use nms::{non_max_suppression_ex, NmsCfg};
