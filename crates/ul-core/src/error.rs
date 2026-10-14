//! 统一错误类型。

use thiserror::Error;

#[derive(Error, Debug)]
pub enum UlError {
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tch: {0}")]
    Tch(#[from] tch::TchError),
    #[error("{0}")]
    Msg(String),
}

impl UlError {
    pub fn msg(m: impl Into<String>) -> Self {
        UlError::Msg(m.into())
    }
}

pub type Result<T> = std::result::Result<T, UlError>;

/// 便捷宏：返回带位置的 UlError::Msg。
#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::UlError::Msg(format!($($arg)*))) };
}

#[macro_export]
macro_rules! ensure {
    ($cond:expr, $($arg:tt)*) => { if !($cond) { return Err($crate::UlError::Msg(format!($($arg)*))); } };
}
