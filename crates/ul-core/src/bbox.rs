//! 框几何运算（纯 CPU，f32 切片）。
//!
//! 坐标约定与 ultralytics 一致：
//! - `xywh`：中心点 + 宽高（YOLO txt 标注、模型原始输出）
//! - `xyxy`：左上/右下（NMS、可视化、评测）

/// xywh → xyxy，`(x, y)` 为中心点。
pub fn xywh2xyxy(xywh: &[f32]) -> Vec<f32> {
    assert_eq!(xywh.len() % 4, 0);
    xywh.chunks_exact(4)
        .flat_map(|b| [b[0] - b[2] / 2.0, b[1] - b[3] / 2.0, b[0] + b[2] / 2.0, b[1] + b[3] / 2.0])
        .collect()
}

/// xyxy → xywh，`(x, y)` 为中心点。
pub fn xyxy2xywh(xyxy: &[f32]) -> Vec<f32> {
    assert_eq!(xyxy.len() % 4, 0);
    xyxy.chunks_exact(4)
        .flat_map(|b| [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, b[2] - b[0], b[3] - b[1]])
        .collect()
}

/// xywh → xyxy，`(x, y)` 为左上角（YOLO txt 标注存储格式）。
pub fn xywhn2xyxy_topleft(xywh: &[f32], w: f32, h: f32) -> Vec<f32> {
    xywh.chunks_exact(4)
        .flat_map(|b| [b[0] * w, b[1] * h, (b[0] + b[2]) * w, (b[1] + b[3]) * h])
        .collect()
}

/// 单对 xyxy 框的 IoU（与 ultralytics `box_iou` 一致，eps 防零除）。
pub fn box_iou_xyxy(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_x = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let inter_y = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = inter_x * inter_y;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    if inter <= 0.0 {
        return 0.0;
    }
    inter / (area_a + area_b - inter + 1e-7)
}

/// 缩放框：从模型输入尺寸坐标映射回原图（`scale_boxes` 语义，gain + pad）。
pub fn scale_boxes_xyxy(box_xyxy: &mut [f32], gain_w: f32, gain_h: f32, pad_w: f32, pad_h: f32, orig_w: f32, orig_h: f32) {
    for b in box_xyxy.chunks_exact_mut(4) {
        b[0] = (b[0] - pad_w) / gain_w;
        b[1] = (b[1] - pad_h) / gain_h;
        b[2] = (b[2] - pad_w) / gain_w;
        b[3] = (b[3] - pad_h) / gain_h;
        b[0] = b[0].clamp(0.0, orig_w);
        b[1] = b[1].clamp(0.0, orig_h);
        b[2] = b[2].clamp(0.0, orig_w);
        b[3] = b[3].clamp(0.0, orig_h);
    }
}

/// letterbox 几何参数计算（缩放系数 + 双向 padding，114 灰边填充由图像层负责）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LetterboxParams {
    pub gain_w: f32,
    pub gain_h: f32,
    pub pad_w: f32,
    pub pad_h: f32,
}

impl LetterboxParams {
    /// ultralytics `Letterbox` 默认 center=True 语义。
    pub fn new(orig_w: f32, orig_h: f32, new_w: f32, new_h: f32) -> Self {
        let gain = (new_w / orig_w).min(new_h / orig_h);
        let (gain_w, gain_h) = (gain, gain);
        let pad_w = (new_w - orig_w * gain) * 0.5;
        let pad_h = (new_h - orig_h * gain) * 0.5;
        LetterboxParams { gain_w, gain_h, pad_w, pad_h }
    }

    /// xyxy 原图框 → letterbox 后的模型输入框。
    pub fn apply_xyxy(&self, b: &mut [f32; 4]) {
        b[0] = b[0] * self.gain_w + self.pad_w;
        b[1] = b[1] * self.gain_h + self.pad_h;
        b[2] = b[2] * self.gain_w + self.pad_w;
        b[3] = b[3] * self.gain_h + self.pad_h;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_basic() {
        let iou = box_iou_xyxy(&[0.0, 0.0, 10.0, 10.0], &[0.0, 0.0, 10.0, 10.0]);
        assert!((iou - 1.0).abs() < 1e-6);
        let iou = box_iou_xyxy(&[0.0, 0.0, 10.0, 10.0], &[5.0, 0.0, 15.0, 10.0]);
        assert!((iou - 50.0 / 150.0).abs() < 1e-6);
        let iou = box_iou_xyxy(&[0.0, 0.0, 1.0, 1.0], &[2.0, 2.0, 3.0, 3.0]);
        assert!(iou < 1e-6);
    }

    #[test]
    fn letterbox_params() {
        let p = LetterboxParams::new(640.0, 480.0, 640.0, 640.0);
        assert!((p.gain_w - 1.0).abs() < 1e-6);
        assert!((p.pad_h - 80.0).abs() < 1e-6);
        let mut b = [0.0, 0.0, 640.0, 480.0];
        p.apply_xyxy(&mut b);
        assert_eq!(b, [0.0, 80.0, 640.0, 560.0]);
    }
}
