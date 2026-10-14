//! 纯 CPU 非极大值抑制（贪心，同类或跨类，与 ultralytics `non_max_suppression`
//! 的最终贪心等价；tensor 批量版在 ul-nn 中实现）。

/// 单条检测（xyxy + 置信度 + 类别）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Det {
    pub xyxy: [f32; 4],
    pub score: f32,
    pub cls: i64,
}

/// 贪心 NMS。
///
/// * `agnostic` — true 时跨类抑制（ultralytics max_wh 偏移的等价语义）
pub fn nms(dets: &mut Vec<Det>, iou_thr: f32, agnostic: bool) {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<Det> = Vec::with_capacity(dets.len());
    'outer: for d in dets.drain(..) {
        for k in &keep {
            let same_group = agnostic || k.cls == d.cls;
            if same_group && crate::bbox::box_iou_xyxy(&k.xyxy, &d.xyxy) > iou_thr {
                continue 'outer;
            }
        }
        keep.push(d);
    }
    *dets = keep;
}

/// 过滤低分并截断 max_det。
pub fn filter(dets: &mut Vec<Det>, conf_thr: f32, max_det: usize) {
    dets.retain(|d| d.score >= conf_thr);
    if dets.len() > max_det {
        dets.truncate(max_det);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suppress_overlap() {
        let mut dets = vec![
            Det { xyxy: [0.0, 0.0, 10.0, 10.0], score: 0.9, cls: 0 },
            Det { xyxy: [1.0, 1.0, 11.0, 11.0], score: 0.8, cls: 0 },
            Det { xyxy: [20.0, 20.0, 30.0, 30.0], score: 0.7, cls: 0 },
        ];
        nms(&mut dets, 0.5, false);
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].score, 0.9);
        assert_eq!(dets[1].xyxy, [20.0, 20.0, 30.0, 30.0]);
    }

    #[test]
    fn class_aware() {
        let mut dets = vec![
            Det { xyxy: [0.0, 0.0, 10.0, 10.0], score: 0.9, cls: 0 },
            Det { xyxy: [0.0, 0.0, 10.0, 10.0], score: 0.8, cls: 1 },
        ];
        nms(&mut dets, 0.5, false);
        assert_eq!(dets.len(), 2);
        nms(&mut dets, 0.5, true);
        assert_eq!(dets.len(), 1);
    }
}
