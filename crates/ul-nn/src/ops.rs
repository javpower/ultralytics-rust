//! 张量运算（对齐 ultralytics/utils/ops.py 与 utils/tal.py 的 tensor 函数）。

use tch::{Kind, Tensor};

/// nearest 上采样到指定 (h, w)。
pub fn up_nearest(x: &Tensor, size: [i64; 2]) -> Tensor {
    x.f_upsample_nearest2d(size, None, None).unwrap()
}

use ul_core::bbox::box_iou_xyxy;

/// `make_anchors`：各层特征图生成锚点中心 (2, A) 与 strides (1, A)。
pub fn make_anchors(feats: &[Tensor], strides: &[f64], grid_cell_offset: f64) -> (Tensor, Tensor) {
    let device = feats[0].device();
    let mut anchor_points: Vec<Tensor> = Vec::new();
    let mut anchor_strides: Vec<Tensor> = Vec::new();
    for (i, feat) in feats.iter().enumerate() {
        let (h, w) = (feat.size()[2], feat.size()[3]);
        let sx = (Tensor::arange(w, (Kind::Float, device)) + grid_cell_offset).reshape([1, w]);
        let sy = (Tensor::arange(h, (Kind::Float, device)) + grid_cell_offset).reshape([h, 1]);
        let sx = sx.expand([h, w], true);
        let sy = sy.expand([h, w], true);
        let stride = Tensor::full([1, h * w], strides[i], (Kind::Float, device));
        // 每点 (x, y) 成对（与官方 stack((sx, sy), -1).view(-1, 2).T 一致）
        let xy = Tensor::stack(&[sx.reshape([-1]), sy.reshape([-1])], 1);
        anchor_points.push(xy.transpose(0, 1));
        anchor_strides.push(stride);
    }
    (Tensor::cat(&anchor_points, 1), Tensor::cat(&anchor_strides, 1))
}

/// `dist2bbox`：锚点距离分布 → 框。distance (B, 4, A)（l/t/r/b 沿 dim）；
/// anchor_points (2, A)。xywh=true 输出中心格式，否则 xyxy。
pub fn dist2bbox(distance: &Tensor, anchor_points: &Tensor, xywh: bool, dim: i64) -> Tensor {
    let s = distance.split_with_sizes(&[1, 1, 1, 1], dim);
    let (l, t, r, b) = (&s[0], &s[1], &s[2], &s[3]);
    let lt = Tensor::cat(&[-l, -t], dim); // (B, 2, A)
    let rb = Tensor::cat(&[r, b], dim);
    let x1y1 = anchor_points.unsqueeze(0) + lt;
    let x2y2 = anchor_points.unsqueeze(0) + rb;
    if xywh {
        let cxy = (&x1y1 + &x2y2) / 2.0;
        let wh = (&x2y2 - &x1y1) / 2.0;
        Tensor::cat(&[cxy, wh], dim)
    } else {
        Tensor::cat(&[x1y1, x2y2], dim)
    }
}

/// `dist2rbox`：旋转框解码。pred_dist (B, 4, A)，pred_angle (B, 1, A)，
/// anchor_points (2, A)。官方语义（dim=1 布局）：
/// lt,rb = split(2)；xf,yf = ((rb−lt)/2).split(1)；xy = cat[xf·cos−yf·sin, xf·sin+yf·cos] + anchor；
/// 返回 (B, 4, A) = [cx, cy, w, h]，其中 w,h = lt+rb。
pub fn dist2rbox(pred_dist: &Tensor, pred_angle: &Tensor, anchor_points: &Tensor, dim: i64) -> Tensor {
    let halves = pred_dist.split_with_sizes(&[2, 2], dim);
    let (lt, rb) = (&halves[0], &halves[1]); // (B, 2, A)
    let diff = (rb - lt) / 2.0;
    let parts = diff.split_with_sizes(&[1, 1], 1);
    let (xf, yf) = (&parts[0], &parts[1]); // (B, 1, A)
    let cos = pred_angle.cos(); // (B, 1, A)
    let sin = pred_angle.sin();
    let x = xf * &cos - yf * &sin;
    let y = xf * &sin + yf * &cos;
    let xy = Tensor::cat(&[x, y], dim) + anchor_points; // (B,2,A) + (1,2,A)
    Tensor::cat(&[xy, lt + rb], dim)
}

/// `bbox_iou`（张量版，box1: (N,4) xyxy, box2: (M,4) → (N,M)）。
pub fn bbox_iou(box1: &Tensor, box2: &Tensor, giou: bool, diou: bool, ciou: bool, eps: f64) -> Tensor {
    let a_parts = box1.chunk(2, 1); // (N,2) xy1 xy2
    let (a1, a2) = (&a_parts[0], &a_parts[1]);
    let b_parts = box2.chunk(2, 1);
    let (b1, b2) = (&b_parts[0], &b_parts[1]);
    let a1 = a1.unsqueeze(1); // (N,1,2)
    let a2 = a2.unsqueeze(1);
    let b1 = b1.unsqueeze(0); // (1,M,2)
    let b2 = b2.unsqueeze(0);
    let inter = (a2.minimum(&b2) - a1.maximum(&b1)).clamp_min(0.0);
    let it0 = inter.select(2, 0);
    let it1 = inter.select(2, 1);
    let inter = it0 * it1;
    let wh_a = &a2 - &a1;
    let area_a = wh_a.select(2, 0) * wh_a.select(2, 1);
    let wh_b = &b2 - &b1;
    let area_b = wh_b.select(2, 0) * wh_b.select(2, 1);
    let union = &area_a + &area_b - &inter + eps;
    let mut iou = &inter / &union;
    if giou || diou || ciou {
        let cw = a2.maximum(&b2) - a1.minimum(&b1); // 包围盒 (N,M,2)
        let c_area = cw.select(2, 0) * cw.select(2, 1) + eps;
        if giou {
            iou = &iou - ((&c_area - &union) / &c_area);
        }
        if ciou || diou {
            let d = b1 + b2 - a1 - a2;
            let rho2 = (d.select(2, 0) * d.select(2, 0)) + (d.select(2, 1) * d.select(2, 1));
            iou = &iou - (rho2 / c_area);
        }
        if ciou {
            // 官方 v = (4/π²)(arctan(w2/h2) − arctan(w1/h1))²，其中 w/h 取各自分量
            let w1 = wh_a.select(1, 0).reshape([-1, 1]);
            let h1 = wh_a.select(1, 1).reshape([-1, 1]);
            let w2 = wh_b.select(1, 0).reshape([1, -1]);
            let h2 = wh_b.select(1, 1).reshape([1, -1]);
            let v = ((w2 / h2).atan() - (w1 / h1).atan()).pow_tensor_scalar(2.0)
                * (4.0 / std::f64::consts::PI.powi(2));
            let alpha = &v / ((1 - &iou) + &v);
            iou = &iou - (&v * alpha);
        }
    }
    iou
}

/// `xywh2xyxy`（张量，最后一维 4）。
pub fn xywh2xyxy(x: &Tensor) -> Tensor {
    let s = x.chunk(2, -1);
    let (xy, wh) = (&s[0], &s[1]);
    Tensor::cat(&[xy - wh / 2.0, xy + wh / 2.0], -1)
}

/// `clip_boxes`：裁剪到图像边界（boxes (N,4) xyxy，最后一维）。
pub fn clip_boxes(boxes: &Tensor, height: f64, width: f64) -> Tensor {
    let s = boxes.split_with_sizes(&[1, 1, 1, 1], -1);
    Tensor::cat(
        &[
            s[0].clamp(0.0, width),
            s[1].clamp(0.0, height),
            s[2].clamp(0.0, width),
            s[3].clamp(0.0, height),
        ],
        -1,
    )
}

/// `crop_mask`：bbox 外的掩码区域置 0。masks (B, N, H, W)，boxes (B, N, 4) 原图坐标。
pub fn crop_mask(masks: &Tensor, boxes: &Tensor) -> Tensor {
    let (b, n, h, w) = (masks.size()[0], masks.size()[1], masks.size()[2], masks.size()[3]);
    let bb = boxes.chunk(4, 2); // 各 (B, N, 1)
    let c = Tensor::arange(w, (Kind::Float, masks.device())).reshape([1, 1, 1, w]);
    let r = Tensor::arange(h, (Kind::Float, masks.device())).reshape([1, 1, h, 1]);
    let x1 = bb[0].reshape([b, n, 1, 1]);
    let y1 = bb[1].reshape([b, n, 1, 1]);
    let x2 = bb[2].reshape([b, n, 1, 1]);
    let y2 = bb[3].reshape([b, n, 1, 1]);
    let m = c
        .ge_tensor(&x1)
        .le_tensor(&x2)
        .logical_and(&r.ge_tensor(&y1).le_tensor(&y2))
        .to_kind(masks.kind());
    masks * m
}

/// `process_mask`：原型 × 系数 → 实例掩码（原型分辨率），带 bbox 裁剪。
pub fn process_mask(protos: &Tensor, masks_in: &Tensor, bboxes: &Tensor, img_shape: (i64, i64), upsample: bool) -> Tensor {
    let c = protos.size()[1];
    let (mh, mw) = (protos.size()[2], protos.size()[3]);
    let _ = img_shape;
    let b = protos.size()[0];
    let n = masks_in.size()[1];
    let protos_f = protos.reshape([b, c, mh * mw]);
    let masks = masks_in.matmul(&protos_f.transpose(1, 2)).sigmoid();
    let masks = crop_mask(&masks, bboxes);
    if upsample {
        up_nearest(&masks.reshape([b, n, mh, mw]), [mh * 4, mw * 4])
    } else {
        masks.reshape([b, n, mh, mw])
    }
}

/// 纯 CPU 贪心 NMS（类间偏移技巧，与 torchvision.ops.nms 语义一致）。
pub fn greedy_nms_cpu(
    boxes_xyxy: &[(f32, f32, f32, f32)],
    scores: &[f32],
    class_ids: &[i64],
    iou_thr: f32,
    max_wh: f32,
    agnostic: bool,
) -> Vec<usize> {
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<usize> = Vec::new();
    'outer: for &i in &idx {
        for &k in &keep {
            let same = agnostic || class_ids[k] == class_ids[i];
            if !same {
                continue;
            }
            let (off, offk) = if agnostic { (0.0, 0.0) } else { (class_ids[i] as f32 * max_wh, class_ids[k] as f32 * max_wh) };
            let bi = [boxes_xyxy[i].0 + off, boxes_xyxy[i].1 + off, boxes_xyxy[i].2 + off, boxes_xyxy[i].3 + off];
            let bk = [boxes_xyxy[k].0 + offk, boxes_xyxy[k].1 + offk, boxes_xyxy[k].2 + offk, boxes_xyxy[k].3 + offk];
            if box_iou_xyxy(&bi, &bk) > iou_thr {
                continue 'outer;
            }
        }
        keep.push(i);
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_anchors_shape() {
        let f = Tensor::zeros([1, 8, 8, 8], (Kind::Float, tch::Device::Cpu));
        let (pts, strides) = make_anchors(&[f], &[8.0], 0.5);
        assert_eq!(pts.size(), [2, 64]);
        assert_eq!(strides.size(), [1, 64]);
        let pts_v: Vec<f32> = Vec::try_from(&pts.reshape([-1])).unwrap();
        // (2, A)：行 0 是各点 x，行 1 是各点 y。8×8 图第一点 (0.5,0.5)，第二点 (1.5,0.5)
        assert!((pts_v[0] - 0.5).abs() < 1e-6 && (pts_v[1] - 1.5).abs() < 1e-6);
        assert!((pts_v[64] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dist2bbox_xyxy() {
        // 锚点在 (5.5, 5.5)，距离 l=t=1, r=b=3 → xyxy (4.5,4.5)-(8.5,8.5)
        let dist = Tensor::from_slice(&[1.0f32, 1.0, 3.0, 3.0]).reshape([1, 4, 1]);
        let anchor = Tensor::from_slice(&[5.5f32, 5.5]).reshape([2, 1]);
        let out = dist2bbox(&dist, &anchor, false, 1);
        let v: Vec<f32> = Vec::try_from(&out.reshape([-1])).unwrap();
        assert!((v[0] - 4.5).abs() < 1e-6 && (v[1] - 4.5).abs() < 1e-6);
        assert!((v[2] - 8.5).abs() < 1e-6 && (v[3] - 8.5).abs() < 1e-6);
    }

    #[test]
    fn nms_greedy() {
        let keep = greedy_nms_cpu(
            &[(0.0, 0.0, 10.0, 10.0), (1.0, 1.0, 11.0, 11.0), (20.0, 20.0, 30.0, 30.0)],
            &[0.9, 0.8, 0.7],
            &[0, 0, 0],
            0.5,
            512.0,
            false,
        );
        assert_eq!(keep, vec![0, 2]);
    }
}

/// (Tensor, Tensor) 浅克隆辅助。
pub trait ShallowPair {
    fn clone_pair(&self) -> Self;
}
impl ShallowPair for (Tensor, Tensor) {
    fn clone_pair(&self) -> Self {
        (self.0.shallow_clone(), self.1.shallow_clone())
    }
}
