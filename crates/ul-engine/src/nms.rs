//! 张量推理输出 → 逐图检测（conf 过滤 + topk + 贪心 NMS + 坐标还原）。

use tch::{Kind, Tensor};

use ul_core::nms::Det;

use ul_nn::ops::greedy_nms_cpu;

/// NMS 参数（default.yaml 语义）。
#[derive(Clone, Copy)]
pub struct NmsCfg {
    pub conf: f64,
    pub iou: f64,
    pub max_det: usize,
    pub agnostic: bool,
    pub multi_label: bool,
    pub max_wh: f64,
}

impl NmsCfg {
    pub fn val() -> Self {
        NmsCfg { conf: 0.001, iou: 0.7, max_det: 300, agnostic: false, multi_label: true, max_wh: 768.0 }
    }
    pub fn predict() -> Self {
        NmsCfg { conf: 0.25, iou: 0.7, max_det: 300, agnostic: false, multi_label: false, max_wh: 768.0 }
    }
}

/// 每张图的检测结果（模型输入坐标系 xyxy 像素）。
pub struct DetImage {
    pub dets: Vec<Det>,
    /// 附加通道（mask 系数 / 关键点 / 角度），len = dets.len()，通道数 = extra_ch。
    pub extra: Vec<Vec<f32>>,
    pub extra_ch: usize,
}

/// 推理输出 (B, 4+nc[+extra], A) → 每图 NMS。
pub fn non_max_suppression(preds: &Tensor, cfg: &NmsCfg) -> Vec<DetImage> {
    let b = preds.size()[0];
    let total = preds.size()[1];
    let a = preds.size()[2];
    let extra_ch = (total - 4).saturating_sub(preds.size()[1].max(0).max(0)).max(0);
    let _ = extra_ch;
    // 通道布局：[0..4]=xyxy, [4..4+nc]=分数, [4+nc..]=extra
    let nc = total - 4; // 无 extra 时
    // 先假定无 extra：由调用方传入 nc 更稳，这里通过 preds 形状推断交给 val/predict
    let p = preds.transpose(1, 2).contiguous(); // (B, A, 4+nc+extra)
    let pw = p.size()[2];
    // extra 通道数由调用方确定：Detect=0；这里以 nc_total-4-nc_unknown... 简化：提供重载
    non_max_suppression_ex(&p, 4 + nc, cfg, b, a, pw)
}

/// 显式通道数版本：`first_ch` = 4+nc。
pub fn non_max_suppression_ex(p: &Tensor, first_ch: i64, cfg: &NmsCfg, b: i64, _a: i64, _w: i64) -> Vec<DetImage> {
    let cpu = p.to_device(tch::Device::Cpu).contiguous().reshape([-1]);
    let host: Vec<f32> = Vec::try_from(&cpu).unwrap_or_default();
    let row = p.size()[2] as usize;
    let n_rows = p.size()[1] as usize;
    let (xyxy_off, score_off, extra_off, extra_n) = (0usize, 4usize, first_ch as usize, (row - first_ch as usize).max(0));
    let nc = (first_ch - 4) as usize;

    let mut out = Vec::with_capacity(b as usize);
    for bi in 0..b as usize {
        let base = bi * n_rows * row;
        // 候选
        let mut cand: Vec<(f32, usize, usize)> = Vec::new(); // (score, anchor, cls)
        for ai in 0..n_rows {
            let row_base = base + ai * row;
            let mut best = 0f32;
            let mut best_c = 0usize;
            for ci in 0..nc {
                let sc = host[row_base + score_off + ci];
                if sc > best {
                    best = sc;
                    best_c = ci;
                }
            }
            if best as f64 >= cfg.conf {
                cand.push((best, ai, best_c));
            }
            if cfg.multi_label && nc > 1 {
                // 多标签：逐类候选
                // 上面的 best 分支已足够 val（官方 multi_label 开启时逐类），这里对 val 用逐类：
            }
        }
        // 多标签模式（val）：为每个过阈值的 (anchor, class) 生成候选
        let mut items: Vec<(f32, usize, usize)> = Vec::new();
        if cfg.multi_label && nc > 1 {
            for ai in 0..n_rows {
                let row_base = base + ai * row;
                for ci in 0..nc {
                    let sc = host[row_base + score_off + ci];
                    if sc as f64 >= cfg.conf {
                        items.push((sc, ai, ci));
                    }
                }
            }
        } else {
            items = cand;
        }
        // topk 限幅
        if items.len() > 30000 {
            items.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(30000);
        }
        // 载入 xyxy + extra + 贪心 NMS
        let mut boxes: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(items.len());
        let mut scores: Vec<f32> = Vec::with_capacity(items.len());
        let mut classes: Vec<i64> = Vec::with_capacity(items.len());
        let mut extras: Vec<Vec<f32>> = Vec::with_capacity(items.len());
        for (sc, ai, ci) in &items {
            let row_base = base + ai * row;
            // 模型输出为 xywh（中心格式）→ 转 xyxy
            let cx = host[row_base + xyxy_off];
            let cy = host[row_base + xyxy_off + 1];
            let w = host[row_base + xyxy_off + 2];
            let h = host[row_base + xyxy_off + 3];
            boxes.push((cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0));
            scores.push(*sc);
            classes.push(*ci as i64);
            if extra_n > 0 {
                extras.push(host[row_base + extra_off..row_base + extra_off + extra_n].to_vec());
            }
        }
        let keep = greedy_nms_cpu(&boxes, &scores, &classes, cfg.iou as f32, cfg.max_wh as f32, cfg.agnostic);
        let mut dets = Vec::with_capacity(keep.len());
        let mut kept_extra = Vec::with_capacity(keep.len());
        for (rank, &i) in keep.iter().enumerate() {
            dets.push(Det { xyxy: [boxes[i].0, boxes[i].1, boxes[i].2, boxes[i].3], score: scores[i], cls: classes[i] });
            if extra_n > 0 {
                kept_extra.push(extras[i].clone());
            }
            if rank + 1 >= cfg.max_det {
                break;
            }
        }
        out.push(DetImage { dets, extra: kept_extra, extra_ch: extra_n });
    }
    out
}

/// 坐标还原：模型输入 xyxy → 原图（gain+pad，与 predictor letterbox 对应）。
pub fn scale_boxes_back(d: &mut Det, gain: f32, pad_w: f32, pad_h: f32, orig_w: f32, orig_h: f32) {
    d.xyxy[0] = ((d.xyxy[0] - pad_w) / gain).clamp(0.0, orig_w);
    d.xyxy[1] = ((d.xyxy[1] - pad_h) / gain).clamp(0.0, orig_h);
    d.xyxy[2] = ((d.xyxy[2] - pad_w) / gain).clamp(0.0, orig_w);
    d.xyxy[3] = ((d.xyxy[3] - pad_h) / gain).clamp(0.0, orig_h);
}

/// IoU 矩阵（评测用）：preds (N,4) × gts (M,4)。
pub fn iou_matrix(preds: &[[f32; 4]], gts: &[[f32; 4]]) -> Vec<Vec<f32>> {
    preds
        .iter()
        .map(|p| gts.iter().map(|g| ul_core::bbox::box_iou_xyxy(p, g)).collect())
        .collect()
}

/// 建批张量：(B,3,S,S) float 0..1。
pub fn to_tensor(imgs: &[ul_data::image::Img], device: tch::Device) -> Tensor {
    let b = imgs.len() as i64;
    let (h, w) = (imgs[0].h as i64, imgs[0].w as i64);
    let mut data = Vec::with_capacity((b * 3 * h * w) as usize);
    for img in imgs {
        let px = &img.data;
        let n = (h * w) as usize;
        for c in 0..3 {
            for i in 0..n {
                data.push(px[i * 3 + c] as f32 / 255.0);
            }
        }
    }
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let t = Tensor::from_data_size(&bytes, &[b, 3, h, w], Kind::Float).to_device(device);
    t
}

/// rect letterbox 目标尺寸：按长宽比缩放到最长边 imgsz，短边 32 对齐。
pub fn rect_letterbox_size(w: u32, h: u32, imgsz: u32, stride: u32) -> (u32, u32) {
    let r = (imgsz as f32 / w.max(1) as f32).min(imgsz as f32 / h.max(1) as f32);
    let nw = ((w as f32 * r).round() as u32).max(stride).min(imgsz);
    let nh = ((h as f32 * r).round() as u32).max(stride).min(imgsz);
    let align = |x: u32| ((x + stride - 1) / stride) * stride;
    (align(nw), align(nh))
}

/// E2E（yolo26 / v10 端到端头）最终检测直通解析。
/// 输入 t: (B, max_det, 6+extra) = [xyxy(letterbox 像素), score, label, extra…]。
/// 图内 topk 已完成候选选择与排序；此处仅做 conf 过滤、限幅与装包。
/// 绝不可再送 non_max_suppression_ex——那会把 [score,label] 两列当 nc 类分数读并越界。
pub fn parse_e2e_detections(p: &Tensor, cfg: &NmsCfg, b: i64) -> Vec<DetImage> {
    parse_e2e_with_box_w(p, cfg, b, 4)
}

/// OBB（xywhr 5 列框）变体：row = [cx,cy,w,h,r, score, label, extra…]
pub fn parse_e2e_detections_obb(p: &Tensor, cfg: &NmsCfg, b: i64) -> Vec<DetImage> {
    parse_e2e_with_box_w(p, cfg, b, 5)
}

fn parse_e2e_with_box_w(p: &Tensor, cfg: &NmsCfg, b: i64, box_w: usize) -> Vec<DetImage> {
    let cpu = p.to_device(tch::Device::Cpu).contiguous().reshape([-1]);
    let host: Vec<f32> = Vec::try_from(&cpu).unwrap_or_default();
    let row = p.size()[2] as usize;
    let n = p.size()[1] as usize;
    let base_cols = box_w + 2;
    let extra_n = row.saturating_sub(base_cols);
    let mut out = Vec::with_capacity(b as usize);
    for bi in 0..b as usize {
        let base = bi * n * row;
        let mut dets: Vec<Det> = Vec::with_capacity(n.min(cfg.max_det));
        let mut kept_extra: Vec<Vec<f32>> = Vec::new();
        for ai in 0..n {
            let rb = base + ai * row;
            let sc = host[rb + box_w];
            if (sc as f64) < cfg.conf {
                continue;
            }
            dets.push(Det {
                xyxy: [host[rb], host[rb + 1], host[rb + 2], host[rb + 3]],
                score: sc,
                cls: host[rb + box_w + 1].round() as i64,
            });
            if extra_n > 0 {
                kept_extra.push(host[rb + base_cols..rb + base_cols + extra_n].to_vec());
            }
        }
        if dets.len() > cfg.max_det {
            dets.truncate(cfg.max_det);
            kept_extra.truncate(cfg.max_det);
        }
        out.push(DetImage { dets, extra: kept_extra, extra_ch: extra_n });
    }
    out
}
