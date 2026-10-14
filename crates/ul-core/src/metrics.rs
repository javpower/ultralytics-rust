//! COCO 协议评测：`compute_ap` / `ap_per_class` / `match_predictions` 忠实移植
//! （ultralytics/utils/metrics.py），纯 CPU，无张量依赖。

/// ultralytics `smooth(F1_curve.mean(0), 0.1)`：窗口 0.1*len 的盒式平滑。
fn smooth(y: &[f64], f: f64) -> Vec<f64> {
    let nf: usize = ((f * y.len() as f64) / 2.0).floor().max(1.0) as usize * 2 + 1;
    let half = nf / 2;
    let mut padded = vec![0.0f64; y.len() + nf - 1];
    for (i, v) in y.iter().enumerate() {
        padded[i + half] = *v;
    }
    // 边界复制填充（np.pad mode=edge）
    let n = padded.len();
    for i in 0..half {
        padded[i] = y[0];
        padded[n - 1 - i] = y[y.len() - 1];
    }
    let kernel = vec![1.0 / nf as f64; nf];
    let mut out = vec![0.0f64; y.len()];
    for i in 0..y.len() {
        let mut acc = 0.0;
        for k in 0..nf {
            acc += padded[i + k] * kernel[k];
        }
        out[i] = acc;
    }
    out
}

/// ultralytics `compute_ap`（101 点插值，COCO 协议）。
pub fn compute_ap(recall: &[f64], precision: &[f64]) -> f64 {
    if recall.is_empty() {
        return 0.0;
    }
    // mrec = [0.0, recall..., recall[-1], 1.0]; mpre = [1.0, precision..., 0.0, 0.0]
    let mut mrec = Vec::with_capacity(recall.len() + 3);
    mrec.push(0.0);
    mrec.extend_from_slice(recall);
    mrec.push(*recall.last().unwrap_or(&1.0));
    mrec.push(1.0);

    let mut mpre = Vec::with_capacity(precision.len() + 3);
    mpre.push(1.0);
    mpre.extend_from_slice(precision);
    mpre.push(0.0);
    mpre.push(0.0);

    // precision envelope：从右往左取累计最大
    for i in (0..mpre.len() - 1).rev() {
        mpre[i] = mpre[i].max(mpre[i + 1]);
    }

    // 101 点插值 + 梯形积分（np.trapz，与官方逐位一致）
    let mut ap = 0.0;
    let mut ys: Vec<f64> = Vec::with_capacity(101);
    for k in 0..=100 {
        let x = k as f64 / 100.0;
        ys.push(interp1(&mrec, &mpre, x));
    }
    for k in 0..100 {
        ap += 0.5 * (ys[k] + ys[k + 1]) * (1.0 / 100.0);
    }
    ap
}

/// np.interp 的 numpy 语义复刻：upper-bound 二分（重复结点取最后一段），
/// x == xp[-1]（含重复）返回 fp[-1]；x < xp[0] 返回 `left`（缺省 fp[0]）。
fn interp1(xp: &[f64], fp: &[f64], x: f64) -> f64 {
    interp1_left(xp, fp, x, fp[0])
}

fn interp1_left(xp: &[f64], fp: &[f64], x: f64, left: f64) -> f64 {
    let n = xp.len();
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        return left;
    }
    if lo == n {
        return fp[n - 1];
    }
    let a = lo - 1;
    let t = (x - xp[a]) / (xp[lo] - xp[a]);
    fp[a] * (1.0 - t) + fp[lo] * t
}

#[derive(Debug, Clone, Default)]
pub struct ApPerClass {
    /// 每类 max-F1 置信度点的 TP 计数。
    pub tp: Vec<f64>,
    pub fp: Vec<f64>,
    pub p: Vec<f64>,
    pub r: Vec<f64>,
    pub f1: Vec<f64>,
    /// (nc, 10) 每类 10 档 IoU 的 AP。
    pub ap: Vec<Vec<f64>>,
    /// 出现在 gt 里的类 id。
    pub unique_classes: Vec<i64>,
}

/// ultralytics `ap_per_class`。
///
/// * `tp` — (N, 10) 布尔矩阵：每个预测在各 IoU 阈值下是否 TP
/// * `conf` — (N,) 置信度
/// * `pred_cls` — (N,) 预测类
/// * `target_cls` — (M,) gt 类
pub fn ap_per_class(tp: &[Vec<bool>], conf: &[f32], pred_cls: &[i64], target_cls: &[i64]) -> ApPerClass {
    let eps = 1e-16f64;
    let n_iou = if tp.is_empty() { 10 } else { tp[0].len() };
    // 按 conf 降序排序
    let mut idx: Vec<usize> = (0..conf.len()).collect();
    idx.sort_by(|a, b| conf[*b].partial_cmp(&conf[*a]).unwrap_or(std::cmp::Ordering::Equal));

    let conf_s: Vec<f32> = idx.iter().map(|&i| conf[i]).collect();
    let pred_cls_s: Vec<i64> = idx.iter().map(|&i| pred_cls[i]).collect();
    let tp_s: Vec<Vec<bool>> = idx.iter().map(|&i| tp[i].clone()).collect();

    // gt 中的唯一类及计数
    let mut classes: Vec<i64> = target_cls.to_vec();
    classes.sort();
    classes.dedup();
    let nt: Vec<i64> = classes
        .iter()
        .map(|c| target_cls.iter().filter(|&&x| x == *c).count() as i64)
        .collect();
    let nc = classes.len();

    let mut out = ApPerClass {
        unique_classes: classes.clone(),
        ..Default::default()
    };
    out.ap = vec![vec![0.0; n_iou]; nc];

    // P/R 曲线（用于 max-F1 点）
    let mut p_curve = vec![0.0f64; nc];
    let mut r_curve = vec![0.0f64; nc];
    // 1000 点曲线（近似 ultralytics 的 p/r/f1 曲线，仅用于 max-F1 索引）
    let x: Vec<f64> = (0..1000).map(|i| i as f64 / 999.0).collect();
    let mut p_curves = vec![vec![0.0f64; 1000]; nc];
    let mut r_curves = vec![vec![0.0f64; 1000]; nc];

    for (ci, &c) in classes.iter().enumerate() {
        let sel: Vec<usize> = (0..pred_cls_s.len()).filter(|&i| pred_cls_s[i] == c).collect();
        let n_l = nt[ci] as f64;
        let n_p = sel.len();
        if n_p == 0 || n_l == 0.0 {
            continue;
        }
        // 累计 FP/TP
        let mut fpc = vec![0.0f64; n_p];
        let mut tpc = vec![0.0f64; n_p];
        let (mut fc, mut tc) = (0.0, 0.0);
        for k in 0..n_p {
            fc += if tp_s[sel[k]][0] { 0.0 } else { 1.0 };
            tc += if tp_s[sel[k]][0] { 1.0 } else { 0.0 };
            fpc[k] = fc;
            tpc[k] = tc;
        }
        // recall/precision 曲线（IoU=0.5 档，列 0）
        let recall: Vec<f64> = tpc.iter().map(|t| t / (n_l + eps)).collect();
        let precision: Vec<f64> = tpc
            .iter()
            .zip(fpc.iter())
            .map(|(t, f)| t / (t + f + eps))
            .collect();

        // np.interp(-x, -conf, recall, left=0)：conf 降序 → -conf 升序
        let conf_col: Vec<f64> = sel.iter().map(|&i| conf_s[i] as f64).collect();
        let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();
        let mut neg_conf_c: Vec<f64> = conf_col.clone();
        for v in neg_conf_c.iter_mut() {
            *v = -*v;
        }
        for k in 0..1000 {
            r_curves[ci][k] = interp1_left(&neg_conf_c, &recall, neg_x[k], 0.0);
            p_curves[ci][k] = interp1_left(&neg_conf_c, &precision, neg_x[k], 1.0);
        }

        // 10 档 AP
        for j in 0..n_iou {
            let (mut fc, mut tc) = (0.0, 0.0);
            let mut rec_j = vec![0.0f64; n_p];
            let mut pre_j = vec![0.0f64; n_p];
            for k in 0..n_p {
                fc += if tp_s[sel[k]][j] { 0.0 } else { 1.0 };
                tc += if tp_s[sel[k]][j] { 1.0 } else { 0.0 };
                rec_j[k] = tc / (n_l + eps);
                pre_j[k] = tc / (tc + fc + eps);
            }
            out.ap[ci][j] = compute_ap(&rec_j, &pre_j);
        }
        let _ = (p_curve.as_mut_slice(), r_curve.as_mut_slice());
    }

    // max-F1 点：f1 = 2pr/(p+r+eps) 在 1000 点曲线上，smooth 后取均值 argmax
    let mut f1_curves = vec![vec![0.0f64; 1000]; nc];
    for ci in 0..nc {
        for k in 0..1000 {
            let (p, r) = (p_curves[ci][k], r_curves[ci][k]);
            f1_curves[ci][k] = 2.0 * p * r / (p + r + eps);
        }
    }
    let mut mean_f1 = vec![0.0f64; 1000];
    for k in 0..1000 {
        mean_f1[k] = f1_curves.iter().map(|f| f[k]).sum::<f64>() / nc.max(1) as f64;
    }
    let sm = smooth(&mean_f1, 0.1);
    let best = sm
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    for ci in 0..nc {
        let (p, r) = (p_curves[ci][best], r_curves[ci][best]);
        let f1 = 2.0 * p * r / (p + r + eps);
        let tpc = r * nt[ci] as f64;
        let fpc = tpc / (p + eps) - tpc;
        out.p.push(p);
        out.r.push(r);
        out.f1.push(f1);
        out.tp.push(tpc.round());
        out.fp.push(fpc.round());
    }
    out
}

/// np.interp with `left` 填充（xp 升序）。
fn interp_left(xp: &[f64], fp: &[f64], x: f64, left: f64) -> f64 {
    if xp.is_empty() || x < xp[0] {
        return left;
    }
    interp1(xp, fp, x)
}

/// ultralytics `match_predictions`：返回 (N_pred, 10) 的 TP 矩阵。
///
/// * `iou` — N_pred × N_gt 的 IoU 矩阵（类别不同的对已被置零或调用方保证过滤）
/// * 类别匹配在内部判断：pred_cls == gt_cls 才可能成对。
pub fn match_predictions(pred_cls: &[i64], gt_cls: &[i64], iou: &[Vec<f32>], iouv: &[f32]) -> Vec<Vec<bool>> {
    let n_p = pred_cls.len();
    let mut correct = vec![vec![false; iouv.len()]; n_p];
    for (ti, &thr) in iouv.iter().enumerate() {
        // 候选配对：iou >= thr 且同类
        let mut matches: Vec<(f32, usize, usize)> = Vec::new(); // (iou, pred, gt)
        for p in 0..n_p {
            for g in 0..gt_cls.len() {
                let v = iou[p][g];
                if v >= thr && pred_cls[p] == gt_cls[g] {
                    matches.push((v, p, g));
                }
            }
        }
        if matches.is_empty() {
            continue;
        }
        // 按 IoU 降序
        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        // 每 gt 只留最高（unique by col）
        let mut seen_gt = std::collections::HashSet::new();
        let mut step1: Vec<(f32, usize, usize)> = Vec::new();
        for m in matches {
            if seen_gt.insert(m.2) {
                step1.push(m);
            }
        }
        // 每 pred 只留一个（unique by row，保持原顺序——ultralytics 用 return_index 保首现）
        let mut seen_pred = std::collections::HashSet::new();
        for m in step1 {
            if seen_pred.insert(m.1) {
                correct[m.1][ti] = true;
            }
        }
    }
    correct
}

/// 训练期累计的检测指标汇总（等价 ultralytics DetMetrics.results_dict）。
#[derive(Debug, Clone, Default)]
pub struct DetMetrics {
    pub map50_95: f64,
    pub map50: f64,
    pub map75: f64,
    pub mp: f64,
    pub mr: f64,
    pub nc_present: usize,
}

impl DetMetrics {
    pub fn from_ap(ap: &ApPerClass) -> Self {
        let n = ap.ap.len();
        let mean10 = |row: &[f64]| if row.is_empty() { 0.0 } else { row.iter().sum::<f64>() / row.len() as f64 };
        let map50_95 = if n == 0 {
            0.0
        } else {
            ap.ap.iter().map(|row| mean10(row)).sum::<f64>() / n as f64
        };
        let map50 = if n == 0 {
            0.0
        } else {
            ap.ap.iter().map(|row| row.first().copied().unwrap_or(0.0)).sum::<f64>() / n as f64
        };
        let map75 = if n == 0 {
            0.0
        } else {
            ap.ap.iter().map(|row| row.get(5).copied().unwrap_or(0.0)).sum::<f64>() / n as f64
        };
        DetMetrics {
            map50_95,
            map50,
            map75,
            mp: if ap.p.is_empty() { 0.0 } else { ap.p.iter().sum::<f64>() / ap.p.len() as f64 },
            mr: if ap.r.is_empty() { 0.0 } else { ap.r.iter().sum::<f64>() / ap.r.len() as f64 },
            nc_present: n,
        }
    }

    /// ultralytics fitness：0.9*mAP50:95 + 0.1*mAP50。
    pub fn fitness(&self) -> f64 {
        0.9 * self.map50_95 + 0.1 * self.map50
    }

    /// (P, R, mAP50, mAP50:95) —— ultralytics mean_results 顺序。
    pub fn mean_results(&self) -> [f64; 4] {
        [self.mp, self.mr, self.map50, self.map50_95]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_predictions() {
        // 单类单图：1 个 gt，1 个完美预测。官方 trapz 积分下完美 AP = 0.995
        let tp = vec![vec![true; 10]];
        let ap = ap_per_class(&tp, &[0.9], &[0], &[0]);
        assert!((ap.ap[0][0] - 0.995).abs() < 1e-6);
        let m = DetMetrics::from_ap(&ap);
        assert!((m.map50_95 - 0.995).abs() < 1e-6);
    }

    #[test]
    fn half_wrong() {
        // FP 分数高于 TP：AP@0.5 = 0.4975（trapz 端点半权）
        let tp = vec![vec![false; 10], vec![true; 10]];
        let ap = ap_per_class(&tp, &[0.9, 0.8], &[0, 0], &[0]);
        assert!((ap.ap[0][0] - 0.4975).abs() < 1e-6);
        // TP 在前 FP 在后：AP 不受损（0.995）
        let tp = vec![vec![true; 10], vec![false; 10]];
        let ap = ap_per_class(&tp, &[0.9, 0.8], &[0, 0], &[0]);
        assert!((ap.ap[0][0] - 0.995).abs() < 1e-6);
    }

    #[test]
    fn match_greedy() {
        let pred_cls = [0, 0];
        let gt_cls = [0];
        // 两个预测都和同一个 gt 高 IoU：只有分数序在前的那个（这里给相等 IoU 时按排序稳定性）拿 TP
        let iou = vec![vec![0.9], vec![0.95]];
        let c = match_predictions(&pred_cls, &gt_cls, &iou, &[0.5]);
        assert!(c[1][0]);
        assert!(!c[0][0]);
    }
}
