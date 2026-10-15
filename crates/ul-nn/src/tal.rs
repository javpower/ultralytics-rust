//! TaskAlignedAssigner —— 对齐 ultralytics/utils/tal.py（8.4.x）。
//!
//! 布局约定（与官方一致）：align/overlaps/mask 均为 (bs, m, A)（gt 主序）。

use tch::{Kind, Tensor};

use crate::ops::bbox_iou_ciou_batched;
#[allow(unused_imports)]
use crate::ops::bbox_iou_pairwise;

/// TAL 分配器。
pub struct TaskAlignedAssigner {
    pub topk: i64,
    pub num_classes: i64,
    pub alpha: f64,
    pub beta: f64,
    pub eps: f64,
    /// 小目标框最小边（stride[1]，官方 8.4 语义；0 = 关闭）。
    pub stride_val: f64,
}

impl Default for TaskAlignedAssigner {
    fn default() -> Self {
        TaskAlignedAssigner { topk: 10, num_classes: 80, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 }
    }
}

/// 分配结果。
pub struct AssignOut {
    pub target_labels: Tensor,  // (bs, A)
    pub target_bboxes: Tensor,  // (bs, A, 4) xyxy 像素
    pub target_scores: Tensor,  // (bs, A, nc)
    pub fg_mask: Tensor,        // (bs, A) bool
    pub target_gt_idx: Tensor,  // (bs, A) i64
}

impl TaskAlignedAssigner {
    /// 旋转框分配（官方 RotatedTaskAlignedAssigner）：pd/gt 均为 xywhr 像素 (bs, A/m, 5)。
    pub fn assign_rotated(
        &self,
        pd_scores: &Tensor,
        pd_xywhr: &Tensor,
        anc_points: &Tensor,
        gt_labels: &Tensor,
        gt_xywhr: &Tensor,
        mask_gt: &Tensor,
    ) -> AssignOut {
        let dummy = Tensor::zeros([1, 1, 4], (Kind::Float, pd_xywhr.device()));
        self.assign_impl(pd_scores, Some(pd_xywhr), &dummy, anc_points, gt_labels, gt_xywhr, mask_gt)
    }

    /// pd_scores: (bs, A, nc) 已 sigmoid；pd_bboxes: (bs, A, 4) xyxy 像素；
    /// anc_points: (2, A)；gt_labels: (bs, m, 1)；gt_bboxes: (bs, m, 4) xyxy 像素；
    /// mask_gt: (bs, m, 1) bool。
    pub fn assign(
        &self,
        pd_scores: &Tensor,
        pd_bboxes: &Tensor,
        anc_points: &Tensor,
        gt_labels: &Tensor,
        gt_bboxes: &Tensor,
        mask_gt: &Tensor,
    ) -> AssignOut {
        self.assign_impl(pd_scores, None, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    }

    #[allow(clippy::too_many_arguments)]
    fn assign_impl(
        &self,
        pd_scores: &Tensor,
        pd_xywhr: Option<&Tensor>,
        pd_bboxes: &Tensor,
        anc_points: &Tensor,
        gt_labels: &Tensor,
        gt_bboxes: &Tensor,
        mask_gt: &Tensor,
    ) -> AssignOut {
        let rotated = pd_xywhr.is_some();
        let pd_bboxes: &Tensor = pd_xywhr.unwrap_or(pd_bboxes);
        let bs = pd_bboxes.size()[0];
        let na = pd_bboxes.size()[1];
        let m = gt_bboxes.size()[1];
        let device = pd_bboxes.device();
        let gw = gt_bboxes.size()[2]; // 4 = xyxy / 5 = xywhr

        if m == 0 {
            return AssignOut {
                target_labels: Tensor::full([bs, na], self.num_classes as f64, (Kind::Int64, device)),
                target_bboxes: Tensor::zeros([bs, na, gw], (Kind::Float, device)),
                target_scores: Tensor::zeros([bs, na, self.num_classes], (Kind::Float, device)),
                fg_mask: Tensor::zeros([bs, na], (Kind::Bool, device)),
                target_gt_idx: Tensor::zeros([bs, na], (Kind::Int64, device)),
            };
        }

        // 1) 框内锚 + 对齐指标（(bs, m, A)）
        let mask_in_gts = if rotated {
            let m = self.select_candidates_in_gts_rotated(anc_points, gt_bboxes, mask_gt);
            if std::env::var("UL_OBB_DEBUG").is_ok() {
                let per = m.sum_dim_intlist(Some(&[2i64][..]), false, Kind::Float);
                let v = Vec::<f32>::try_from(&per.reshape([-1])).unwrap_or_default();
                eprintln!("[obb-tal] in_gts={} per={v:?}", v.iter().map(|x| *x as i64).sum::<i64>());
            }
            m
        } else {
            self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        };
        let (mut align_metric, mut overlaps) =
            self.get_box_metrics_impl(pd_scores, pd_bboxes, gt_labels, gt_bboxes, &(mask_in_gts.to_kind(Kind::Bool) * mask_gt), rotated);

        // 2) topk 候选（每 gt 取 align 最高的 topk 锚；重复命中置 0）
        let topk_mask = mask_gt.expand([bs, m, self.topk], true);
        let mask_topk = self.select_topk_candidates(&align_metric, Some(&topk_mask));

        // 3) 合并正样本掩码
        let mut mask_pos = mask_topk
            * mask_in_gts.to_kind(Kind::Float)
            * mask_gt.to_kind(Kind::Float);

        // 4) 一锚多 gt 仲裁
        let (fg_mask_f, target_gt_idx) = self.select_highest_overlaps(&mut mask_pos, &mut overlaps);

        // 5) 目标
        let (target_labels, target_bboxes, target_scores) =
            self.get_targets(gt_labels, gt_bboxes, &target_gt_idx, &fg_mask_f, bs, m, device);

        // 6) 归一化（官方 parity）
        align_metric = align_metric * mask_pos.shallow_clone();
        let pos_align_metrics = align_metric.amax_last(-1, true); // (bs, m, 1)
        overlaps = overlaps * mask_pos;
        let pos_overlaps = overlaps.amax_last(-1, true); // (bs, m, 1)
        let normed = &align_metric * &pos_overlaps / (&pos_align_metrics + self.eps);
        let norm_align_metric = normed.amax_last(-2, false).unsqueeze(-1).clamp(0.0, 1.0); // (bs, A, 1)
        let target_scores = (target_scores * &norm_align_metric).clamp(0.0, 1.0);

        AssignOut {
            target_labels,
            target_bboxes,
            target_scores,
            fg_mask: fg_mask_f.gt(0.0),
            target_gt_idx,
        }
    }

    /// 旋转框内锚掩码（官方 corner 投影法）：wh<stride_val 放大后 → 四角点 → 锚点投影判内。
    fn select_candidates_in_gts_rotated(&self, anc_points: &Tensor, gt_xywhr: &Tensor, mask_gt: &Tensor) -> Tensor {
        let bs = gt_xywhr.size()[0];
        let m = gt_xywhr.size()[1];
        let a = anc_points.size()[1];
        // 小框放大：wh < stride_val → stride_val（官方条件含 mask_gt：padding 空框不放大）
        let wh = gt_xywhr.narrow(2, 2, 2);
        let small = wh.lt(self.stride_val).logical_and(&mask_gt.to_kind(Kind::Bool));
        let wh_fixed = small.to_kind(Kind::Float) * self.stride_val + (1.0 - small.to_kind(Kind::Float)) * wh;
        let gt = Tensor::cat(&[gt_xywhr.narrow(2, 0, 2).shallow_clone(), wh_fixed, gt_xywhr.narrow(2, 4, 1).shallow_clone()], 2);
        // 逐 gt 求角点（(bs, m, 8)），再投影判定
        let corners = rotated_corners_batched(&gt); // (bs, m, 4, 2): a, b, d(?), 顺序 p0 p1 p3
        let pa = corners.select(2, 0);
        let pb = corners.select(2, 1);
        let pd = corners.select(2, 3);
        let ab = &pb - &pa; // (bs, m, 2)
        let ad = &pd - &pa;
        let apx = anc_points.get(0).to_kind(Kind::Float).reshape([1, 1, a]) - pa.select(2, 0).unsqueeze(2); // (bs, m, A)
        let apy = anc_points.get(1).to_kind(Kind::Float).reshape([1, 1, a]) - pa.select(2, 1).unsqueeze(2);
        let norm_ab = (&ab.select(2, 0) * &ab.select(2, 0) + &ab.select(2, 1) * &ab.select(2, 1)).unsqueeze(2); // (bs, m, 1)
        let norm_ad = (&ad.select(2, 0) * &ad.select(2, 0) + &ad.select(2, 1) * &ad.select(2, 1)).unsqueeze(2);
        let ap_dot_ab = &apx * &ab.select(2, 0).unsqueeze(2) + &apy * &ab.select(2, 1).unsqueeze(2);
        let ap_dot_ad = &apx * &ad.select(2, 0).unsqueeze(2) + &apy * &ad.select(2, 1).unsqueeze(2);
        let az = &ap_dot_ab * 0.0;
        let dz = &ap_dot_ad * 0.0;
        (ap_dot_ab.ge_tensor(&az))
            .logical_and(&ap_dot_ab.le_tensor(&norm_ab))
            .logical_and(&ap_dot_ad.ge_tensor(&dz))
            .logical_and(&ap_dot_ad.le_tensor(&norm_ad))
    }

    /// 框内锚掩码（含小框放大到 stride_val），返回 (bs, m, A) bool。
    fn select_candidates_in_gts(&self, anc_points: &Tensor, gt_bboxes: &Tensor, mask_gt: &Tensor) -> Tensor {
        let bs = gt_bboxes.size()[0];
        let m = gt_bboxes.size()[1];
        let a = anc_points.size()[1];
        // 小框放大：wh < stride_val → stride_val
        let xywh = xyxy_to_xywh(gt_bboxes);
        let wh = xywh.narrow(2, 2, 2);
        let small = wh.lt(self.stride_val).logical_and(&mask_gt.to_kind(Kind::Bool));
        let wh_fixed = small.to_kind(Kind::Float) * self.stride_val + (1.0 - small.to_kind(Kind::Float)) * wh;
        let _ = wh;
        let xy = xywh.narrow(2, 0, 2).shallow_clone();
        let xywh = Tensor::cat(&[xy, wh_fixed], 2);
        let gt = xywh_to_xyxy(&xywh); // (bs, m, 4)

        let x = anc_points.get(0).to_kind(Kind::Float); // (A,)
        let y = anc_points.get(1).to_kind(Kind::Float);
        let g = gt.unsqueeze(2).expand([bs, m, a, 4], true); // (bs, m, A, 4)
        let xr = x.reshape([1, 1, a]).expand([bs, m, a], true);
        let yr = y.reshape([1, 1, a]).expand([bs, m, a], true);
        let eps = 1e-9f64;
        let gx0 = g.select(3, 0) + eps;
        let mx = xr.gt_tensor(&gx0);
        let gy1 = g.select(3, 1) + eps;
        let my = yr.gt_tensor(&gy1);
        let xr2 = xr + eps;
        let mx2 = g.select(3, 2).gt_tensor(&xr2);
        let yr2 = yr + eps;
        let my2 = g.select(3, 3).gt_tensor(&yr2);
        mx.logical_and(&my).logical_and(&mx2).logical_and(&my2)
    }

    /// 对齐指标（稀疏计算，官方 indices 语义），返回 (bs, m, A)。
    fn get_box_metrics_impl(
        &self,
        pd_scores: &Tensor,
        pd_bboxes: &Tensor,
        gt_labels: &Tensor,
        gt_bboxes: &Tensor,
        mask: &Tensor, // (bs, m, A) bool
        rotated: bool,
    ) -> (Tensor, Tensor) {
        let bs = pd_bboxes.size()[0];
        let na = pd_bboxes.size()[1];
        let m = gt_bboxes.size()[1];
        let shape = [bs, m, na];
        let zeros = Tensor::zeros(shape, (Kind::Float, pd_bboxes.device()));
        let total = f64::try_from(&mask.to_kind(Kind::Float).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)).unwrap_or(0.0);
        if total == 0.0 {
            return (zeros.shallow_clone(), zeros);
        }
        // indices = mask.nonzero → (K, 3)
        let idx = mask.nonzero();
        let b_idx = idx.select(1, 0);
        let g_idx = idx.select(1, 1);
        let a_idx = idx.select(1, 2);
        // bbox_scores = pd_scores[b, a, gt_label]
        let lbl = gt_labels.select(2, 0).to_kind(Kind::Int64); // (bs, m)
        let lbl_sel = lbl.index_select(0, &b_idx).index_select(1, &g_idx); // (K, 1) g_idx 是逐 K 的列选择——改为 gather
        let _ = lbl_sel;
        // 逐三元组取值：label = lbl[b, g]
        let ps = pd_scores.permute([0, 2, 1]); // (bs, nc, A)
        let lbl_k = batched_select2_i64(&lbl, &b_idx, &g_idx); // (K,)
        let scores_sel = batched_select3(&ps, &b_idx, &lbl_k, &a_idx);
        let gt_sel = batched_select2(gt_bboxes, &b_idx, &g_idx); // (K, 4)
        let pd_sel = batched_select2(pd_bboxes, &b_idx, &a_idx); // (K, 4)
        // 成对 IoU；官方 iou_calculation 末尾 clamp_(0)（负值在 ^β 下会翻正）
        let ov = if rotated {
            crate::ops::probiou_pairwise(&gt_sel, &pd_sel, false, 0.0).squeeze_dim(1).clamp_min(0.0)
        } else {
            crate::ops::bbox_iou_pairwise(&gt_sel, &pd_sel, true).squeeze_dim(1).clamp_min(0.0)
        }; // (K,)
        let align_vals = scores_sel.pow_tensor_scalar(self.alpha) * ov.pow_tensor_scalar(self.beta);

        // 两个独立零张量：shallow_clone 共享存储会让 overlaps/align 相互覆盖
        let overlaps = Tensor::zeros(shape, (Kind::Float, pd_bboxes.device())).index_put_(
            &[Some(&b_idx), Some(&g_idx), Some(&a_idx)],
            &ov,
            false,
        );
        let align = Tensor::zeros(shape, (Kind::Float, pd_scores.device())).index_put_(
            &[Some(&b_idx), Some(&g_idx), Some(&a_idx)],
            &align_vals,
            false,
        );
        (align, overlaps)
    }

    /// topk 候选（官方 scatter_add 计数，重复置 0）。
    fn select_topk_candidates(&self, metrics: &Tensor, topk_mask: Option<&Tensor>) -> Tensor {
        let (topk_metrics, topk_idxs) = metrics.topk(self.topk, -1, true, true); // (bs, m, topk)
        let masked_idxs = match topk_mask {
            Some(tm) => {
                let tm = tm.to_kind(Kind::Bool);
                topk_idxs.where_scalarother(&tm, 0)
            }
            None => {
                let ok = topk_metrics.amax_last(-1, true).gt(self.eps);
                topk_idxs.where_scalarother(&ok, 0)
            }
        };

        // scatter_add 计数
        let ones = Tensor::ones(masked_idxs.size(), (Kind::Float, masked_idxs.device()));
        let count = Tensor::zeros(metrics.size(), (Kind::Float, metrics.device()));
        let count = count.scatter_add(-1, &masked_idxs, &ones);
        // count == 1 保留（官方 >1 置 0；0 本来就没选中）
        count.eq(1.0).to_kind(Kind::Float)
    }

    /// select_highest_overlaps（官方）：一锚多 gt → 保留重叠最大的 gt。
    /// 全程 (bs, m, A)。返回 (fg (bs, A) float, idx (bs, A) i64)。
    fn select_highest_overlaps(&self, mask_pos: &mut Tensor, overlaps: &mut Tensor) -> (Tensor, Tensor) {
        let (bs, m, a) = (mask_pos.size()[0], mask_pos.size()[1], mask_pos.size()[2]);
        let device = mask_pos.device();
        let fg_any = mask_pos.sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float); // (bs, A)
        let multi = fg_any.gt(1.0);
        let anyv = f64::try_from(&multi.to_kind(Kind::Float).sum_dim_intlist(None::<&[i64]>, false, Kind::Float)).unwrap_or(0.0);
        if anyv > 0.0 {
            let mask_multi = multi.unsqueeze(1).expand([bs, m, a], true); // (bs, m, A)
            let max_idx = overlaps.argmax(1, false); // (bs, A)
            // is_max: (bs, m, A) 沿 dim1 scatter
            let ones = Tensor::ones([bs, 1, a], (Kind::Float, device));
            let is_max = Tensor::zeros([bs, m, a], (Kind::Float, device))
                .scatter(1, &max_idx.unsqueeze(1), &ones);
            // mask_pos = where(mask_multi, is_max, mask_pos)
            let keep = mask_pos.shallow_clone() * (1.0 - mask_multi.to_kind(Kind::Float));
            let take = is_max * mask_multi.to_kind(Kind::Float);
            *mask_pos = take + keep;
        }
        let maxed = mask_pos.max_dim(1, false); // (values (bs,A), indices (bs,A))
        (maxed.0, maxed.1)
    }

    #[allow(clippy::too_many_arguments)]
    fn get_targets(
        &self,
        gt_labels: &Tensor,
        gt_bboxes: &Tensor,
        target_gt_idx: &Tensor,
        fg_mask: &Tensor,
        bs: i64,
        m: i64,
        device: tch::Device,
    ) -> (Tensor, Tensor, Tensor) {
        let a = target_gt_idx.size()[1];
        let batch_idx = Tensor::arange(bs, (Kind::Int64, device))
            .reshape([bs, 1])
            .expand([bs, a], true);
        let idx = target_gt_idx + batch_idx * m;
        let flat = idx.reshape([-1]);
        let target_labels = gt_labels
            .to_kind(Kind::Int64)
            .reshape([-1])
            .index_select(0, &flat)
            .reshape([bs, a])
            .clamp(0, self.num_classes - 1);
        let gw = gt_bboxes.size()[2];
        let target_bboxes = gt_bboxes
            .reshape([-1, gw])
            .index_select(0, &flat)
            .reshape([bs, a, gw]);
        let onehot = one_hot_i64(&target_labels, self.num_classes);
        let target_scores = onehot * fg_mask.gt(0.0).to_kind(Kind::Float).unsqueeze(2);
        (target_labels, target_bboxes, target_scores)
    }
}

fn xyxy_to_xywh(x: &Tensor) -> Tensor {
    let parts = x.split_with_sizes(&[2, 2], -1);
    let (xy1, xy2) = (&parts[0], &parts[1]);
    Tensor::cat(&[(xy1 + xy2) / 2.0, xy2 - xy1], -1)
}

fn xywh_to_xyxy(x: &Tensor) -> Tensor {
    let parts = x.split_with_sizes(&[2, 2], -1);
    let (xy, wh) = (&parts[0], &parts[1]);
    Tensor::cat(&[xy - wh / 2.0, xy + wh / 2.0], -1)
}

fn one_hot_i64(idx: &Tensor, n: i64) -> Tensor {
    let mut shape = idx.size().to_vec();
    shape.push(n);
    let zeros = Tensor::zeros(&shape, (Kind::Float, idx.device()));
    let mut src = idx.size().to_vec();
    src.push(1);
    let ones = Tensor::ones(&src, (Kind::Float, idx.device()));
    zeros.scatter(2, &idx.unsqueeze(2), &ones)
}

/// (bs, X) i64 按 (b, g) 取值 → (K,)。
fn batched_select2_i64(t: &Tensor, b: &Tensor, g: &Tensor) -> Tensor {
    let sel = t.index_select(0, b); // (K, m)
    sel.gather(1, &g.reshape([-1, 1]), false).squeeze_dim(1)
}

/// (bs, X, Y) 张量按三元组取值 → (K,)。
fn batched_select3(t: &Tensor, b: &Tensor, y: &Tensor, x: &Tensor) -> Tensor {
    // 先 gather y（通道），再 gather x（空间）
    let sel_y = t.index_select(0, b); // (K, Y, X)
    let yy = y.reshape([-1, 1, 1]).expand([-1, 1, sel_y.size()[2]], true);
    let picked_y = sel_y.gather(1, &yy, false).squeeze_dim(1); // (K, X)
    picked_y.gather(1, &x.reshape([-1, 1]), false).squeeze_dim(1)
}

/// (bs, X, W) 按二元组取行 → (K, W)（W 自适应：detect 4 / obb 5）。
fn batched_select2(t: &Tensor, b: &Tensor, x: &Tensor) -> Tensor {
    let w = t.size()[2];
    let sel = t.index_select(0, b); // (K, X, W)
    sel.gather(1, &x.reshape([-1, 1, 1]).expand([-1, 1, w], true), false).squeeze_dim(1)
}

trait AmaxLast {
    fn amax_last(&self, dim: i64, keepdim: bool) -> Tensor;
}

impl AmaxLast for Tensor {
    fn amax_last(&self, dim: i64, keepdim: bool) -> Tensor {
        self.f_amax(&[dim], keepdim).unwrap()
    }
}

/// (bs, m, 5) xywhr → (bs, m, 4, 2) 四角点（p0..p3 绕行）。
fn rotated_corners_batched(whr: &Tensor) -> Tensor {
    let (xc, yc) = (whr.narrow(2, 0, 1), whr.narrow(2, 1, 1));
    let cos = whr.narrow(2, 4, 1).cos();
    let sin = whr.narrow(2, 4, 1).sin();
    let (w, h) = (whr.narrow(2, 2, 1) / 2.0, whr.narrow(2, 3, 1) / 2.0); // 半宽/半高
    let (ux, uy) = (&w * &cos, &w * &sin);
    let (vx, vy) = (-&h * &sin, &h * &cos);
    let mk = |sx: &Tensor, sy: &Tensor| -> Tensor {
        let px = &xc + &ux * sx + &vx * sy;
        let py = &yc + &uy * sx + &vy * sy;
        Tensor::stack(&[px.squeeze_dim(2), py.squeeze_dim(2)], 2)
    };
    let neg1 = -1.0 * Tensor::ones_like(&xc);
    let pos1 = Tensor::ones_like(&xc);
    let p0 = mk(&neg1, &neg1);
    let p1 = mk(&pos1, &neg1);
    let p2 = mk(&pos1, &pos1);
    let p3 = mk(&neg1, &pos1);
    Tensor::stack(&[p0, p1, p2, p3], 2)
}
