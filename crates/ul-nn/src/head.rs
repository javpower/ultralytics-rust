//! 任务头：Detect / Segment / Pose / Pose26 / OBB / OBB26 / v10Detect / Classify / SemanticSegment。
//!
//! 单一 DetectHead 结构承载所有检测类头部（extra 分支区分任务），
//! 对应 ultralytics head.py 的 Detect 继承树；参数命名逐字对齐。

use tch::{Kind, Tensor};

use crate::block::Dfl;
use crate::conv::{dwconv_init, Conv};
use crate::module::{Init, VarPath};
use crate::ops::{dist2bbox, dist2rbox, make_anchors, ShallowPair};
use crate::special::{Proto, Proto26};

/// 头部分支的一段：Conv 或 (DWConv, Conv) 嵌套 Sequential。
pub enum BranchStage {
    Conv(Conv),
    Pair(Conv, Conv),
}

impl BranchStage {
    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        match self {
            BranchStage::Conv(c) => c.forward(x, train),
            BranchStage::Pair(dw, c) => c.forward(&dw.forward(x, train), train),
        }
    }
}

/// 完整分支：stage0 → stage1 → 裸 1x1 conv2d 输出。
pub struct Branch {
    pub stage0: BranchStage,
    pub stage1: BranchStage,
    pub final_conv: RawOut,
}

/// 分支末端的裸 Conv2d（weight/bias 命名在父路径上，无 conv. 前缀）。
pub struct RawOut {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl RawOut {
    fn init(p: &VarPath, c_in: i64, c_out: i64) -> Self {
        let bound = (1.0 / (c_in * 1 * 1) as f64).sqrt();
        RawOut {
            weight: p.var_conv_weight("weight", c_out, c_in, 1, bound),
            bias: p.var_1d("bias", c_out, Init::UniformSigned((c_in as f64).sqrt())),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        x.f_conv2d(&self.weight, Some(&self.bias), &[1, 1], &[0, 0], &[1, 1], 1).unwrap()
    }
}

impl Branch {
    /// box/cls/mask/kpt/angle 分支统一构造。
    ///
    /// * legacy=true：两段标准 Conv；false：两段 (DWConv, Conv) 嵌套对。
    fn init(p: &VarPath, c1: i64, hidden: i64, c_out: i64, legacy: bool) -> Self {
        let (stage0, stage1) = if legacy {
            (
                BranchStage::Conv(Conv::init(&p.idx(0), c1, hidden, 3, 1, None, 1, 1, true)),
                BranchStage::Conv(Conv::init(&p.idx(1), hidden, hidden, 3, 1, None, 1, 1, true)),
            )
        } else {
            (
                BranchStage::Pair(
                    dwconv_init(&p.idx(0).idx(0), c1, c1, 3, 1, 1, true),
                    Conv::init(&p.idx(0).idx(1), c1, hidden, 1, 1, None, 1, 1, true),
                ),
                BranchStage::Pair(
                    dwconv_init(&p.idx(1).idx(0), hidden, hidden, 3, 1, 1, true),
                    Conv::init(&p.idx(1).idx(1), hidden, hidden, 1, 1, None, 1, 1, true),
                ),
            )
        };
        Branch { stage0, stage1, final_conv: RawOut::init(&p.idx(2), hidden, c_out) }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.stage1.forward(&self.stage0.forward(x, train), train);
        self.final_conv.forward(&y)
    }
}

/// 附加分支类型（Segment/Pose/OBB 的 cv4）。
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ExtraKind {
    None,
    /// Segment：mask 系数（nm 通道）。
    Mask,
    /// Pose：关键点（nk = K*ndim 通道；cv4 三段 Conv）。
    Kpt,
    /// Pose26：cv4 两段 Conv + cv4_kpts/cv4_sigma 1x1。
    Kpt26,
    /// OBB：角度（ne 通道，sigmoid 变换）。
    Angle,
    /// OBB26：角度（原始输出）。
    Angle26,
}

pub enum ProtoVariant {
    P1(Proto),
    P26(Box<Proto26>),
}

/// 检测类头部（覆盖 Detect 继承树全部形态）。
pub struct DetectHead {
    pub nc: i64,
    pub nl: i64,
    pub reg_max: i64,
    pub no: i64,
    pub legacy: bool,
    pub end2end: bool,
    pub xyxy: bool,
    pub max_det: i64,
    pub stride: Vec<f64>,
    pub cv2: Vec<Branch>,
    pub cv3: Vec<Branch>,
    pub dfl: Option<Dfl>,
    pub o2o_cv2: Option<Vec<Branch>>,
    pub o2o_cv3: Option<Vec<Branch>>,
    pub kind: ExtraKind,
    pub cv4: Vec<Branch>,
    pub o2o_cv4: Option<Vec<Branch>>,
    /// Kpt26 专用：cv4 两段 Conv（无 1x1 收尾）。
    pub cv4_pair: Vec<(Conv, Conv)>,
    pub o2o_cv4_pair: Option<Vec<(Conv, Conv)>>,
    pub proto: Option<ProtoVariant>,
    pub nm: i64,
    pub ne: i64,
    pub kpt_shape: (i64, i64),
    /// Pose26 附加 1x1 头。
    pub cv4_kpts: Vec<RawOut>,
    pub cv4_sigma: Vec<RawOut>,
    pub o2o_cv4_kpts: Option<Vec<RawOut>>,
    pub o2o_cv4_sigma: Option<Vec<RawOut>>,
    // 锚点缓存（内部可变性，forward 用 &self）
    cache: std::cell::RefCell<((i64, i64, i64), Option<(Tensor, Tensor)>)>,
    pub bias_inited: bool,
}

/// 训练期原始预测（损失计算输入）。
pub struct RawPreds {
    pub boxes: Tensor,        // (B, 4*reg_max, A)
    pub scores: Tensor,       // (B, nc, A)
    pub extra: Option<Tensor>, // mask coef (B,nm,A) / kpts (B,nk,A) / angle (B,ne,A)
    pub extra2: Option<Tensor>, // Pose26 kpts_sigma（仅训练）
    pub proto: Option<Tensor>,
    pub proto_sem: Option<Tensor>, // Proto26 语义 logits（仅训练）
    pub feats: Vec<Tensor>,
}

/// 头部前向输出。
pub enum HeadOutput {
    /// 训练（单头，legacy 模型）。
    Train(Box<RawPreds>),
    /// 训练（双头 e2e）：(one2many, one2one)。
    TrainDual(Box<RawPreds>, Box<RawPreds>),
    /// 推理：(B, 4+nc+extra, A) 或 e2e (B, max_det, 6+extra)。
    Inference(Tensor),
    /// 分类：(probs, logits)。
    Classify(Tensor, Tensor),
    /// 语义分割 logits。
    SemSeg(Tensor),
}

impl DetectHead {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p: &VarPath,
        nc: i64,
        ch: &[i64],
        reg_max: i64,
        end2end: bool,
        legacy: bool,
        kind: ExtraKind,
        nm: i64,
        npr: i64,
        ne: i64,
        kpt_shape: (i64, i64),
        light_cls: bool, // v10Detect
    ) -> Self {
        let nl = ch.len() as i64;
        let c2 = (16i64).max(ch[0] / 4).max(reg_max * 4);
        let c3 = ch[0].max(nc.min(100));
        let mut cv2 = Vec::new();
        let mut cv3 = Vec::new();
        for (i, &x) in ch.iter().enumerate() {
            let pi = p.idx(i as usize);
            cv2.push(Branch::init(&pi.child("cv2"), x, c2, reg_max * 4, true));
            let c3i = if light_cls { x } else { c3 };
            cv3.push(if light_cls {
                // v10 轻量 cls：Sequential((Conv g=x 3x3, Conv 1x1), (Conv g=c3, Conv 1x1), 1x1)
                let s0 = BranchStage::Pair(
                    Conv::init(&pi.child("cv3").idx(0).idx(0), x, x, 3, 1, None, x, 1, true),
                    Conv::init(&pi.child("cv3").idx(0).idx(1), x, c3i, 1, 1, None, 1, 1, true),
                );
                let s1 = BranchStage::Pair(
                    Conv::init(&pi.child("cv3").idx(1).idx(0), c3i, c3i, 3, 1, None, c3i, 1, true),
                    Conv::init(&pi.child("cv3").idx(1).idx(1), c3i, c3i, 1, 1, None, 1, 1, true),
                );
                Branch { stage0: s0, stage1: s1, final_conv: RawOut::init(&pi.child("cv3").idx(2), c3i, nc) }
            } else {
                Branch::init(&pi.child("cv3"), x, c3, nc, legacy)
            });
        }
        // 附加分支
        let (mut cv4, mut cv4_kpts, mut cv4_sigma) = (Vec::new(), Vec::new(), Vec::new());
        let mut cv4_pair: Vec<(Conv, Conv)> = Vec::new();
        match kind {
            ExtraKind::None => {}
            ExtraKind::Mask => {
                let c4 = (ch[0] / 4).max(nm);
                for (i, &x) in ch.iter().enumerate() {
                    cv4.push(Branch::init(&p.idx(i as usize).child("cv4"), x, c4, nm, true));
                }
            }
            ExtraKind::Kpt => {
                let nk = kpt_shape.0 * kpt_shape.1;
                let c4 = (ch[0] / 4).max(nk);
                for (i, &x) in ch.iter().enumerate() {
                    cv4.push(Branch::init(&p.idx(i as usize).child("cv4"), x, c4, nk, true));
                }
            }
            ExtraKind::Kpt26 => {
                let nk = kpt_shape.0 * kpt_shape.1;
                let nk_sigma = kpt_shape.0 * 2;
                let c4 = (ch[0] / 4).max(kpt_shape.0 * (kpt_shape.1 + 2));
                for (i, &x) in ch.iter().enumerate() {
                    let pi = p.idx(i as usize);
                    // cv4 = Sequential(Conv, Conv)（两段，无 1x1 收尾）
                    cv4_pair.push((
                        Conv::init(&pi.child("cv4").idx(0), x, c4, 3, 1, None, 1, 1, true),
                        Conv::init(&pi.child("cv4").idx(1), c4, c4, 3, 1, None, 1, 1, true),
                    ));
                    cv4_kpts.push(RawOut::init(&pi.child("cv4_kpts").idx(i as usize), c4, nk));
                    cv4_sigma.push(RawOut::init(&pi.child("cv4_sigma").idx(i as usize), c4, nk_sigma));
                }
            }
            ExtraKind::Angle | ExtraKind::Angle26 => {
                let c4 = (ch[0] / 4).max(ne);
                for (i, &x) in ch.iter().enumerate() {
                    cv4.push(Branch::init(&p.idx(i as usize).child("cv4"), x, c4, ne, true));
                }
            }
        }
        let _ = &mut cv4_pair;
        let dfl = if reg_max > 1 { Some(Dfl::new(p, reg_max)) } else { None };
        DetectHead {
            nc,
            nl,
            reg_max,
            no: nc + reg_max * 4,
            legacy,
            end2end,
            xyxy: false,
            max_det: 300,
            stride: vec![0.0; nl as usize],
            cv2,
            cv3,
            dfl,
            o2o_cv2: None,
            o2o_cv3: None,
            kind,
            cv4,
            o2o_cv4: None,
            cv4_pair,
            o2o_cv4_pair: None,
            proto: None,
            nm,
            ne,
            kpt_shape,
            cv4_kpts,
            cv4_sigma,
            o2o_cv4_kpts: None,
            o2o_cv4_sigma: None,
            cache: std::cell::RefCell::new(((0, 0, 0), None)),
            bias_inited: false,
        }
    }

    pub fn set_end2end(&mut self, p: &VarPath, ch: &[i64]) {
        // 深拷贝 one2one 分支（结构与主分支一致）
        let c2 = (16i64).max(ch[0] / 4).max(self.reg_max * 4);
        let c3 = ch[0].max(self.nc.min(100));
        let mut cv2 = Vec::new();
        let mut cv3 = Vec::new();
        let mut cv4 = Vec::new();
        let mut cv4_pair: Vec<(Conv, Conv)> = Vec::new();
        let mut cv4_kpts = Vec::new();
        let mut cv4_sigma = Vec::new();
        for (i, &x) in ch.iter().enumerate() {
            let pi = p.idx(i as usize);
            cv2.push(Branch::init(&pi.child("one2one_cv2"), x, c2, self.reg_max * 4, true));
            cv3.push(Branch::init(&pi.child("one2one_cv3"), x, c3, self.nc, self.legacy));
            match self.kind {
                ExtraKind::None => {}
                ExtraKind::Mask => {
                    let c4 = (ch[0] / 4).max(self.nm);
                    cv4.push(Branch::init(&pi.child("one2one_cv4"), x, c4, self.nm, true));
                }
                ExtraKind::Kpt => {
                    let nk = self.kpt_shape.0 * self.kpt_shape.1;
                    let c4 = (ch[0] / 4).max(nk);
                    cv4.push(Branch::init(&pi.child("one2one_cv4"), x, c4, nk, true));
                }
                ExtraKind::Kpt26 => {
                    let nk = self.kpt_shape.0 * self.kpt_shape.1;
                    let nk_sigma = self.kpt_shape.0 * 2;
                    let c4 = (ch[0] / 4).max(self.kpt_shape.0 * (self.kpt_shape.1 + 2));
                    cv4_pair.push((
                        Conv::init(&pi.child("one2one_cv4").idx(0), x, c4, 3, 1, None, 1, 1, true),
                        Conv::init(&pi.child("one2one_cv4").idx(1), c4, c4, 3, 1, None, 1, 1, true),
                    ));
                    cv4_kpts.push(RawOut::init(&pi.child("one2one_cv4_kpts").idx(i as usize), c4, nk));
                    cv4_sigma.push(RawOut::init(&pi.child("one2one_cv4_sigma").idx(i as usize), c4, nk_sigma));
                }
                ExtraKind::Angle | ExtraKind::Angle26 => {
                    let c4 = (ch[0] / 4).max(self.ne);
                    cv4.push(Branch::init(&pi.child("one2one_cv4"), x, c4, self.ne, true));
                }
            }
        }
        self.o2o_cv2 = Some(cv2);
        self.o2o_cv3 = Some(cv3);
        if !cv4.is_empty() {
            self.o2o_cv4 = Some(cv4);
        }
        if !cv4_pair.is_empty() {
            self.o2o_cv4_pair = Some(cv4_pair);
        }
        if !cv4_kpts.is_empty() {
            self.o2o_cv4_kpts = Some(cv4_kpts);
        }
        if !cv4_sigma.is_empty() {
            self.o2o_cv4_sigma = Some(cv4_sigma);
        }
        self.end2end = true;
    }

    /// ultralytics Detect.bias_init（须在 stride 就绪后调用一次）。
    pub fn bias_init(&mut self) {
        tch::no_grad(|| {
            let run = |branches: &mut Vec<Branch>, cv3: &mut Vec<Branch>, nc: i64, stride: &[f64]| {
                for b in branches.iter_mut() {
                    let _ = b.final_conv.bias.fill_(2.0);
                }
                for (i, b) in cv3.iter_mut().enumerate() {
                    let v = (5.0 / nc as f64 / (640.0 / stride[i]).powi(2)).ln();
                    let _ = b.final_conv.bias.narrow(0, 0, nc).fill_(v);
                }
            };
            run(&mut self.cv2, &mut self.cv3, self.nc, &self.stride);
            if let (Some(c2), Some(c3)) = (&mut self.o2o_cv2, &mut self.o2o_cv3) {
                run(c2, c3, self.nc, &self.stride);
            }
        });
        self.bias_inited = true;
    }

    fn anchors(&self, feats: &[Tensor]) -> (Tensor, Tensor) {
        let shape = (feats[0].size()[0], feats[0].size()[2], feats[0].size()[3]);
        let mut cache = self.cache.borrow_mut();
        let (ref mut cached_shape, ref mut cached) = *cache;
        if cached.is_none() || *cached_shape != shape {
            // 官方缓存语义：points (2, A)，strides (1, A)
            let (a, s) = make_anchors(feats, &self.stride, 0.5);
            *cached = Some((a, s));
            *cached_shape = shape;
        }
        cached.as_ref().unwrap().clone_pair()
    }

    fn forward_branch(
        &self,
        x: &[Tensor],
        cv2: &[Branch],
        cv3: &[Branch],
        cv4: &[Branch],
        cv4_pair: &[(Conv, Conv)],
        cv4_kpts: Option<&Vec<RawOut>>,
        train_extra: bool,
        train: bool,
    ) -> RawPreds {
        let bs = x[0].size()[0];
        let boxes = Tensor::cat(
            &(0..self.nl as usize)
                .map(|i| cv2[i].forward(&x[i], train).reshape([bs, self.reg_max * 4, -1]))
                .collect::<Vec<_>>(),
            2,
        );
        let scores = Tensor::cat(
            &(0..self.nl as usize)
                .map(|i| cv3[i].forward(&x[i], train).reshape([bs, self.nc, -1]))
                .collect::<Vec<_>>(),
            2,
        );
        let mut extra = None;
        let mut extra2 = None;
        match self.kind {
            ExtraKind::None => {}
            ExtraKind::Mask => {
                extra = Some(Tensor::cat(
                    &(0..self.nl as usize)
                        .map(|i| cv4[i].forward(&x[i], train).reshape([bs, self.nm, -1]))
                        .collect::<Vec<_>>(),
                    2,
                ));
            }
            ExtraKind::Kpt => {
                extra = Some(Tensor::cat(
                    &(0..self.nl as usize)
                        .map(|i| cv4[i].forward(&x[i], train).reshape([bs, self.nk(), -1]))
                        .collect::<Vec<_>>(),
                    2,
                ));
            }
            ExtraKind::Kpt26 => {
                // features = cv4 两段 Conv 输出；kpts = cv4_kpts(features)
                let feats: Vec<Tensor> = (0..self.nl as usize)
                    .map(|i| {
                        let (a, b) = &cv4_pair[i];
                        b.forward(&a.forward(&x[i], train), train)
                    })
                    .collect();
                let heads = cv4_kpts.expect("Kpt26 requires cv4_kpts");
                extra = Some(Tensor::cat(
                    &(0..self.nl as usize)
                        .map(|i| heads[i].forward(&feats[i]).reshape([bs, self.nk(), -1]))
                        .collect::<Vec<_>>(),
                    2,
                ));
                if train_extra {
                    let sig = self.cv4_sigma.as_slice();
                    extra2 = Some(Tensor::cat(
                        &(0..self.nl as usize)
                            .map(|i| sig[i].forward(&feats[i]).reshape([bs, self.kpt_shape.0 * 2, -1]))
                            .collect::<Vec<_>>(),
                        2,
                    ));
                }
            }
            ExtraKind::Angle | ExtraKind::Angle26 => {
                let angle = Tensor::cat(
                    &(0..self.nl as usize)
                        .map(|i| cv4[i].forward(&x[i], train).reshape([bs, self.ne, -1]))
                        .collect::<Vec<_>>(),
                    2,
                );
                extra = Some(if self.kind == ExtraKind::Angle {
                    (angle.sigmoid() - 0.25) * std::f64::consts::PI
                } else {
                    angle
                });
            }
        }
        RawPreds { boxes, scores, extra, extra2, proto: None, proto_sem: None, feats: x.iter().map(|t| t.shallow_clone()).collect() }
    }

    fn nk(&self) -> i64 {
        self.kpt_shape.0 * self.kpt_shape.1
    }

    /// 多层特征前向（检测类头统一入口）。
    pub fn forward(&self, x: &[Tensor], train: bool) -> HeadOutput {
        let o2m = self.forward_branch(
            x,
            &self.cv2,
            &self.cv3,
            &self.cv4,
            &self.cv4_pair,
            Some(&self.cv4_kpts),
            true,
            train,
        );
        if self.o2o_cv2.is_some() {
            // 训练时 one2one 分支 detach（不回传主干）
            let x_o2o: Vec<Tensor> =
                if train { x.iter().map(|t| t.detach()).collect() } else { x.iter().map(|t| t.shallow_clone()).collect() };
            let empty_pair: Vec<(Conv, Conv)> = Vec::new();
            let o2o = self.forward_branch(
                &x_o2o,
                self.o2o_cv2.as_ref().unwrap(),
                self.o2o_cv3.as_ref().unwrap(),
                self.o2o_cv4.as_ref().unwrap_or(&self.cv4),
                self.o2o_cv4_pair.as_ref().unwrap_or(if self.kind == ExtraKind::Kpt26 { &self.cv4_pair } else { &empty_pair }),
                Some(self.o2o_cv4_kpts.as_ref().unwrap_or(&self.cv4_kpts)),
                false,
                train,
            );
            if !train {
                let y = self.inference_decode(&o2o);
                return HeadOutput::Inference(self.postprocess_e2e(&y));
            }
            return HeadOutput::TrainDual(Box::new(o2m), Box::new(o2o));
        }
        if !train {
            let y = self.inference_decode(&o2m);
            return HeadOutput::Inference(y);
        }
        HeadOutput::Train(Box::new(o2m))
    }

    /// 解码：dist → 框（×strides），拼 sigmoid 分数与附加通道。
    fn inference_decode(&self, preds: &RawPreds) -> Tensor {
        let (anchors, strides) = self.anchors(&preds.feats);
        let raw_boxes = self
            .dfl
            .as_ref()
            .map(|d| d.forward(&preds.boxes))
            .unwrap_or_else(|| preds.boxes.shallow_clone());
        let dbox = if self.kind == ExtraKind::Angle || self.kind == ExtraKind::Angle26 {
            // OBB：angle (B, ne, A)；anchors (2,A) → unsqueeze(0) (1,2,A)
            let angle = preds.extra.as_ref().unwrap();
            dist2rbox(&raw_boxes, angle, &anchors, 1)
        } else {
            let xywh = !self.end2end && !self.xyxy;
            dist2bbox(&raw_boxes, &anchors, xywh, 1)
        };
        let dbox = dbox * &strides; // (1, A) 广播
        let scores = preds.scores.sigmoid();
        let mut parts = vec![dbox, scores];
        if let Some(e) = &preds.extra {
            // Kpt26/Kpt 的关键点解码在拼通道前进行
            if self.kind == ExtraKind::Kpt {
                parts.push(self.kpts_decode(e, &anchors.squeeze_dim(0), &strides));
            } else {
                parts.push(e.shallow_clone());
            }
        }
        Tensor::cat(&parts, 1)
    }

    /// Pose 关键点解码：kpts (B, nk, A) → 像素坐标（训练/非导出语义：
    /// xy = (y·2 + (anchor − 0.5))·stride，visibility = sigmoid）。
    fn kpts_decode(&self, kpts: &Tensor, anchors: &Tensor, strides: &Tensor) -> Tensor {
        let ndim = self.kpt_shape.1;
        let bs = kpts.size()[0];
        let k = self.kpt_shape.0;
        let a = kpts.size()[2];
        let anchors_b = anchors.unsqueeze(0).unsqueeze(1); // (1,1,2,A)
        let strides_b = strides.reshape([1, 1, 1, -1]); // (1,1,1,A)
        if ndim == 3 {
            let mut sizes: Vec<i64> = Vec::new();
            for _ in 0..k {
                sizes.push(2);
                sizes.push(1);
            }
            let s = kpts.split_with_sizes(&sizes, 1);
            let xy_parts: Vec<Tensor> = (0..k as usize).map(|i| s[i * 2].shallow_clone()).collect();
            let vis_parts: Vec<Tensor> = (0..k as usize).map(|i| s[i * 2 + 1].shallow_clone()).collect();
            let xy_all = Tensor::cat(&xy_parts, 1); // (B, 2k, A)
            let vis = Tensor::cat(&vis_parts, 1).sigmoid(); // (B, k, A)
            let s2 = xy_all.reshape([bs, k, 2, a]);
            let xy = (s2 * 2.0 + (anchors_b - 0.5) * strides_b).reshape([bs, k * 2, a]);
            let mut parts: Vec<Tensor> = Vec::new();
            for i in 0..k {
                parts.push(xy.narrow(1, i * 2, 2));
                parts.push(vis.narrow(1, i, 1));
            }
            Tensor::cat(&parts, 1)
        } else {
            let y2 = kpts.reshape([bs, k, 2, a]);
            let dec = y2 * 2.0 + (anchors_b - 0.5) * strides_b;
            dec.reshape([bs, k * 2, a])
        }
    }

    /// e2e 后处理：topk(max_det) → (B, max_det, 6+extra)。
    fn postprocess_e2e(&self, preds: &Tensor) -> Tensor {
        // preds (B, 4+nc+extra, A) → (B, A, 4+nc+extra)
        let p = preds.transpose(1, 2);
        let total = p.size()[2];
        let extra_n = total - 4 - self.nc;
        let parts = p.split_with_sizes(&[4, self.nc, extra_n.max(0)], 2);
        let boxes = &parts[0];
        let scores_raw = &parts[1];
        let extra = if extra_n > 0 { Some(&parts[2]) } else { None };
        // scores.max(-1) → (B, A)
        let max_scores = scores_raw.f_amax(&[2], false).unwrap();
        let k = self.max_det.min(p.size()[1]);
        let (_sv, idx0) = max_scores.topk(k, 1, true, false);
        // 收集 (B, k, nc)
        let sel_scores = scores_raw.gather(1, &idx0.unsqueeze(2).expand([-1, -1, self.nc], true), false);
        let flat = sel_scores.reshape([sel_scores.size()[0], -1]); // (B, k*nc)
        let (scores, idx1) = flat.topk(k, 1, true, false);
        // idx0 行 = idx1 // nc
        let anchor_idx = idx0.gather(1, &idx1.floor_divide_scalar(self.nc), false);
        let q = idx1.floor_divide_scalar(self.nc);
        let labels = ((&idx1 - &q) * self.nc).to_kind(Kind::Float).unsqueeze(2);
        let gboxes = boxes.gather(1, &anchor_idx.unsqueeze(2).expand([-1, -1, 4], true), false);
        let out = match extra {
            Some(e) => {
                let ge = e.gather(1, &anchor_idx.unsqueeze(2).expand([-1, -1, e.size()[2]], true), false);
                Tensor::cat(&[gboxes, scores.unsqueeze(2), labels, ge], 2)
            }
            None => Tensor::cat(&[gboxes, scores.unsqueeze(2), labels], 2),
        };
        out
    }
}

// ---------------- Classify / SemanticSegment ----------------

/// 分类头：Conv(c1,1280) + GAP + Linear。
pub struct ClassifyHead {
    pub conv: Conv,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl ClassifyHead {
    pub fn init(p: &VarPath, c1: i64, c2: i64) -> Self {
        let (weight, bias) = p.var_linear("linear.weight", c2, 1280);
        ClassifyHead { conv: Conv::init(p, c1, 1280, 1, 1, None, 1, 1, true), weight, bias }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        let y = self.conv.forward(x, train);
        let pooled = y.f_adaptive_avg_pool2d(&[1, 1]).unwrap().flatten(1, -1);
        let logits = pooled.matmul(&self.weight.transpose(0, 1)) + self.bias.reshape([1, -1]);
        if train {
            (logits.shallow_clone(), logits)
        } else {
            (logits.softmax(-1, tch::Kind::Float), logits)
        }
    }
}

/// 语义分割头（YOLO26-sem）。
pub struct SemSegHead {
    pub nc: i64,
    pub classifier: (Conv, RawOut),
    pub aux_head: Option<(Conv, RawOut)>,
}

impl SemSegHead {
    pub fn init(p: &VarPath, nc: i64, ch: &[i64]) -> Self {
        let c_mid = ch[0];
        let classifier = (
            Conv::init(&p.child("classifier").idx(0), c_mid, c_mid, 3, 1, None, 1, 1, true),
            RawOut::init(&p.child("classifier").idx(1), c_mid, nc),
        );
        let aux_head = if ch.len() > 1 {
            Some((
                Conv::init(&p.child("aux_head").idx(0), ch[1], c_mid, 3, 1, None, 1, 1, true),
                RawOut::init(&p.child("aux_head").idx(1), c_mid, nc),
            ))
        } else {
            None
        };
        SemSegHead { nc, classifier, aux_head }
    }

    /// 训练：返回 (main, aux?)；推理：logits。
    pub fn forward(&self, x: &[Tensor], train: bool) -> (Tensor, Option<Tensor>) {
        let logits = self.classifier.1.forward(&self.classifier.0.forward(&x[0], train));
        let aux = if train {
            self.aux_head.as_ref().map(|(a, b)| b.forward(&a.forward(&x[1], train)))
        } else {
            None
        };
        (logits, aux)
    }
}
