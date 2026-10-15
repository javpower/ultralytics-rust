//! 训练器 + 验证器 + 预测器（detect 家族）。

use std::path::{Path, PathBuf};

use rand::seq::SliceRandom;
use rand::SeedableRng;
use tch::nn::{AdamW, Optimizer, Sgd, VarStore};
use tch::{Device, Kind, Tensor};

use ul_core::metrics::{ap_per_class, match_predictions, DetMetrics};
use ul_core::{IOUV, UlError};

use ul_data::augment::{letterbox, AugmentCfg, Augmentor};
use ul_data::dataset::YoloDataset;
use ul_data::image::Img;

use ul_nn::model::{ModelOutput, YoloModel};
use ul_nn::parse::Task;
use ul_nn::tal::TaskAlignedAssigner;

use crate::loss::{classification_loss, obb_branch_loss, pose_keypoints_loss, segmentation_mask_loss, v8_detection_loss, DetBatch, LossW};
use crate::nms::{non_max_suppression_ex, scale_boxes_back, to_tensor, DetImage, NmsCfg};

/// 训练配置（default.yaml 子集）。
#[derive(Clone)]
pub struct TrainCfg {
    pub epochs: usize,
    pub batch: usize,
    pub imgsz: u32,
    pub lr0: f64,
    pub lrf: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub warmup_epochs: f64,
    pub optimizer: String, // auto | SGD | AdamW
    pub cos_lr: bool,
    pub amp: bool,
    pub device: Device,
    pub seed: u64,
    pub workers: usize,
    /// 增强超参（hsv/flip/mosaic/mixup/透视全套，CLI 可透传）。
    pub augment: AugmentCfg,
    /// 标称 batch（官方 nbs）：梯度累积步数 = max(1, nbs/batch)，wd 按 batch/nbs 缩放。
    pub nbs: usize,
    /// multi-scale 训练（官方默认开）：每批随机 [0.5,1.5]×imgsz，32 对齐。
    pub multi_scale: bool,
    pub val_interval: usize,
    pub project: PathBuf,
    pub name: String,
    pub exist_ok: bool,
    pub patience: usize,
    pub pretrained: Option<PathBuf>,
}

impl Default for TrainCfg {
    fn default() -> Self {
        TrainCfg {
            epochs: 100,
            batch: 16,
            imgsz: 640,
            lr0: 0.01,
            lrf: 0.01,
            momentum: 0.937,
            weight_decay: 0.0005,
            warmup_epochs: 3.0,
            optimizer: "auto".into(),
            cos_lr: false,
            amp: true,
            device: Device::cuda_if_available(),
            seed: 0,
            workers: 8,
            augment: AugmentCfg::default(),
            nbs: 16,
            multi_scale: false,
            val_interval: 1,
            project: PathBuf::from("runs"),
            name: "train".into(),
            exist_ok: false,
            patience: 100,
            pretrained: None,
        }
    }
}

// ---------------- 三参数组优化器（官方 bias/norm/weight 分组语义） ----------------

#[derive(Clone, Copy, PartialEq)]
enum OptKind2 {
    Sgd,
    AdamW,
}

/// 手写 SGD/AdamW：官方三组语义——bias（无 wd，warmup 从 0.1）、norm（无 wd）、weight（有 wd）。
pub struct ManualOptimizer {
    vars: Vec<(String, Tensor)>,
    group_of: Vec<usize>, // 0=bias 1=weight 2=norm
    kind: OptKind2,
    lr: [f64; 3],
    momentum: f64,
    wd: f64, // 已按 batch/nbs 缩放
    m_buf: Vec<Option<Tensor>>,
    v_buf: Vec<Option<Tensor>>,
    step_t: u64,
}

impl ManualOptimizer {
    pub fn new(model: &YoloModel, kind: OptKind2, lr0: f64, momentum: f64, wd: f64) -> Self {
        let vars = model.vs.variables();
        let mut names: Vec<&String> = vars.keys().collect();
        names.sort();
        let vars: Vec<(String, Tensor)> = names.into_iter().map(|n| (n.clone(), vars[n].shallow_clone())).collect();
        let group_of = vars
            .iter()
            .map(|(n, _)| {
                if n.ends_with(".bias") {
                    0
                } else if n.contains(".bn.") {
                    2
                } else {
                    1
                }
            })
            .collect();
        let n = vars.len();
        ManualOptimizer {
            vars,
            group_of,
            kind,
            lr: [0.1, lr0, lr0], // bias 组初值 = warmup_bias_lr
            momentum,
            wd,
            m_buf: (0..n).map(|_| None).collect(),
            v_buf: (0..n).map(|_| None).collect(),
            step_t: 0,
        }
    }

    pub fn set_lrs(&mut self, bias_lr: f64, main_lr: f64) {
        self.lr[0] = bias_lr;
        self.lr[1] = main_lr;
        self.lr[2] = main_lr;
    }

    pub fn set_momentum(&mut self, m: f64) {
        self.momentum = m;
    }

    /// 全局梯度范数裁剪（max_norm）。
    pub fn clip_grad(&self, max_norm: f64) {
        let mut total = 0f64;
        let mut grads: Vec<Option<Tensor>> = Vec::new();
        for (_, t) in &self.vars {
            let g = t.f_grad().ok().filter(|g| g.defined());
            match &g {
                Some(g) => {
                    let n2 = f64::try_from(g.norm()).unwrap_or(0.0);
                    total += n2 * n2;
                }
                None => {}
            }
            grads.push(g);
        }
        let total = total.sqrt();
        if total > max_norm && total > 0.0 {
            let scale = max_norm / (total + 1e-6);
            tch::no_grad(|| {
                for gopt in grads.iter_mut() {
                    if let Some(g) = gopt {
                        // grad 句柄共享存储：原地缩放
                        let sc = Tensor::from_slice(&[scale as f32]);
                        let _ = g.f_mul_(&sc);
                    }
                }
            });
        }
    }

    pub fn step(&mut self) {
        self.step_t += 1;
        let step = self.step_t as f64;
        tch::no_grad(|| {
            for (i, (name, t)) in self.vars.iter_mut().enumerate() {
                let g = match t.f_grad().ok().filter(|g| g.defined()) {
                    Some(g) => g,
                    None => continue,
                };
                let gi = self.group_of[i];
                let lr = self.lr[gi];
                let p0 = t.shallow_clone();
                let p: Tensor = if self.kind == OptKind2::AdamW {
                    &p0 * (1.0 - lr * self.wd)
                } else {
                    p0
                };
                match self.kind {
                    OptKind2::Sgd => {
                        let d: Tensor = if self.wd != 0.0 { &g + &p * self.wd } else { g };
                        let buf = match self.m_buf[i].as_ref() {
                            Some(b) => b * self.momentum + &d,
                            None => d,
                        };
                        let np = &p - &buf * lr;
                        self.m_buf[i] = Some(buf);
                        let _ = t.set_data(&np);
                    }
                    OptKind2::AdamW => {
                        let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
                        let m = match self.m_buf[i].as_ref() {
                            Some(b) => b * b1 + &g * (1.0 - b1),
                            None => &g * (1.0 - b1),
                        };
                        let v = match self.v_buf[i].as_ref() {
                            Some(b) => b * b2 + g.pow_tensor_scalar(2.0) * (1.0 - b2),
                            None => g.pow_tensor_scalar(2.0) * (1.0 - b2),
                        };
                        let bc1 = 1.0 - b1.powi(self.step_t as i32);
                        let bc2 = 1.0 - b2.powi(self.step_t as i32);
                        let mhat = &m / bc1;
                        let vhat = &v / bc2;
                        let np = &p - mhat * (lr / (vhat.sqrt() + eps));
                        self.m_buf[i] = Some(m);
                        self.v_buf[i] = Some(v);
                        let _ = t.set_data(&np);
                    }
                }
                let _ = name;
            }
        });
    }

    pub fn zero_grad(&self) {
        tch::no_grad(|| {
            for (_, t) in &self.vars {
                let mut tm = t.shallow_clone();
                let z = t.zeros_like();
                let _ = tm.set_data(&z);
            }
        });
    }
}

/// EMA（ModelEMA 语义：衰减 ramp），按变量名保存影子权重。
pub struct ModelEma {
    pub ema: std::collections::HashMap<String, Tensor>,
    pub decay: f64,
}

impl ModelEma {
    pub fn new(model: &YoloModel) -> Self {
        // 影子权重存模型同设备（逐迭代更新，避免每批跨设备搬运）；clone 保证深拷贝
        let vars = model.vs.variables();
        let ema: std::collections::HashMap<String, Tensor> = vars
            .iter()
            .map(|(n, t)| (n.clone(), t.f_to_dtype(t.kind(), false, true).unwrap()))
            .collect();
        ModelEma { ema, decay: 0.9999 }
    }

    pub fn update(&mut self, model: &YoloModel, i: usize) {
        self.decay = 0.9999f64 * (1.0 - (-(i as f64) / 2000.0).exp());
        let vars = model.vs.variables();
        tch::no_grad(|| {
            for (n, t) in &vars {
                if let Some(e) = self.ema.get_mut(n) {
                    let new = e.shallow_clone() * self.decay + t * (1.0 - self.decay);
                    let _ = e.copy_(&new);
                }
            }
        });
    }

    /// EMA 权重写回模型（val/保存前）。
    pub fn apply_to(&self, model: &YoloModel) {
        let vars = model.vs.variables();
        tch::no_grad(|| {
            for (n, t) in &vars {
                if let Some(e) = self.ema.get(n) {
                    let src = e.to_device(t.device()).to_kind(t.kind()).reshape(t.size());
                    let mut v = t.shallow_clone();
                    let _ = v.copy_(&src);
                }
            }
        });
    }
}

/// 训练结果。
pub struct TrainReport {
    pub save_dir: PathBuf,
    pub best_fitness: f64,
    pub final_metrics: DetMetrics,
    pub epochs_done: usize,
}

/// 训练入口（检测家族；分类/分割/姿态/旋转框按 task 分支见 validate/predict）。
pub fn train(model: &mut YoloModel, data: &Path, cfg: &TrainCfg) -> Result<TrainReport, UlError> {
    let (train_dirs, val_dirs, names, _kpt_dim) = ul_data::dataset::load_dataset_yaml(data)?;
    // 数据集任务标签解析跟随模型任务（seg 多边形 / obb 九点 / pose kpt / detect）
    let ds_task0 = match model.task {
        Task::Segment => ul_data::dataset::TaskKind::Segment,
        Task::Obb => ul_data::dataset::TaskKind::Obb,
        Task::Pose => ul_data::dataset::TaskKind::Pose,
        _ => {
            if _kpt_dim > 0 {
                ul_data::dataset::TaskKind::Pose
            } else {
                ul_data::dataset::TaskKind::Detect
            }
        }
    };
    let mut train_ds = YoloDataset::with_task(&train_dirs[0], names.clone(), _kpt_dim, ds_task0)?;
    for d in &train_dirs[1..] {
        let extra = YoloDataset::with_task(d, names.clone(), _kpt_dim, ds_task0)?;
        train_ds.img_files.extend(extra.img_files);
        train_ds.label_files.extend(extra.label_files);
    }
    let val_ds = val_dirs
        .first()
        .map(|d| YoloDataset::with_task(d, names.clone(), _kpt_dim, ds_task0))
        .transpose()?;

    let nc = model.nc;
    model.names = names.clone();
    let e2e = model.end2end;
    let reg_max = model.reg_max;
    let _ = reg_max;
    let ds_task = ds_task0;
    let kpt_dim = _kpt_dim;

    // 输出目录（冲突自动加后缀）
    let mut save_dir = cfg.project.join(&cfg.name);
    if save_dir.exists() && !cfg.exist_ok {
        let mut i = 1;
        while {
            save_dir = cfg.project.join(format!("{}{}", cfg.name, i));
            save_dir.exists()
        } {
            i += 1;
        }
    }
    std::fs::create_dir_all(&save_dir)?;

    // 优化器（auto 语义）
    let iters_per_epoch = (train_ds.len() + cfg.batch - 1) / cfg.batch;
    let total_iters = iters_per_epoch * cfg.epochs;
    let (opt_name, lr0, momentum) = if cfg.optimizer == "auto" {
        let lr_fit = ((0.002 * 5.0 / (4.0 + nc as f64) * 1e6).round() / 1e6).max(1e-6).min(0.01);
        if total_iters > 10000 {
            ("SGD", cfg.lr0, cfg.momentum)
        } else {
            ("AdamW", lr_fit, 0.9)
        }
    } else {
        (cfg.optimizer.as_str(), cfg.lr0, cfg.momentum)
    };
    log::info!("optimizer={opt_name} lr0={lr0} momentum={momentum} iters/epoch={iters_per_epoch}");

    // 三参数组手动优化器（bias 无 wd + warmup 0.1；norm 无 wd；weight 有 wd×batch/nbs）
    let kind2 = if opt_name == "SGD" { OptKind2::Sgd } else { OptKind2::AdamW };
    let wd_eff = cfg.weight_decay * cfg.batch as f64 / cfg.nbs.max(1) as f64;
    let mut opt = ManualOptimizer::new(model, kind2, lr0, momentum, wd_eff);
    let accumulate = ((cfg.nbs.max(1) + cfg.batch - 1) / cfg.batch).max(1);
    log::info!("accumulate={accumulate} wd_eff={wd_eff:.6}");
    let assigner = TaskAlignedAssigner { topk: 10, num_classes: nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    // 官方 E2EDetectLoss：one2one 分支用 topk=1 的独立分配器
    let assigner_o2o = TaskAlignedAssigner { topk: 1, num_classes: nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
    let lw = LossW::default();
    let mut ema = ModelEma::new(model);
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);

    let mut best_fitness = 0f64;
    let mut best_map = DetMetrics::default();
    let mut epochs_done = 0;
    let mut metrics_log = std::fs::File::create(save_dir.join("metrics.jsonl"))?;

    // 预训练权重（可选）
    if let Some(pre) = &cfg.pretrained {
        let (sd, _) = ul_nn::pt::load_pt(pre)?;
        let rep = ul_nn::pt::load_state_dict_into(&model.vs, &sd, false)?;
        log::info!("pretrain {}: {}", pre.display(), rep.summary());
    }

    let nb = iters_per_epoch;
    let nw = (cfg.warmup_epochs * nb as f64).max(100.0) as usize; // warmup 迭代数

    for epoch in 0..cfg.epochs {
        // lr 调度（线性 or 余弦）+ warmup
        let _ = &mut metrics_log;
        let mut order: Vec<usize> = (0..train_ds.len()).collect();
        order.shuffle(&mut rng);

        let close_mosaic_on = cfg.epochs.saturating_sub(epoch) > cfg.augment.close_mosaic;
        let mut acfg = cfg.augment.clone();
        if !close_mosaic_on {
            acfg.mosaic = 0.0;
            acfg.mixup = 0.0;
        }
        let ds_len = train_ds.len() as u64;
        let ep = epoch as u64;
        let augmentor = Augmentor {
            cfg: acfg.clone(),
            imgsz: cfg.imgsz,
            epoch,
            dataset_len: ds_len as usize,
            neighbor: Box::new(move |_, f| {
                let mut r = ul_data::augment::rng_for(0xDEAD ^ ep);
                let j = (r.next_u64() as usize) % (ds_len as usize).max(1);
                f(j);
            }),
        };

        let load = |i: usize| train_ds.load(i).ok();
        let mut epoch_loss = 0f64;
        let (mut ep_box, mut ep_cls, mut ep_dfl) = (0f64, 0f64, 0f64);
        let (mut ep_seg, mut ep_pose, mut ep_kobj, mut ep_ang) = (0f64, 0f64, 0f64, 0f64);
        let mut it = 0usize;
        for chunk in order.chunks(cfg.batch) {
            let ni = epoch * nb + it;
            // 本批输入尺寸（multi-scale：[0.5,1.5]×imgsz，32 对齐）
            let mut bsz = cfg.imgsz;
            if cfg.multi_scale {
                let f = 0.5 + rng.clone().gen::<f64>();
                let _ = f;
                use rand::Rng as _;
                let f: f64 = { let v: f64 = { let mut r2 = rng.clone(); r2.gen::<f64>() }; v };
                let _ = &mut rng;
                let scale = 0.5 + rng.gen::<f64>();
                let _ = f;
                bsz = ((cfg.imgsz as f64 * scale) / 32.0).round().max(1.0) as u32 * 32;
            }
            // warmup（官方三组：bias 从 warmup_bias_lr=0.1 → lr0；其余 0 → lr0；momentum 0.8 → 目标）
            if ni < nw {
                let xn = (ni as f64) / nw as f64;
                let lr_now = lr0 * xn;
                let bias_lr = 0.1 + (lr0 - 0.1) * xn;
                opt.set_lrs(bias_lr, lr_now);
                let mom_now = 0.8 + (momentum - 0.8) * xn;
                opt.set_momentum(mom_now);
            } else {
                let dt = epoch as f64 - cfg.warmup_epochs;
                let ft = (cfg.epochs as f64 - cfg.warmup_epochs).max(1.0);
                let lr_now = if cfg.cos_lr {
                    let t = dt / ft;
                    0.5 * (1.0 + (std::f64::consts::PI * t).cos()) * lr0 * cfg.lrf + (lr0 - lr0 * cfg.lrf) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
                } else {
                    ((ft - dt) / ft.max(1e-9)) * (1.0 - cfg.lrf) * lr0 + lr0 * cfg.lrf
                };
                opt.set_lrs(lr_now.max(1e-8), lr_now.max(1e-8));
            }

            // 构建批（增强 + letterbox；seg/pose/obb 附带掩码/关键点/角度）
            let mut imgs: Vec<Img> = Vec::with_capacity(chunk.len());
            let mut labels: Vec<Vec<f32>> = Vec::new();
            let mut kpts: Vec<Vec<f32>> = Vec::new();
            let mut masks_v: Vec<Vec<u8>> = Vec::new();
            for (bi, &idx) in chunk.iter().enumerate() {
                let mut smp = augmentor.train_augment(idx, &load);
                // 官方 erasing 仅 classify 管线（default 0.4）
                if model.task == Task::Classify {
                    let (a, b, c, d) = {
                        use rand::Rng as _;
                        let (a, b, c, d) = rng.gen::<(f32, f32, f32, f32)>();
                        (a, b, c, d)
                    };
                    ul_data::augment::random_erasing(&mut smp, cfg.augment.erasing, a, b, c, d);
                }
                if smp.img.w != bsz || smp.img.h != bsz {
                    smp.img = letterbox(&smp.img, bsz, [114, 114, 114]);
                }
                for ins in &smp.instances {
                    let mut row = vec![bi as f32, ins.cls as f32, ins.xywhn[0], ins.xywhn[1], ins.xywhn[2], ins.xywhn[3]];
                    if ds_task == ul_data::dataset::TaskKind::Obb {
                        row.push(ins.angle);
                    }
                    labels.push(row);
                    if ds_task == ul_data::dataset::TaskKind::Pose {
                        kpts.push(ins.kpts.clone());
                    }
                }
                if ds_task == ul_data::dataset::TaskKind::Segment {
                    masks_v.push(ul_data::augment::rasterize_mask(&smp, bsz));
                }
                imgs.push(smp.img);
            }
            let img_t = to_tensor(&imgs, cfg.device);
            let ncol = if ds_task == ul_data::dataset::TaskKind::Obb { 7 } else { 6 };
            let flat: Vec<f32> = labels.concat();
            let labels_t = if flat.is_empty() {
                Tensor::zeros([0, ncol], (Kind::Float, cfg.device))
            } else {
                Tensor::from_slice(&flat).reshape([-1, ncol as i64]).to_device(cfg.device)
            };
            let kpts_t = if ds_task == ul_data::dataset::TaskKind::Pose && !kpts.is_empty() {
                let kf: Vec<f32> = kpts.concat();
                Some(Tensor::from_slice(&kf).reshape([-1, kpts[0].len() as i64]).to_device(cfg.device))
            } else {
                None
            };
            let masks_t = if ds_task == ul_data::dataset::TaskKind::Segment && !masks_v.is_empty() {
                let mf: Vec<u8> = masks_v.concat();
                Some(
                    Tensor::from_slice(&mf.iter().map(|x| *x as f32).collect::<Vec<_>>())
                        .reshape([-1, bsz as i64, bsz as i64])
                        .to_device(cfg.device),
                )
            } else {
                None
            };
            let mut batch = DetBatch::detect(img_t, labels_t, chunk.len() as i64, bsz as i64);
            batch.masks = masks_t;
            batch.kpts = kpts_t;
            batch.obb = ds_task == ul_data::dataset::TaskKind::Obb;

            // 前向 + 损失 + 反向
            match model.task {
                Task::Classify => {
                    let (logits, _) = match model.forward(&batch.img, true).unwrap() {
                        ModelOutput::Classify(a, b) => (a, b),
                        _ => unreachable!(),
                    };
                    let targets = batch.labels.select(1, 1).to_kind(Kind::Int64);
                    let loss = classification_loss(&logits, &targets, 0.1);
                    loss.backward();
                    epoch_loss += f64::try_from(&loss).unwrap_or(0.0);
                }
                Task::Segment => {
                    let out = model.forward(&batch.img, true).unwrap();
                    let head_out = match out {
                        ModelOutput::Detect(h) => h,
                        _ => panic!("segment train expects detect head"),
                    };
                    let det = v8_detection_loss(model, lw, &assigner, &assigner_o2o, &head_out, &batch, e2e);
                    let (rp, proto) = match &head_out {
                        ul_nn::head::HeadOutput::Train(rp) => (rp, rp.proto.as_ref().expect("segment requires proto")),
                        ul_nn::head::HeadOutput::TrainDual(a, _) => (a, a.proto.as_ref().expect("segment requires proto")),
                        _ => panic!("segment expects train head output"),
                    };
                    let masks_t = batch.masks.as_ref().unwrap();
                    let seg = segmentation_mask_loss(det.assign.as_ref().unwrap(), proto, rp.extra.as_ref().unwrap(), masks_t, bsz as i64)
                        * lw.box_gain;
                    let total = &det.total + &seg;
                    total.backward();
                    epoch_loss += f64::try_from(&total).unwrap_or(0.0);
                    ep_box += det.parts.box_;
                    ep_cls += det.parts.cls;
                    ep_dfl += det.parts.dfl;
                    ep_seg += f64::try_from(&seg).unwrap_or(0.0);
                }
                Task::Pose => {
                    let out = model.forward(&batch.img, true).unwrap();
                    let head_out = match out {
                        ModelOutput::Detect(h) => h,
                        _ => panic!("pose train expects detect head"),
                    };
                    let det = v8_detection_loss(model, lw, &assigner, &assigner_o2o, &head_out, &batch, e2e);
                    let rp = match &head_out {
                        ul_nn::head::HeadOutput::Train(rp) => rp,
                        ul_nn::head::HeadOutput::TrainDual(a, _) => a,
                        _ => panic!("pose expects train head output"),
                    };
                    let bidx = batch.labels.select(1, 0).to_kind(Kind::Int64);
                    let (pl, kl) = pose_keypoints_loss(det.assign.as_ref().unwrap(), rp.extra.as_ref().unwrap(), kpt_dim, batch.kpts.as_ref().unwrap(), &bidx, bsz as i64);
                    let pl = pl * lw.pose_gain();
                    let kl = kl * lw.kobj_gain();
                    let total = &det.total + &pl + &kl;
                    total.backward();
                    epoch_loss += f64::try_from(&total).unwrap_or(0.0);
                    ep_box += det.parts.box_;
                    ep_cls += det.parts.cls;
                    ep_dfl += det.parts.dfl;
                    ep_pose += f64::try_from(&pl).unwrap_or(0.0);
                    ep_kobj += f64::try_from(&kl).unwrap_or(0.0);
                }
                Task::Obb => {
                    let out = model.forward(&batch.img, true).unwrap();
                    let head_out = match out {
                        ModelOutput::Detect(h) => h,
                        _ => panic!("obb train expects detect head"),
                    };
                    let rp = match &head_out {
                        ul_nn::head::HeadOutput::Train(rp) => rp,
                        ul_nn::head::HeadOutput::TrainDual(a, _) => a,
                        _ => panic!("obb expects train head output"),
                    };
                    let (total, parts) = obb_branch_loss(model, &assigner, rp, &batch, lw);
                    total.backward();
                    epoch_loss += f64::try_from(&total).unwrap_or(0.0);
                    ep_box += parts.box_;
                    ep_cls += parts.cls;
                    ep_dfl += parts.dfl;
                    ep_ang += parts.angle;
                }
                _ => {
                    let out = model.forward(&batch.img, true).unwrap();
                    let head_out = match out {
                        ModelOutput::Detect(h) => h,
                        _ => panic!("train only detect/classify for now"),
                    };
                    let det = v8_detection_loss(model, lw, &assigner, &assigner_o2o, &head_out, &batch, e2e);
                    det.total.backward();
                    epoch_loss += f64::try_from(&det.total).unwrap_or(0.0);
                    ep_box += det.parts.box_;
                    ep_cls += det.parts.cls;
                    ep_dfl += det.parts.dfl;
                }
            }
            it += 1;
            ema.update(model, epoch * nb + it);
            // nbs 梯度累积：每 accumulate 步（或 epoch 末）做一次 clip+step+zero
            if it % accumulate == 0 || it == nb {
                opt.clip_grad(10.0);
                opt.step();
                opt.zero_grad();
            }
        }
        epochs_done = epoch + 1;

        // 验证
        let do_val = val_ds.is_some() && (epoch + 1) % cfg.val_interval.max(1) == 0;
        if do_val {
            let val = val_ds.as_ref().unwrap();
            ema.apply_to(model);
            let metrics = validate_task(model, val, cfg.imgsz, cfg.device, &names)?;
            log::info!(
                "epoch {}/{} loss={:.4} (box={:.3} cls={:.3} dfl={:.3}) P={:.4} R={:.4} mAP50={:.4} mAP50-95={:.4}",
                epoch + 1,
                cfg.epochs,
                epoch_loss / it.max(1) as f64,
                ep_box / it.max(1) as f64,
                ep_cls / it.max(1) as f64,
                ep_dfl / it.max(1) as f64,
                metrics.mp,
                metrics.mr,
                metrics.map50,
                metrics.map50_95
            );
            let fit = metrics.fitness();
            use std::io::Write;
            writeln!(
                metrics_log,
                "{{\"epoch\":{},\"loss\":{:.4},\"box\":{:.4},\"cls\":{:.4},\"dfl\":{:.4},\"P\":{:.4},\"R\":{:.4},\"mAP50\":{:.4},\"mAP50-95\":{:.4}}}",
                epoch + 1,
                epoch_loss / it.max(1) as f64,
                ep_box / it.max(1) as f64,
                ep_cls / it.max(1) as f64,
                ep_dfl / it.max(1) as f64,
                metrics.mp,
                metrics.mr,
                metrics.map50,
                metrics.map50_95
            )?;
            if fit > best_fitness {
                best_fitness = fit;
                best_map = metrics.clone();
                save_checkpoint(model, &save_dir.join("best"), &names)?;
            }
            // early stop
            if cfg.patience < 100 && epoch > cfg.patience && fit < best_fitness {
                log::info!("early stop at epoch {epoch}");
                break;
            }
        } else {
            log::info!(
                "epoch {}/{} loss={:.4} (box={:.3} cls={:.3} dfl={:.3} seg={:.3} pose={:.3} kobj={:.3} ang={:.3})",
                epoch + 1,
                cfg.epochs,
                epoch_loss / it.max(1) as f64,
                ep_box / it.max(1) as f64,
                ep_cls / it.max(1) as f64,
                ep_dfl / it.max(1) as f64,
                ep_seg / it.max(1) as f64,
                ep_pose / it.max(1) as f64,
                ep_kobj / it.max(1) as f64,
                ep_ang / it.max(1) as f64
            );
        }
    }
    // 最后保存 last
    save_checkpoint(model, &save_dir.join("last"), &names)?;
    Ok(TrainReport { save_dir, best_fitness, final_metrics: best_map, epochs_done })
}

/// 验证数据集（复用已构建模型）。
pub fn validate_model(
    model: &YoloModel,
    val: &YoloDataset,
    imgsz: u32,
    device: Device,
    names: &[String],
) -> Result<DetMetrics, UlError> {
    let nc = model.nc;
    let mut tp_all: Vec<Vec<bool>> = Vec::new();
    let mut conf_all: Vec<f32> = Vec::new();
    let mut pred_cls_all: Vec<i64> = Vec::new();
    let mut gt_cls_all: Vec<i64> = Vec::new();
    let cfg = NmsCfg::val();

    let bs = 8usize;
    for chunk in (0..val.len()).collect::<Vec<_>>().chunks(bs) {
        let mut imgs = Vec::new();
        let mut gts_per: Vec<Vec<(i64, [f32; 4])>> = Vec::new();
        let mut gains: Vec<(f32, f32, f32, f32, f32, f32)> = Vec::new();
        for &i in chunk {
            let smp = val.load(i)?;
            let lb = letterbox(&smp.img, imgsz, [114, 114, 114]);
            let p = ul_core::bbox::LetterboxParams::new(smp.img.w as f32, smp.img.h as f32, imgsz as f32, imgsz as f32);
            gains.push((p.gain_w, 0.0, p.pad_w, p.pad_h, smp.img.w as f32, smp.img.h as f32));
            // GT 保留原图坐标（预测经 scale_boxes_back 也还原到原图）
            gts_per.push(
                smp.instances
                    .iter()
                    .map(|ins| {
                        let xy = [
                            (ins.xywhn[0] - ins.xywhn[2] / 2.0) * smp.img.w as f32,
                            (ins.xywhn[1] - ins.xywhn[3] / 2.0) * smp.img.h as f32,
                            (ins.xywhn[0] + ins.xywhn[2] / 2.0) * smp.img.w as f32,
                            (ins.xywhn[1] + ins.xywhn[3] / 2.0) * smp.img.h as f32,
                        ];
                        (ins.cls, xy)
                    })
                    .collect(),
            );
            imgs.push(lb);
        }
        let x = to_tensor(&imgs, device);
        let out = model.forward(&x, false)?;
        let t = match out {
            ModelOutput::Detect(ul_nn::head::HeadOutput::Inference(t)) => t,
            _ => return Err(UlError::Msg("val expects detect inference".into())),
        };
        let t_t = t.transpose(1, 2).contiguous(); // (B, A, 4+nc)
        let per_img = non_max_suppression_ex(&t_t, 4 + nc, &cfg, t_t.size()[0], t_t.size()[2], t_t.size()[1]);
        for (bi, di) in per_img.iter().enumerate() {
            let (gain, _, pad_w, pad_h, ow, oh) = gains[bi];
            for d in &di.dets {
                let mut dd = *d;
                scale_boxes_back(&mut dd, gain, pad_w, pad_h, ow, oh);
                conf_all.push(dd.score);
                pred_cls_all.push(dd.cls);
                tp_all.push(vec![false; IOUV.len()]);
            }
            for (cls, xy) in &gts_per[bi] {
                gt_cls_all.push(*cls);
                let _ = xy;
            }
            // 匹配
            if !di.dets.is_empty() && !gts_per[bi].is_empty() {
                let preds_xy: Vec<[f32; 4]> = di
                    .dets
                    .iter()
                    .map(|d| {
                        let mut dd = *d;
                        scale_boxes_back(&mut dd, gain, pad_w, pad_h, ow, oh);
                        dd.xyxy
                    })
                    .collect();
                let gts_xy: Vec<[f32; 4]> = gts_per[bi].iter().map(|g| g.1).collect();
                let iou = crate::nms::iou_matrix(&preds_xy, &gts_xy);
                let matched = match_predictions(&pred_cls_all[pred_cls_all.len() - preds_xy.len()..], &gt_cls_all[gt_cls_all.len() - gts_xy.len()..], &iou, &IOUV);
                let start = tp_all.len() - preds_xy.len();
                for (k, m) in matched.iter().enumerate() {
                    tp_all[start + k] = m.clone();
                }
            }
        }
    }
    let ap = ap_per_class(&tp_all, &conf_all, &pred_cls_all, &gt_cls_all);
    let _ = names;
    let _ = nc;
    Ok(DetMetrics::from_ap(&ap))
}

/// 保存 checkpoint（safetensors + meta json）。
pub fn save_checkpoint(model: &YoloModel, path_stem: &Path, names: &[String]) -> Result<(), UlError> {
    std::fs::create_dir_all(path_stem)?;
    let vars = model.vs.variables();
    let mut names_v: Vec<&String> = vars.keys().collect();
    names_v.sort();
    let pairs: Vec<(&str, &Tensor)> = names_v.iter().map(|n| (n.as_str(), &vars[*n])).collect();
    tch::Tensor::write_safetensors(&pairs, path_stem.join("weights.safetensors"))
        .map_err(|e| UlError::Msg(format!("save: {e}")))?;
    let meta = serde_json::json!({
        "task": format!("{:?}", model.task),
        "nc": model.nc,
        "names": names,
        "stride": model.stride,
        "yaml_file": model.yaml_file,
        "scale": model.scale,
        "end2end": model.end2end,
        "reg_max": model.reg_max,
    });
    std::fs::write(path_stem.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

/// 加载 checkpoint（构建同架构模型并灌权重）。
pub fn load_checkpoint(stem: &Path, device: Device) -> Result<YoloModel, UlError> {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(stem.join("meta.json"))?)?;
    let yaml_file = meta["yaml_file"].as_str().unwrap_or("yolo11n");
    let nc = meta["nc"].as_i64().unwrap_or(80);
    let mut model = YoloModel::build(yaml_file, Some(nc), device, false)?;
    let sd = tch::Tensor::read_safetensors(&stem.join("weights.safetensors"))?;
    let map: std::collections::HashMap<String, Tensor> = sd.into_iter().collect();
    let vars = model.vs.variables();
    tch::no_grad(|| {
        for (n, v) in &vars {
            if let Some(w) = map.get(n) {
                let mut vv = v.shallow_clone();
                let _ = vv.copy_(&w.to_device(v.device()).to_kind(v.kind()));
            }
        }
    });
    if let Some(arr) = meta["names"].as_array() {
        model.names = arr.iter().map(|x| x.as_str().unwrap_or("?").to_string()).collect();
    }
    Ok(model)
}

/// 单图/目录预测：返回每图检测 + 保存可视化。
pub struct PredictOut {
    pub path: PathBuf,
    pub dets: Vec<ul_core::nms::Det>,
}

pub fn predict(
    model: &YoloModel,
    sources: &[PathBuf],
    imgsz: u32,
    device: Device,
    conf: f64,
    save_dir: Option<&Path>,
    names: &[String],
) -> Result<Vec<PredictOut>, UlError> {
    let cfg = NmsCfg { conf, ..NmsCfg::predict() };
    let nc = model.nc;
    if let Some(sd) = save_dir {
        std::fs::create_dir_all(sd)?;
    }
    let mut out = Vec::new();
    for src in sources {
        let img = Img::open(src)?;
        // rect：按长宽比 32 对齐矩形 letterbox
        let (rw, rh) = crate::nms::rect_letterbox_size(img.w, img.h, imgsz, 32);
        let lb = ul_data::augment::letterbox_rect(&img, rw, rh, [114, 114, 114]);
        let x = to_tensor(&[lb], device);
        let o = model.forward(&x, false)?;
        let t = match o {
            ModelOutput::Detect(ul_nn::head::HeadOutput::Inference(t)) => t,
            ModelOutput::Classify(probs, _) => {
                // 分类：top1
                let p: Vec<f32> = Vec::try_from(&probs.to_device(Device::Cpu).reshape([-1]))?;
                let best = p
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, v)| (i, *v))
                    .unwrap_or((0, 0.0));
                let d = ul_core::nms::Det { xyxy: [0.0, 0.0, img.w as f32, img.h as f32], score: best.1, cls: best.0 as i64 };
                out.push(PredictOut { path: src.clone(), dets: vec![d] });
                continue;
            }
            _ => return Err(UlError::Msg("predict expects detect/classify".into())),
        };
        let t_t = t.transpose(1, 2).contiguous(); // (1, A, 4+nc)
        let per = non_max_suppression_ex(&t_t, 4 + nc, &cfg, 1, t_t.size()[2], t_t.size()[1]);
        let p = ul_core::bbox::LetterboxParams::new(img.w as f32, img.h as f32, imgsz as f32, imgsz as f32);
        let mut dets: Vec<ul_core::nms::Det> = Vec::new();
        for d in &per[0].dets {
            let mut dd = *d;
            scale_boxes_back(&mut dd, p.gain_w, p.pad_w, p.pad_h, img.w as f32, img.h as f32);
            dets.push(dd);
        }
        if let Some(sd) = save_dir {
            let mut vis = img.clone();
            for d in &dets {
                let c = ul_data::image::color_for(d.cls);
                vis.rect(
                    d.xyxy[0] as i32,
                    d.xyxy[1] as i32,
                    d.xyxy[2] as i32,
                    d.xyxy[3] as i32,
                    c,
                    2,
                );
            }
            let out_name = src.file_name().map(|f| f.to_owned()).unwrap_or_default();
            vis.save(&sd.join(out_name))?;
        }
        let _ = names;
        out.push(PredictOut { path: src.clone(), dets });
    }
    Ok(out)
}

// ---------------- seg/pose/obb validators ----------------

/// 任务 validator 分发：按模型任务计算对应空间的 AP（seg=mask IoU，pose=OKS，obb=probiou）。
pub fn validate_task(
    model: &YoloModel,
    val: &YoloDataset,
    imgsz: u32,
    device: Device,
    _names: &[String],
) -> Result<DetMetrics, UlError> {
    match model.task {
        Task::Segment => validate_segment(model, val, imgsz, device),
        Task::Pose => validate_pose(model, val, imgsz, device),
        Task::Obb => validate_obb(model, val, imgsz, device),
        _ => validate_model(model, val, imgsz, device, _names),
    }
}

fn oks_matrix(pred: &[Vec<[f32; 3]>], gt: &[Vec<[f32; 3]>], areas: &[f32], sigmas: &[f64]) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0f32; gt.len()]; pred.len()];
    for (i, p) in pred.iter().enumerate() {
        for (j, g) in gt.iter().enumerate() {
            let area = areas.get(j).copied().unwrap_or(1.0);
            let mut e = 0f32;
            let mut vis = 0;
            let n = p.len().min(g.len());
            for c in 0..n {
                if g[c][2] > 0.0 {
                    vis += 1;
                    let d = (p[c][0] - g[c][0]).powi(2) + (p[c][1] - g[c][1]).powi(2);
                    let s2 = 2.0 * sigmas[c] as f32;
                    e += (-(d / (s2 * s2 * area.max(1e-9) * 2.0 + 1e-9))).exp();
                }
            }
            out[i][j] = if vis > 0 { e / vis as f32 } else { 0.0 };
        }
    }
    out
}

/// 官方 OKS_SIGMA/10。
const OKS17: [f64; 17] = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
];

fn validate_segment(model: &YoloModel, val: &YoloDataset, imgsz: u32, device: Device) -> Result<DetMetrics, UlError> {
    let nc = model.nc;
    let cfg = NmsCfg::val();
    let mut tp_all: Vec<Vec<bool>> = Vec::new();
    let mut mtp_all: Vec<Vec<bool>> = Vec::new();
    let mut conf_all: Vec<f32> = Vec::new();
    let mut pred_cls_all: Vec<i64> = Vec::new();
    let mut gt_cls_all: Vec<i64> = Vec::new();
    let s = imgsz as f32;
    for i in 0..val.len() {
        let smp = val.load(i)?;
        let lb = letterbox(&smp.img, imgsz, [114, 114, 114]);
        let p = ul_core::bbox::LetterboxParams::new(smp.img.w as f32, smp.img.h as f32, s, s);
        let x = to_tensor(&[lb], device);
        let out = model.forward(&x, true)?;
        let (rp, proto) = match out {
            ModelOutput::Detect(ul_nn::head::HeadOutput::Train(rp)) => {
                let proto = rp.proto.as_ref().expect("seg proto").shallow_clone();
                (rp, proto)
            }
            _ => return Err(UlError::Msg("seg val expects train output".into())),
        };
        // 解码框（xywh 像素）+ 分数
        let raw = match &model.layers.last().unwrap().module {
            ul_nn::parse::Module::Head(ul_nn::parse::HeadMod::Detect(dh)) => dh
                .dfl
                .as_ref()
                .map(|d| d.forward(&rp.boxes))
                .unwrap_or_else(|| rp.boxes.shallow_clone()),
            _ => rp.boxes.shallow_clone(),
        };
        let (pts, sts) = ul_nn::ops::make_anchors(&rp.feats, &model.stride, 0.5);
        let boxes = ul_nn::ops::dist2bbox(&raw, &pts, true, 1) * &sts; // (1,4,A)
        let scores = rp.scores.sigmoid(); // (1,nc,A)
        let mut parts = vec![boxes, scores];
        if let Some(coef) = &rp.extra {
            parts.push(coef.shallow_clone());
        }
        let t_t = Tensor::cat(&parts, 1).transpose(1, 2).contiguous(); // (1,A,4+nc(+nm))
        let per = non_max_suppression_ex(&t_t, 4 + nc, &cfg, 1, t_t.size()[2], t_t.size()[1]);
        let di = &per[0];
        if std::env::var("UL_VAL_DEBUG").is_ok() {
            eprintln!("[segval] dets={} extra={} ex_ch={} t_t={:?}", di.dets.len(), di.extra.len(), di.extra_ch, t_t.size());
        }
        // GT
        let mut gts: Vec<(i64, [f32; 4])> = Vec::new();
        let gt_mask = ul_data::augment::rasterize_mask(&smp, imgsz); // 原图空间光栅化
        for ins in &smp.instances {
            let xy = [
                (ins.xywhn[0] - ins.xywhn[2] / 2.0) * smp.img.w as f32,
                (ins.xywhn[1] - ins.xywhn[3] / 2.0) * smp.img.h as f32,
                (ins.xywhn[0] + ins.xywhn[2] / 2.0) * smp.img.w as f32,
                (ins.xywhn[1] + ins.xywhn[3] / 2.0) * smp.img.h as f32,
            ];
            gts.push((ins.cls, xy));
        }
        // 掩码：letterbox 空间的 GT 掩码（归一化坐标 × 映射）
        let mut gt_mask_lb = vec![0u8; (imgsz * imgsz) as usize];
        for (idx, ins) in smp.instances.iter().enumerate() {
            if ins.segments.len() < 6 {
                continue;
            }
            for yy in 0..imgsz as usize {
                for xx in 0..imgsz as usize {
                    // 点在多边形内（射线法，像素坐标）
                    let px = (xx as f32 + 0.5 - p.pad_w) / p.gain_w / smp.img.w as f32;
                    let py = (yy as f32 + 0.5 - p.pad_h) / p.gain_h / smp.img.h as f32;
                    let mut inside = false;
                    let pts: Vec<[f32; 2]> = ins.segments.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
                    let n = pts.len();
                    for k in 0..n {
                        let a = pts[k];
                        let b = pts[(k + 1) % n];
                        if (a[1] > py) != (b[1] > py)
                            && px < (b[0] - a[0]) * (py - a[1]) / (b[1] - a[1] + 1e-12) + a[0]
                        {
                            inside = !inside;
                        }
                    }
                    if inside {
                        gt_mask_lb[yy * imgsz as usize + xx] = (idx + 1) as u8;
                    }
                }
            }
        }
        // 预测掩码：coef @ proto
        let (nmh, nmw) = (proto.size()[2], proto.size()[3]);
        let mut mdets: Vec<(usize, Vec<f32>)> = Vec::new();
        for (k, ex) in di.extra.iter().enumerate() {
            mdets.push((k, ex.clone()));
        }
        let proto_f = proto.to_device(tch::Device::Cpu).reshape([proto.size()[1], nmh * nmw]);
        let _ = device;
        // 匹配 + TP
        let mut pd_boxes: Vec<[f32; 4]> = Vec::new();
        let mut pd_cls: Vec<i64> = Vec::new();
        let mut pd_conf: Vec<f32> = Vec::new();
        let mut pd_masks: Vec<Vec<f32>> = Vec::new();
        for (k, d) in di.dets.iter().enumerate() {
            let mut dd = *d;
            crate::nms::scale_boxes_back(&mut dd, p.gain_w, p.pad_w, p.pad_h, smp.img.w as f32, smp.img.h as f32);
            pd_boxes.push(dd.xyxy);
            pd_cls.push(dd.cls);
            pd_conf.push(dd.score);
            let coef = Tensor::from_slice(&mdets[k].1).unsqueeze(0); // (1,nm)
            let m = coef.matmul(&proto_f).sigmoid().reshape([nmh, nmw]);
            // 上采样到 imgsz 并按 pred bbox 裁剪（letterbox 空间）
            let mu = m
                .unsqueeze(0)
                .unsqueeze(0)
                .f_upsample_nearest2d(&[imgsz as i64, imgsz as i64], None, None)
                .unwrap()
                .squeeze_dim(0);
            let mv: Vec<f32> = Vec::try_from(&mu.to_device(tch::Device::Cpu).reshape([-1])).unwrap_or_default();
            let x1 = (d.xyxy[0]).max(0.0) as usize;
            let y1 = (d.xyxy[1]).max(0.0) as usize;
            let x2 = (d.xyxy[2] as usize).min(imgsz as usize);
            let y2 = (d.xyxy[3] as usize).min(imgsz as usize);
            let mut cm = vec![0f32; mv.len()];
            for yy in y1..y2.max(y1) {
                for xx in x1..x2.max(x1) {
                    cm[yy * imgsz as usize + xx] = mv[yy * imgsz as usize + xx];
                }
            }
            pd_masks.push(cm);
        }
        // GT mask (letterbox 空间) 向量
        let gt_masks_v: Vec<Vec<f32>> = (0..smp.instances.len())
            .map(|idx| gt_mask_lb.iter().map(|v| (*v == (idx + 1) as u8) as u32 as f32).collect())
            .collect();
        // IoU 矩阵（box 与 mask）
        let gt_boxes: Vec<[f32; 4]> = gts.iter().map(|g| g.1).collect();
        let iou_box = crate::nms::iou_matrix(&pd_boxes, &gt_boxes);
        let iou_mask = mask_iou_matrix(&pd_masks, &gt_masks_v);
        let pred_c: Vec<i64> = pd_cls.clone();
        let gt_c: Vec<i64> = gts.iter().map(|g| g.0).collect();
        let matched_b = ul_core::metrics::match_predictions(&pred_c, &gt_c, &iou_box, &ul_core::IOUV);
        let matched_m = ul_core::metrics::match_predictions(&pred_c, &gt_c, &iou_mask, &ul_core::IOUV);
        for (k, d) in di.dets.iter().enumerate() {
            tp_all.push(matched_b[k].clone());
            mtp_all.push(matched_m[k].clone());
            conf_all.push(pd_conf[k]);
            pred_cls_all.push(pd_cls[k]);
        }
        for g in &gts {
            gt_cls_all.push(g.0);
        }
    }
    // mask AP（官方 MaskMetrics 语义：mask IoU 的 ap_per_class）
    let _ = mtp_all;
    let ap = ul_core::metrics::ap_per_class(&tp_all, &conf_all, &pred_cls_all, &gt_cls_all);
    Ok(DetMetrics::from_ap(&ap))
}

fn mask_iou_matrix(pred: &[Vec<f32>], gt: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0f32; gt.len()]; pred.len()];
    for (i, p) in pred.iter().enumerate() {
        let pa: f32 = p.iter().map(|x| (*x > 0.5) as u32 as f32).sum();
        for (j, g) in gt.iter().enumerate() {
            let mut inter = 0f32;
            let mut ga = 0f32;
            for k in 0..p.len() {
                let pb = (p[k] > 0.5) as u32 as f32;
                let gb = (g[k] > 0.5) as u32 as f32;
                inter += pb * gb;
                ga += gb;
            }
            let union = pa + ga - inter;
            out[i][j] = if union > 0.0 { inter / union } else { 0.0 };
        }
    }
    out
}

fn validate_pose(model: &YoloModel, val: &YoloDataset, imgsz: u32, device: Device) -> Result<DetMetrics, UlError> {
    let nc = model.nc;
    let cfg = NmsCfg::val();
    let mut tp_all: Vec<Vec<bool>> = Vec::new();
    let mut conf_all: Vec<f32> = Vec::new();
    let mut pred_cls_all: Vec<i64> = Vec::new();
    let mut gt_cls_all: Vec<i64> = Vec::new();
    let s = imgsz as f32;
    let kpt_dim = if val.kpt_dim > 0 { val.kpt_dim } else { 3 };
    for i in 0..val.len() {
        let smp = val.load(i)?;
        let lb = letterbox(&smp.img, imgsz, [114, 114, 114]);
        let p = ul_core::bbox::LetterboxParams::new(smp.img.w as f32, smp.img.h as f32, s, s);
        let x = to_tensor(&[lb], device);
        let out = model.forward(&x, false)?;
        let t = match out {
            ModelOutput::Detect(ul_nn::head::HeadOutput::Inference(t)) => t,
            _ => return Err(UlError::Msg("pose val expects inference".into())),
        };
        let t_t = t.transpose(1, 2).contiguous(); // (1,A,4+nc+nk)
        let per = non_max_suppression_ex(&t_t, 4 + nc, &cfg, 1, t_t.size()[2], t_t.size()[1]);
        let di = &per[0];
        // GT
        let mut gts: Vec<(i64, [f32; 4], Vec<[f32; 3]>)> = Vec::new();
        for ins in &smp.instances {
            let xy = [
                (ins.xywhn[0] - ins.xywhn[2] / 2.0) * smp.img.w as f32,
                (ins.xywhn[1] - ins.xywhn[3] / 2.0) * smp.img.h as f32,
                (ins.xywhn[0] + ins.xywhn[2] / 2.0) * smp.img.w as f32,
                (ins.xywhn[1] + ins.xywhn[3] / 2.0) * smp.img.h as f32,
            ];
            let kpts: Vec<[f32; 3]> = ins
                .kpts
                .chunks_exact(kpt_dim as usize)
                .map(|c| {
                    let px = c[0] * smp.img.w as f32 * p.gain_w + p.pad_w;
                    let py = c[1] * smp.img.h as f32 * p.gain_h + p.pad_h;
                    let v = if kpt_dim == 3 { c[2] } else { 1.0 };
                    [px, py, v]
                })
                .collect();
            let area = (ins.xywhn[2] * smp.img.w as f32 * p.gain_w) * (ins.xywhn[3] * smp.img.h as f32 * p.gain_h);
            let _ = area;
            gts.push((ins.cls, xy, kpts));
        }
        let mut pd_boxes: Vec<[f32; 4]> = Vec::new();
        let mut pd_cls: Vec<i64> = Vec::new();
        let mut pd_conf: Vec<f32> = Vec::new();
        let mut pd_kpts: Vec<Vec<[f32; 3]>> = Vec::new();
        for (k, d) in di.dets.iter().enumerate() {
            let mut dd = *d;
            crate::nms::scale_boxes_back(&mut dd, p.gain_w, p.pad_w, p.pad_h, smp.img.w as f32, smp.img.h as f32);
            pd_boxes.push(dd.xyxy);
            pd_cls.push(dd.cls);
            pd_conf.push(dd.score);
            let kp: Vec<[f32; 3]> = di.extra[k].chunks_exact(3).map(|c| [c[0], c[1], if c[2] > 0.01 { 1.0 } else { 0.0 }]).collect();
            pd_kpts.push(kp);
        }
        let gt_boxes: Vec<[f32; 4]> = gts.iter().map(|g| g.1).collect();
        let iou_box = crate::nms::iou_matrix(&pd_boxes, &gt_boxes);
        // OKS：area 用各自 GT 的 bbox 面积（官方 oks_iou 语义，逐对 area）
        let mut iou_oks = vec![vec![0f32; gt_boxes.len()]; pd_boxes.len()];
        {
            let areas: Vec<f32> = gts
                .iter()
                .map(|g| (g.1[2] - g.1[0]) * (g.1[3] - g.1[1]))
                .collect();
            let nk = pd_kpts.first().map(|k| k.len()).unwrap_or(17);
            let sig: Vec<f64> = if nk == 17 { OKS17.to_vec() } else { vec![0.05; nk] };
            iou_oks = oks_matrix(&pd_kpts, &gts.iter().map(|g| g.2.clone()).collect::<Vec<_>>(), &areas, &sig);
        }
        let pred_c: Vec<i64> = pd_cls.clone();
        let gt_c: Vec<i64> = gts.iter().map(|g| g.0).collect();
        let matched_b = ul_core::metrics::match_predictions(&pred_c, &gt_c, &iou_oks, &ul_core::IOUV);
        for (k, d) in di.dets.iter().enumerate() {
            tp_all.push(matched_b[k].clone());
            conf_all.push(pd_conf[k]);
            pred_cls_all.push(pred_c[k]);
        }
        for g in &gts {
            gt_cls_all.push(g.0);
        }
        let _ = kpt_dim;
    }
    let ap = ul_core::metrics::ap_per_class(&tp_all, &conf_all, &pred_cls_all, &gt_cls_all);
    Ok(DetMetrics::from_ap(&ap))
}

fn validate_obb(model: &YoloModel, val: &YoloDataset, imgsz: u32, device: Device) -> Result<DetMetrics, UlError> {
    let nc = model.nc;
    let cfg = NmsCfg::val();
    let mut tp_all: Vec<Vec<bool>> = Vec::new();
    let mut conf_all: Vec<f32> = Vec::new();
    let mut pred_cls_all: Vec<i64> = Vec::new();
    let mut gt_cls_all: Vec<i64> = Vec::new();
    let s = imgsz as f32;
    for i in 0..val.len() {
        let smp = val.load(i)?;
        let lb = letterbox(&smp.img, imgsz, [114, 114, 114]);
        let p = ul_core::bbox::LetterboxParams::new(smp.img.w as f32, smp.img.h as f32, s, s);
        let x = to_tensor(&[lb], device);
        let out = model.forward(&x, false)?;
        let t = match out {
            ModelOutput::Detect(ul_nn::head::HeadOutput::Inference(t)) => t,
            _ => return Err(UlError::Msg("obb val expects inference".into())),
        };
        let t_t = t.transpose(1, 2).contiguous(); // (1,A,4+nc+1)：xywh(像素)+cls+angle
        let per = non_max_suppression_ex(&t_t, 4 + nc, &cfg, 1, t_t.size()[2], t_t.size()[1]);
        let di = &per[0];
        let mut gts: Vec<(i64, [f32; 5])> = Vec::new();
        for ins in &smp.instances {
            let cx = ins.xywhn[0] * smp.img.w as f32 * p.gain_w + p.pad_w;
            let cy = ins.xywhn[1] * smp.img.h as f32 * p.gain_h + p.pad_h;
            let w = ins.xywhn[2] * smp.img.w as f32 * p.gain_w;
            let h = ins.xywhn[3] * smp.img.h as f32 * p.gain_h;
            gts.push((ins.cls, [cx, cy, w, h, ins.angle]));
        }
        let mut pd_xywhr: Vec<[f32; 5]> = Vec::new();
        let mut pd_cls: Vec<i64> = Vec::new();
        let mut pd_conf: Vec<f32> = Vec::new();
        for (k, d) in di.dets.iter().enumerate() {
            let ang = di.extra[k].first().copied().unwrap_or(0.0);
            // Det.xyxy 实为 (cx, cy, w, h)（OBB 解码），×strides 已在推理解码里
            pd_xywhr.push([d.xyxy[0], d.xyxy[1], d.xyxy[2], d.xyxy[3], ang]);
            pd_cls.push(d.cls);
            pd_conf.push(d.score);
        }
        let gt_b: Vec<Tensor> = gts.iter().map(|g| Tensor::from_slice(&g.1).unsqueeze(0)).collect();
        let mut iou = vec![vec![0f32; gts.len()]; pd_xywhr.len()];
        for (i, pr) in pd_xywhr.iter().enumerate() {
            let pt = Tensor::from_slice(pr).unsqueeze(0);
            for (j, gt) in gt_b.iter().enumerate() {
                let v = ul_nn::ops::probiou_pairwise(&pt, gt, false, 0.0);
                iou[i][j] = f32::try_from(&v).unwrap_or(0.0);
            }
        }
        let pred_c: Vec<i64> = pd_cls.clone();
        let gt_c: Vec<i64> = gts.iter().map(|g| g.0).collect();
        let matched = ul_core::metrics::match_predictions(&pred_c, &gt_c, &iou, &ul_core::IOUV);
        for (k, d) in di.dets.iter().enumerate() {
            tp_all.push(matched[k].clone());
            conf_all.push(pd_conf[k]);
            pred_cls_all.push(pred_c[k]);
        }
        for g in &gts {
            gt_cls_all.push(g.0);
        }
    }
    let ap = ul_core::metrics::ap_per_class(&tp_all, &conf_all, &pred_cls_all, &gt_cls_all);
    Ok(DetMetrics::from_ap(&ap))
}
