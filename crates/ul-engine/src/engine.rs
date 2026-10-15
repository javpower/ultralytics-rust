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

use crate::loss::{classification_loss, v8_detection_loss, DetBatch, LossW};
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
    pub close_mosaic: usize,
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
            close_mosaic: 10,
            val_interval: 1,
            project: PathBuf::from("runs"),
            name: "train".into(),
            exist_ok: false,
            patience: 100,
            pretrained: None,
        }
    }
}

/// EMA（ModelEMA 语义：衰减 ramp），按变量名保存影子权重。
pub struct ModelEma {
    pub ema: std::collections::HashMap<String, Tensor>,
    pub decay: f64,
}

impl ModelEma {
    pub fn new(model: &YoloModel) -> Self {
        let vars = model.vs.variables();
        let ema: std::collections::HashMap<String, Tensor> = vars
            .iter()
            .map(|(n, t)| (n.clone(), t.to_device(Device::Cpu).to_kind(Kind::Float)))
            .collect();
        ModelEma { ema, decay: 0.9999 }
    }

    pub fn update(&mut self, model: &YoloModel, i: usize) {
        self.decay = 0.9999f64 * (1.0 - (-(i as f64) / 2000.0).exp());
        let vars = model.vs.variables();
        tch::no_grad(|| {
            for (n, t) in &vars {
                if let Some(e) = self.ema.get_mut(n) {
                    let cur = t.to_device(Device::Cpu).to_kind(Kind::Float);
                    let new = e.shallow_clone() * self.decay + cur * (1.0 - self.decay);
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
    let mut train_ds = YoloDataset::from_img_dir(&train_dirs[0], names.clone(), _kpt_dim)?;
    for d in &train_dirs[1..] {
        let extra = YoloDataset::from_img_dir(d, names.clone(), _kpt_dim)?;
        train_ds.img_files.extend(extra.img_files);
        train_ds.label_files.extend(extra.label_files);
    }
    let val_ds = val_dirs
        .first()
        .map(|d| YoloDataset::from_img_dir(d, names.clone(), _kpt_dim))
        .transpose()?;

    let nc = model.nc;
    model.names = names.clone();
    let e2e = model.end2end;
    let reg_max = model.reg_max;
    let _ = reg_max;

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

    // 枚举两种优化器（OptimizerConfig 非 dyn 兼容）
    enum OptKind {
        Sgd(tch::nn::Sgd),
        AdamW(tch::nn::AdamW),
    }
    let opt_kind = if opt_name == "SGD" {
        OptKind::Sgd(tch::nn::Sgd { momentum, dampening: 0.0, wd: cfg.weight_decay, nesterov: false })
    } else {
        OptKind::AdamW(tch::nn::AdamW { beta1: 0.9, beta2: 0.999, wd: cfg.weight_decay, eps: 1e-8, amsgrad: false })
    };
    use tch::nn::OptimizerConfig;
    let mut opt = match opt_kind {
        OptKind::Sgd(c) => c.build(&model.vs, lr0),
        OptKind::AdamW(c) => c.build(&model.vs, lr0),
    }
    .map_err(|e| UlError::Msg(format!("optimizer: {e}")))?;

    let assigner = TaskAlignedAssigner { topk: 10, num_classes: nc, alpha: 0.5, beta: 6.0, eps: 1e-9, stride_val: 16.0 };
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

        let close_mosaic_on = cfg.epochs.saturating_sub(epoch) > cfg.close_mosaic;
        let mut acfg = AugmentCfg::default();
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
        let mut it = 0usize;
        for chunk in order.chunks(cfg.batch) {
            let ni = epoch * nb + it;
            // warmup lr/momentum
            if ni < nw {
                // 官方：lr 从 0 线性升到 lr0；momentum 从 0.8 升到目标
                let xn = (ni as f64) / nw as f64;
                let lr_now = lr0 * xn;
                let _ = opt.set_lr(lr_now);
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
                let _ = opt.set_lr(lr_now.max(1e-8));
            }

            // 构建批（增强 + letterbox）
            let mut imgs: Vec<Img> = Vec::with_capacity(chunk.len());
            let mut labels: Vec<Vec<f32>> = Vec::new();
            for (bi, &idx) in chunk.iter().enumerate() {
                let mut smp = augmentor.train_augment(idx, &load);
                if smp.img.w != cfg.imgsz || smp.img.h != cfg.imgsz {
                    smp.img = letterbox(&smp.img, cfg.imgsz, [114, 114, 114]);
                }
                for ins in &smp.instances {
                    labels.push(vec![
                        bi as f32,
                        ins.cls as f32,
                        ins.xywhn[0],
                        ins.xywhn[1],
                        ins.xywhn[2],
                        ins.xywhn[3],
                    ]);
                }
                imgs.push(smp.img);
            }
            let img_t = to_tensor(&imgs, cfg.device);
            let flat: Vec<f32> = labels.concat();
            let labels_t = if flat.is_empty() {
                Tensor::zeros([0, 6], (Kind::Float, cfg.device))
            } else {
                Tensor::from_slice(&flat).reshape([-1, 6]).to_device(cfg.device)
            };
            let batch = DetBatch { img: img_t, labels: labels_t, batch_size: chunk.len() as i64, imgsz: cfg.imgsz as i64 };

            // 前向 + 损失 + 反向
            match model.task {
                Task::Classify => {
                    let (logits, _) = match model.forward(&batch.img, true).unwrap() {
                        ModelOutput::Classify(a, b) => (a, b),
                        _ => unreachable!(),
                    };
                    let targets = batch.labels.select(1, 1).to_kind(Kind::Int64);
                    let loss = classification_loss(&logits, &targets, 0.1);
                    opt.backward_step_clip_norm(&loss, 10.0);
                    epoch_loss += f64::try_from(&loss).unwrap_or(0.0);
                }
                _ => {
                    let out = model.forward(&batch.img, true).unwrap();
                    let head_out = match out {
                        ModelOutput::Detect(h) => h,
                        _ => panic!("train only detect/classify for now"),
                    };
                    let loss = v8_detection_loss(model, lw, &assigner, &head_out, &batch, e2e);
                    opt.backward_step_clip_norm(&loss, 10.0);
                    epoch_loss += f64::try_from(&loss).unwrap_or(0.0);
                }
            }
            it += 1;
        }
        epochs_done = epoch + 1;
        ema.update(model, epoch * nb + it);

        // 验证
        let do_val = val_ds.is_some() && (epoch + 1) % cfg.val_interval.max(1) == 0;
        if do_val {
            let val = val_ds.as_ref().unwrap();
            ema.apply_to(model);
            let metrics = validate_model(model, val, cfg.imgsz, cfg.device, &names)?;
            log::info!(
                "epoch {}/{} loss={:.4} P={:.4} R={:.4} mAP50={:.4} mAP50-95={:.4}",
                epoch + 1,
                cfg.epochs,
                epoch_loss / it.max(1) as f64,
                metrics.mp,
                metrics.mr,
                metrics.map50,
                metrics.map50_95
            );
            let fit = metrics.fitness();
            use std::io::Write;
            writeln!(
                metrics_log,
                "{{\"epoch\":{},\"loss\":{:.4},\"P\":{:.4},\"R\":{:.4},\"mAP50\":{:.4},\"mAP50-95\":{:.4}}}",
                epoch + 1,
                epoch_loss / it.max(1) as f64,
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
            log::info!("epoch {}/{} loss={:.4}", epoch + 1, cfg.epochs, epoch_loss / it.max(1) as f64);
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
        let lb = letterbox(&img, imgsz, [114, 114, 114]);
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
