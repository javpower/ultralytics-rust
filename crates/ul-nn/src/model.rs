//! YoloModel：完整模型（DAG 前向 + stride 推断 + 权重管理入口）。

use std::sync::Arc;
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

use ul_core::cfg::load_model_yaml;
use ul_core::UlError;

use crate::head::{HeadOutput, RawPreds};
use crate::parse::{parse_model, Layer, Module, Parsed, Task};

/// 模型前向输出。
pub enum ModelOutput {
    /// 检测类头：训练原始预测 / 推理解码张量。
    Detect(HeadOutput),
    /// 分类：(probs, logits)（eval）或 (logits, logits)（train）。
    Classify(Tensor, Tensor),
    /// 语义分割：(logits, aux?)。
    SemSeg(Tensor, Option<Tensor>),
}

pub struct YoloModel {
    pub vs: Arc<VarStore>,
    pub layers: Vec<Layer>,
    pub save: Vec<usize>,
    pub task: Task,
    pub nc: i64,
    pub names: Vec<String>,
    pub stride: Vec<f64>,
    pub end2end: bool,
    pub reg_max: i64,
    pub kpt_shape: Option<(i64, i64)>,
    pub yaml_file: String,
    pub scale: String,
    pub imgsz_hint: i64,
}

impl YoloModel {
    /// 构建：`model("yolo11n", 80, device)` / `model("path/to/yolo11n.yaml", ...)`
    pub fn build(name: &str, nc: Option<i64>, device: Device, verbose: bool) -> Result<Self, UlError> {
        let yaml = load_model_yaml(name)?;
        Self::build_from_yaml(yaml, nc, device, verbose)
    }

    pub fn build_from_yaml(
        yaml: ul_core::cfg::ModelYaml,
        nc_override: Option<i64>,
        device: Device,
        verbose: bool,
    ) -> Result<Self, UlError> {
        let nc = nc_override.unwrap_or(yaml.nc);
        let vs = Arc::new(VarStore::new(device));
        let Parsed { layers, save, task } = parse_model(&vs, &yaml, 3, Some(nc), false)?;

        let mut model = YoloModel {
            vs,
            layers,
            save,
            task,
            nc,
            names: (0..nc).map(|i| format!("class_{i}")).collect(),
            stride: vec![],
            end2end: yaml.end2end,
            reg_max: yaml.reg_max,
            kpt_shape: yaml.kpt_shape,
            yaml_file: yaml.yaml_file.clone(),
            scale: yaml.scale.clone(),
            imgsz_hint: 640,
        };

        // stride 推断：256 尺寸 dummy 前向（BN eval，head 返回 feats）
        let s = 256;
        let x = Tensor::zeros([1, 3, s, s], (Kind::Float, device));
        let strides = model.compute_strides(&x)?;
        model.stride = strides.clone();
        // 写回头部并执行 bias_init
        if let Some(last) = model.layers.last_mut() {
            if let Module::Head(crate::parse::HeadMod::Detect(head)) = &mut last.module {
                head.stride = strides;
                head.bias_init();
                model.end2end = head.end2end;
            }
        }
        if verbose {
            log::info!("{} summary: {} layers, {} parameters", yaml.yaml_file, model.layers.len(), model.vs.variables().len() as i64);
        }
        Ok(model)
    }

    /// dummy 前向取各 head 输入层的空间尺寸 → stride。
    fn compute_strides(&mut self, x: &Tensor) -> Result<Vec<f64>, UlError> {
        let head_idx = self.layers.len() - 1;
        // 先按普通路径前向到倒数第二层，然后取头部的 feats
        let mut y: Vec<Option<Tensor>> = (0..self.layers.len()).map(|_| None).collect();
        let mut x = x.shallow_clone();
        for (li, layer) in self.layers.iter().enumerate() {
            let xs = self.gather_inputs(layer, &y, &x, li)?;
            if li == head_idx {
                // 头层：提取 feats 尺寸
                if let Module::Head(crate::parse::HeadMod::Detect(head)) = &layer.module {
                    let feats = &xs;
                    let strides = feats.iter().map(|f| 256.0 / f.size()[2] as f64).collect();
                    return Ok(strides);
                } else if let Module::Head(_) = &layer.module {
                    return Ok(vec![32.0]);
                }
            }
            if std::env::var("UL_DEBUG_SHAPES").is_ok() {
                eprintln!("probe L{li:>2} {} from={:?} in={:?}", layer.type_name, layer.from, xs[0].size());
            }
            let out = layer.module.forward(&xs, false);
            if std::env::var("UL_DEBUG_SHAPES").is_ok() {
                match &out {
                    Ok(t) => eprintln!("probe L{li:>2} out={:?}", t.size()),
                    Err(e) => eprintln!("probe L{li:>2} ERR {e}"),
                }
            }
            let out = out?;
            if self.save.contains(&li) {
                y[li] = Some(out.shallow_clone());
            }
            x = out;
        }
        Ok(vec![32.0])
    }

    fn gather_inputs(
        &self,
        layer: &Layer,
        y: &[Option<Tensor>],
        x: &Tensor,
        li: usize,
    ) -> Result<Vec<Tensor>, UlError> {
        let mut out = Vec::with_capacity(layer.from.len());
        for f in &layer.from {
            if *f == -1 {
                out.push(x.shallow_clone());
            } else {
                let idx = if *f < 0 { (li as i64 + f) as usize } else { *f as usize };
                let t = y
                    .get(idx)
                    .and_then(|o| o.as_ref())
                    .ok_or_else(|| UlError::Msg(format!("layer {li} input {f} not saved")))?;
                out.push(t.shallow_clone());
            }
        }
        Ok(out)
    }

    /// 前向主入口。
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<ModelOutput, UlError> {
        let mut y: Vec<Option<Tensor>> = (0..self.layers.len()).map(|_| None).collect();
        let mut cur = x.shallow_clone();
        for (li, layer) in self.layers.iter().enumerate() {
            let xs = self.gather_inputs(layer, &y, &cur, li)?;
            if std::env::var("UL_DEBUG_SHAPES").is_ok() {
                let in_ch = xs[0].size();
                eprintln!("L{li:>2} {} from={:?} in={:?}", layer.type_name, layer.from, in_ch);
            }
            let out = match &layer.module {
                Module::Head(head_mod) => {
                    let o = dispatch_head(head_mod, &xs, train)?;
                    if li == self.layers.len() - 1 {
                        return Ok(o);
                    }
                    // 中间出现头（不常见）：转张量继续
                    match o {
                        ModelOutput::Detect(HeadOutput::Inference(t)) => t,
                        ModelOutput::Classify(a, _) => a,
                        ModelOutput::SemSeg(a, _) => a,
                        _ => return Err(UlError::Msg("mid-graph head in training mode".into())),
                    }
                }
                m => m.forward(&xs, train)?,
            };
            if self.save.contains(&li) {
                y[li] = Some(out.shallow_clone());
            }
            cur = out;
        }
        Err(UlError::Msg("no head layer reached".into()))
    }

    /// 参数量。
    pub fn num_params(&self) -> i64 {
        self.vs
            .variables()
            .values()
            .map(|t| t.numel())
            .sum::<usize>() as i64
    }

    pub fn is_classify(&self) -> bool {
        self.task == Task::Classify
    }
}

/// 头部前向分发（含 proto 注入）。
fn dispatch_head(head: &crate::parse::HeadMod, xs: &[Tensor], train: bool) -> Result<ModelOutput, UlError> {
    match head {
        crate::parse::HeadMod::Detect(dh) => {
            // Segment 家族：proto 需在头前向后附加
            let out = dh.forward(xs, train);
            Ok(match out {
                HeadOutput::Train(mut rp) => {
                    attach_proto(dh, &mut rp, xs, train)?;
                    ModelOutput::Detect(HeadOutput::Train(rp))
                }
                HeadOutput::TrainDual(mut o2m, mut o2o) => {
                    attach_proto(dh, &mut o2m, xs, train)?;
                    attach_proto(dh, &mut o2o, xs, train)?;
                    ModelOutput::Detect(HeadOutput::TrainDual(o2m, o2o))
                }
                HeadOutput::Inference(t) => ModelOutput::Detect(HeadOutput::Inference(t)),
                _ => return Err(UlError::Msg("unexpected head output".into())),
            })
        }
        crate::parse::HeadMod::Classify(ch) => {
            let (a, b) = ch.forward(&xs[0], train);
            Ok(ModelOutput::Classify(a, b))
        }
        crate::parse::HeadMod::SemSeg(sh) => {
            let (a, aux) = sh.forward(xs, train);
            Ok(ModelOutput::SemSeg(a, aux))
        }
    }
}

fn attach_proto(dh: &crate::head::DetectHead, rp: &mut RawPreds, xs: &[Tensor], train: bool) -> Result<(), UlError> {
    match &dh.proto {
        Some(crate::head::ProtoVariant::P1(proto)) => {
            rp.proto = Some(proto.forward(&xs[0], train));
        }
        Some(crate::head::ProtoVariant::P26(proto)) => {
            let (p, sem) = proto.forward(xs, train);
            rp.proto = Some(p);
            rp.proto_sem = sem;
        }
        None => {}
    }
    Ok(())
}
