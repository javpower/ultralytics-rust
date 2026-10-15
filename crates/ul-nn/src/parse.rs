//! parse_model：官方模型 YAML → 层 DAG（与 ultralytics/nn/tasks.py::parse_model 语义一致）。

use std::sync::Arc;
use tch::nn::VarStore;
use tch::Tensor;

use ul_core::cfg::{ModelYaml, YamlVal};

use crate::attention::{A2C2f, C2fPsa, C2Psa, Psa};
use crate::block::*;
use crate::conv::{Conv, Conv2, ConvTranspose, Focus, GhostConv, LightConv};
use crate::head::{ClassifyHead, DetectHead, ExtraKind, HeadOutput, SemSegHead};
use crate::module::VarPath;
use crate::head::ProtoVariant;
use ul_core::UlError;

/// 数学：make_divisible（ceil 到 divisor 的倍数）。
pub fn make_divisible(x: f64, divisor: i64) -> i64 {
    ((x / divisor as f64).ceil() * divisor as f64) as i64
}

/// 图中的一个层。
pub struct Layer {
    pub i: usize,
    pub from: Vec<i64>,
    pub module: Module,
    pub type_name: String,
    pub params: i64,
}

/// 头部模块。
pub enum HeadMod {
    Detect(DetectHead),
    Classify(ClassifyHead),
    SemSeg(SemSegHead),
}

/// 全部模块的统一枚举。
pub enum Module {
    Conv(Conv),
    DwConv(Conv),
    Conv2(Conv2),
    ConvTranspose(ConvTranspose),
    Focus(Focus),
    GhostConv(GhostConv),
    LightConv(LightConv),
    Spp(Spp),
    Sppf(Sppf),
    C1(C1),
    C2(C2),
    C2f(C2f),
    C3(C3),
    C3x(C3x),
    C3f(C3f),
    C3Ghost(C3Ghost),
    BottleneckCsp(BottleneckCsp),
    RepC3(RepC3),
    RepNcspeLan4(RepNcspeLan4),
    AConv(AConv),
    ADown(ADown),
    SppeLan(SppeLan),
    CbLinear(CbLinear),
    CbFuse(CbFuse),
    C3k2(C3k2),
    C2fCib(C2fCib),
    C2fPsa(C2fPsa),
    C2Psa(C2Psa),
    Psa(Psa),
    ScDown(ScDown),
    A2C2f(A2C2f),
    Bottleneck(Bottleneck),
    GhostBottleneck(GhostBottleneck),
    ResNetLayer(ResNetLayer),
    HgStem(HgStem),
    HgBlock(HgBlock),
    Head(HeadMod),
    Concat(i64),
    Upsample(f64),
    Identity,
    Index(i64),
    MaxPool2d(i64, i64, i64),
    ZeroPad2d(Vec<i64>),
}

impl Module {
    pub fn forward(&self, xs: &[Tensor], train: bool) -> Result<Tensor, UlError> {
        let x = &xs[0];
        Ok(match self {
            Module::Conv(m) => m.forward(x, train),
            Module::DwConv(m) => m.forward(x, train),
            Module::Conv2(m) => m.forward(x, train),
            Module::ConvTranspose(m) => m.forward(x, train),
            Module::Focus(m) => m.forward(x, train),
            Module::GhostConv(m) => m.forward(x, train),
            Module::LightConv(m) => m.forward(x, train),
            Module::Spp(m) => m.forward(x, train),
            Module::Sppf(m) => m.forward(x, train),
            Module::C1(m) => m.forward(x, train),
            Module::C2(m) => m.forward(x, train),
            Module::C2f(m) => m.forward(x, train),
            Module::C3(m) => m.forward(x, train),
            Module::C3x(m) => m.forward(x, train),
            Module::C3f(m) => m.forward(x, train),
            Module::C3Ghost(m) => m.forward(x, train),
            Module::BottleneckCsp(m) => m.forward(x, train),
            Module::RepC3(m) => m.forward(x, train),
            Module::RepNcspeLan4(m) => m.forward(x, train),
            Module::AConv(m) => m.forward(x, train),
            Module::ADown(m) => m.forward(x, train),
            Module::SppeLan(m) => m.forward(x, train),
            Module::CbLinear(m) => m.forward(x),
            Module::CbFuse(m) => m.forward(xs),
            Module::C3k2(m) => m.forward(x, train),
            Module::C2fCib(m) => m.forward(x, train),
            Module::C2fPsa(m) => m.forward(x, train),
            Module::C2Psa(m) => m.forward(x, train),
            Module::Psa(m) => m.forward(x, train),
            Module::ScDown(m) => m.forward(x, train),
            Module::A2C2f(m) => m.forward(x, train),
            Module::Bottleneck(m) => m.forward(x, train),
            Module::GhostBottleneck(m) => m.forward(x, train),
            Module::ResNetLayer(m) => m.forward(x, train),
            Module::HgStem(m) => m.forward(x, train),
            Module::HgBlock(m) => m.forward(x, train),
            Module::Concat(d) => Tensor::cat(xs, *d),
            Module::Upsample(s) => {
                let (h, w) = (x.size()[2], x.size()[3]);
                crate::ops::up_nearest(x, [(h as f64 * s) as i64, (w as f64 * s) as i64])
            }
            Module::Identity => x.shallow_clone(),
            Module::Index(i) => xs[*i as usize].shallow_clone(),
            Module::MaxPool2d(k, s, p) => {
                let (k, s, p) = (*k, *s, *p);
                let pk = if p < 0 { k / 2 } else { p };
                x.f_max_pool2d([k, k], [s, s], [pk, pk], [1, 1], false)?
            }
            Module::ZeroPad2d(pads) => {
                let pads = if pads.len() == 4 {
                    [pads[0], pads[1], pads[2], pads[3]]
                } else {
                    [pads[0], pads[0], pads[1], pads[1]]
                };
                x.f_constant_pad_nd(pads)?
            }
            Module::Head(_) => return Err(UlError::Msg("head module must be dispatched by model forward".into())),
        })
    }
}

/// parse_model 的输出。
pub struct Parsed {
    pub layers: Vec<Layer>,
    pub save: Vec<usize>,
    pub task: Task,
}

/// 任务枚举。
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Task {
    Detect,
    Segment,
    Pose,
    Obb,
    Classify,
    SemSeg,
}

/// parse_model 主流程。
pub fn parse_model(
    vs: &Arc<VarStore>,
    d: &ModelYaml,
    ch_in: i64,
    nc_override: Option<i64>,
    verbose: bool,
) -> Result<Parsed, UlError> {
    let nc = nc_override.unwrap_or(d.nc);
    let reg_max = d.reg_max;
    let end2end = d.end2end;
    let depth = d.depth_multiple;
    let width = d.width_multiple;
    let max_channels = d.max_channels;
    let scale = d.scale.clone();
    let kpt_shape = d.kpt_shape.unwrap_or((17, 3));
    // legacy：v3/v5/v8/v9 用旧式全宽 cls 头；v10/11/12/26 用 DWConv 对 cls 头
    let legacy = !(d.yaml_file.contains("yolo11")
        || d.yaml_file.contains("yolo26")
        || d.yaml_file.contains("yolo12")
        || d.yaml_file.contains("yolov10"));

    let mut ch = vec![ch_in];
    let mut layers: Vec<Layer> = Vec::new();
    let mut save: Vec<usize> = Vec::new();
    let root = VarPath::root(vs);

    let rows: Vec<&ul_core::cfg::LayerRow> = d.backbone.iter().chain(d.head.iter()).collect();

    for (i, row) in rows.iter().enumerate() {
        let f: Vec<i64> = row.from.clone();
        let m = row.module.as_str();
        let mut n = row.repeats;
        let args = &row.args;
        let n_ = if n > 1 { (n as f64 * depth).round().max(1.0) as i64 } else { n };
        let ch_f = |fi: i64| -> Result<i64, UlError> {
            let idx = if fi < 0 { (i as i64 + fi) } else { fi };
            if idx < 0 {
                return Ok(ch_in); // 层 0 的 -1 即网络输入
            }
            ch.get(idx as usize).copied().ok_or_else(|| UlError::Msg(format!("from index {fi} out of range at layer {i}")))
        };

        let (module, mut c2, type_name): (Module, i64, String) = match m {
            "Conv" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 1);
                let s = arg_i(args, 2, 1);
                let pe = args.get(3).and_then(|a| a.as_i64()); // 显式 pad（可缺省 → autopad）
                (
                    Module::Conv(Conv::init(&root.child(&fmt(i)), c1, c2m, k, s, pe, 1, 1, true)),
                    c2m,
                    "models.yolo.conv.Conv".into(),
                )
            }
            "DWConv" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 1);
                let s = arg_i(args, 2, 1);
                let pe = args.get(3).and_then(|a| a.as_i64());
                let g = gcd(c1, c2m);
                (
                    Module::DwConv(Conv::init(&root.child(&fmt(i)), c1, c2m, k, s, pe, g, 1, true)),
                    c2m,
                    "models.yolo.conv.DWConv".into(),
                )
            }
            "GhostConv" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 1);
                let s = arg_i(args, 2, 1);
                (
                    Module::GhostConv(GhostConv::init(&root.child(&fmt(i)), c1, c2m, k, s, 1, true)),
                    c2m,
                    "models.yolo.conv.GhostConv".into(),
                )
            }
            "LightConv" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 1);
                (Module::LightConv(LightConv::init(&root.child(&fmt(i)), c1, c2m, k)), c2m, "models.yolo.conv.LightConv".into())
            }
            "ConvTranspose" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 2);
                let s = arg_i(args, 2, 2);
                (
                    Module::ConvTranspose(ConvTranspose::init(&root.child(&fmt(i)), c1, c2m, k, s, 0, true, true)),
                    c2m,
                    "models.yolo.conv.ConvTranspose".into(),
                )
            }
            "nn.ConvTranspose2d" => {
                // 裸转置卷积（bias=True，无 BN）
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 2);
                let s = arg_i(args, 2, 2);
                let p = arg_i(args, 3, 0);
                (
                    Module::ConvTranspose(ConvTranspose::init(&root.child(&fmt(i)), c1, c2m, k, s, p, false, false)),
                    c2m,
                    "torch.nn.ConvTranspose2d".into(),
                )
            }
            "Focus" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 1);
                let s = arg_i(args, 2, 1);
                (Module::Focus(Focus::init(&root.child(&fmt(i)), c1, c2m, k, s, 1, true)), c2m, "models.yolo.conv.Focus".into())
            }
            "SPP" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k: Vec<i64> = args.get(1).and_then(|a| a.as_list().map(|l| l.to_vec())).unwrap_or(vec![5, 9, 13]);
                (Module::Spp(Spp::init(&root.child(&fmt(i)), c1, c2m, k)), c2m, "models.yolo.block.SPP".into())
            }
            "SPPF" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 5);
                let sppf_n = arg_i(args, 2, 3);
                let shortcut = arg_b(args, 3, false);
                let mut sppf = Sppf::init(&root.child(&fmt(i)), c1, c2m, k, sppf_n, shortcut);
                // legacy YAML rows（args<=3）需给 cv1 补 SiLU
                if args.len() <= 2 {
                    sppf.legacy = true;
                }
                (Module::Sppf(sppf), c2m, "models.yolo.block.SPPF".into())
            }
            "C1" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                (Module::C1(C1::init(&root.child(&fmt(i)), c1, c2m, n_)), c2m, "models.yolo.block.C1".into())
            }
            "C2" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let shortcut = arg_b(args, 1, true);
                let g = arg_i(args, 2, 1);
                let e = arg_f(args, 3, 0.5);
                (Module::C2(C2::init(&root.child(&fmt(i)), c1, c2m, n_, shortcut, g, e)), c2m, "models.yolo.block.C2".into())
            }
            "C2f" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let shortcut = arg_b(args, 1, false);
                let g = arg_i(args, 2, 1);
                let e = arg_f(args, 3, 0.5);
                (Module::C2f(C2f::init(&root.child(&fmt(i)), c1, c2m, n_, shortcut, g, e)), c2m, "models.yolo.block.C2f".into())
            }
            "C3" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let shortcut = arg_b(args, 1, true);
                let g = arg_i(args, 2, 1);
                let e = arg_f(args, 3, 0.5);
                (Module::C3(C3::init(&root.child(&fmt(i)), c1, c2m, n_, shortcut, g, e, (1, 3))), c2m, "models.yolo.block.C3".into())
            }
            "C3x" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let shortcut = arg_b(args, 1, true);
                let g = arg_i(args, 2, 1);
                let e = arg_f(args, 3, 0.5);
                (Module::C3x(C3x::init(&root.child(&fmt(i)), c1, c2m, n_, shortcut, g, e)), c2m, "models.yolo.block.C3x".into())
            }
            "C3f" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let shortcut = arg_b(args, 1, false);
                let g = arg_i(args, 2, 1);
                let e = arg_f(args, 3, 0.5);
                (Module::C3f(C3f::init(&root.child(&fmt(i)), c1, c2m, n_, shortcut, g, e)), c2m, "models.yolo.block.C3f".into())
            }
            "C3Ghost" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let e = arg_f(args, 3, 0.5);
                (
                    Module::C3Ghost(C3Ghost::init(&root.child(&fmt(i)), c1, c2m, n_, e)),
                    c2m,
                    "models.yolo.block.C3Ghost".into(),
                )
            }
            "C3k" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let shortcut = arg_b(args, 1, true);
                let e = arg_f(args, 3, 0.5);
                let k = arg_i(args, 4, 3);
                (
                    Module::C3(C3::init(&root.child(&fmt(i)), c1, c2m, n_, shortcut, 1, e, (k, k))),
                    c2m,
                    "models.yolo.block.C3k".into(),
                )
            }
            "C3k2" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                // args: [c2, c3k?, e?, attn?] —— C3k2 构造序 (c1, c2, n, c3k, e, attn)
                let has_c3k = args.len() > 1;
                let c3k = if has_c3k {
                    args[1].as_bool().unwrap_or(false) || (scale == "m" || scale == "l" || scale == "x")
                } else {
                    scale == "m" || scale == "l" || scale == "x"
                };
                let e = arg_f(args, 2, 0.5);
                let attn = arg_b(args, 3, false);
                (
                    Module::C3k2(C3k2::init(&root.child(&fmt(i)), c1, c2m, n_, c3k, e, attn, 1, true)),
                    c2m,
                    "models.yolo.block.C3k2".into(),
                )
            }
            "C2fCIB" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let shortcut = arg_b(args, 1, false);
                let lk = arg_b(args, 2, false);
                let g = arg_i(args, 3, 1);
                let e = arg_f(args, 4, 0.5);
                (
                    Module::C2fCib(C2fCib::init(&root.child(&fmt(i)), c1, c2m, n_, shortcut, lk, g, e)),
                    c2m,
                    "models.yolo.block.C2fCIB".into(),
                )
            }
            "C2fPSA" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let e = arg_f(args, 1, 0.5);
                (Module::C2fPsa(C2fPsa::init(&root.child(&fmt(i)), c1, c2m, n_, e)), c2m, "models.yolo.block.C2fPSA".into())
            }
            "C2PSA" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let e = arg_f(args, 1, 0.5);
                (Module::C2Psa(C2Psa::init(&root.child(&fmt(i)), c1, c2m, n_, e)), c2m, "models.yolo.block.C2PSA".into())
            }
            "PSA" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let e = arg_f(args, 2, 0.5);
                (Module::Psa(Psa::init(&root.child(&fmt(i)), c1, c2m, e)), c2m, "models.yolo.block.PSA".into())
            }
            "SCDown" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 1);
                let s = arg_i(args, 2, 1);
                (Module::ScDown(ScDown::init(&root.child(&fmt(i)), c1, c2m, k, s)), c2m, "models.yolo.block.SCDown".into())
            }
            "A2C2f" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                // 官方 A2C2f args：(c2, a2, area, residual?, mlp_ratio?)
                let a2 = arg_b(args, 1, true);
                let area = arg_i(args, 2, 1);
                let (residual, mlp_ratio) = if scale == "l" || scale == "x" {
                    (true, arg_f(args, 3, 1.2))
                } else {
                    (false, 2.0)
                };
                (
                    Module::A2C2f(A2C2f::init(&root.child(&fmt(i)), c1, c2m, n_, a2, area, residual, mlp_ratio, 0.5, 1, true)),
                    c2m,
                    "models.yolo.block.A2C2f".into(),
                )
            }
            "RepNCSPELAN4" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let c3 = scaled(&args[1], width, max_channels);
                let c4 = scaled(&args[2], width, max_channels);
                (
                    Module::RepNcspeLan4(RepNcspeLan4::init(&root.child(&fmt(i)), c1, c2m, c3, c4, n_, false)),
                    c2m,
                    "models.yolo.block.RepNCSPELAN4".into(),
                )
            }
            "ELAN1" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let c3 = scaled(&args[1], width, max_channels);
                let c4 = scaled(&args[2], width, max_channels);
                (
                    Module::RepNcspeLan4(RepNcspeLan4::init(&root.child(&fmt(i)), c1, c2m, c3, c4, n_, true)),
                    c2m,
                    "models.yolo.block.ELAN1".into(),
                )
            }
            "AConv" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                (Module::AConv(AConv::init(&root.child(&fmt(i)), c1, c2m)), c2m, "models.yolo.block.AConv".into())
            }
            "ADown" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                (Module::ADown(ADown::init(&root.child(&fmt(i)), c1, c2m)), c2m, "models.yolo.block.ADown".into())
            }
            "SPPELAN" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let c3 = scaled(&args[1], width, max_channels);
                let k = arg_i(args, 2, 5);
                (Module::SppeLan(SppeLan::init(&root.child(&fmt(i)), c1, c2m, c3, k)), c2m, "models.yolo.block.SPPELAN".into())
            }
            "RepC3" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let e = arg_f(args, 2, 1.0);
                (Module::RepC3(RepC3::init(&root.child(&fmt(i)), c1, c2m, n_, e)), c2m, "models.yolo.dyamahead.RepC3".into())
            }
            "CBLinear" => {
                let c1 = ch_f(f[0])?;
                let c2s: Vec<i64> = args[0]
                    .as_list()
                    .ok_or_else(|| UlError::Msg("CBLinear requires c2s list".into()))?
                    .iter()
                    .map(|&c| scaled(&YamlVal::Int(c), width, max_channels))
                    .collect();
                let k = arg_i(args, 1, 1);
                let s = arg_i(args, 2, 1);
                let c2: i64 = c2s.iter().sum();
                (
                    Module::CbLinear(CbLinear::init(&root.child(&fmt(i)), c1, c2s, k, s, 1)),
                    c2,
                    "models.yolo.block.CBLinear".into(),
                )
            }
            "CBFuse" => {
                let idx: Vec<i64> = args[0].as_list().map(|l| l.to_vec()).unwrap_or_default();
                // 收集各 CBLinear 输入的切分表
                let mut split_c2s = Vec::new();
                for &fi in &f {
                    if fi >= 0 {
                        if let Module::CbLinear(cb) = &layers[fi as usize].module {
                            split_c2s.push(cb.c2s.clone());
                        }
                    }
                }
                let c2 = ch_f(*f.last().unwrap())?;
                let mut fuse = CbFuse::new(idx);
                fuse.split_c2s = split_c2s;
                (Module::CbFuse(fuse), c2, "models.yolo.block.CBFuse".into())
            }
            "Bottleneck" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let shortcut = arg_b(args, 1, true);
                let g = arg_i(args, 2, 1);
                let e = arg_f(args, 3, 0.5);
                (
                    Module::Bottleneck(Bottleneck::init(&root.child(&fmt(i)), c1, c2m, shortcut, g, (3, 3), e)),
                    c2m,
                    "models.yolo.block.Bottleneck".into(),
                )
            }
            "GhostBottleneck" => {
                let (c1, c2m) = (ch_f(f[0])?, scaled(&args[0], width, max_channels));
                let k = arg_i(args, 1, 3);
                let s = arg_i(args, 2, 1);
                (
                    Module::GhostBottleneck(GhostBottleneck::init(&root.child(&fmt(i)), c1, c2m, k, s)),
                    c2m,
                    "models.yolo.block.GhostBottleneck".into(),
                )
            }
            "ResNetLayer" => {
                // YAML args = [c1, c2, s, is_first, n]：c1 由图提供，n 取 repeats 列
                let c1 = ch_f(f[0])?;
                let c2m = if arg_b(args, 3, false) { args[1].as_i64().unwrap() } else { args[1].as_i64().unwrap() * 4 };
                let s = arg_i(args, 2, 1);
                let is_first = arg_b(args, 3, false);
                (
                    Module::ResNetLayer(ResNetLayer::init(&root.child(&fmt(i)), c1, c2m, s, is_first, n_, 4)),
                    c2m,
                    "models.yolo.block.ResNetLayer".into(),
                )
            }
            "HGStem" => {
                let c1 = ch_f(f[0])?;
                let cm = scaled(&args[0], width, max_channels);
                let c2m = scaled(&args[1], width, max_channels);
                (Module::HgStem(HgStem::init(&root.child(&fmt(i)), c1, cm, c2m)), c2m, "models.yolo.block.HGStem".into())
            }
            "HGBlock" => {
                let c1 = ch_f(f[0])?;
                let cm = scaled(&args[0], width, max_channels);
                let c2m = scaled(&args[1], width, max_channels);
                let k = arg_i(args, 2, 3);
                let lightconv = arg_b(args, 3, false);
                let shortcut = arg_b(args, 4, false);
                (
                    Module::HgBlock(HgBlock::init(&root.child(&fmt(i)), c1, cm, c2m, k, n_, lightconv, shortcut)),
                    c2m,
                    "models.yolo.block.HGBlock".into(),
                )
            }
            "Concat" => {
                let d = arg_i(args, 0, 1);
                let c2m: i64 = f.iter().map(|fi| ch_f(*fi)).collect::<Result<Vec<_>, _>>()?.iter().sum();
                (Module::Concat(d), c2m, "models.yolo.conv.Concat".into())
            }
            "nn.Upsample" => {
                let s = arg_f(args, 1, 2.0);
                (Module::Upsample(s), ch_f(f[0])?, "torch.nn.Upsample".into())
            }
            "nn.Identity" => (Module::Identity, ch_f(f[0])?, "torch.nn.Identity".into()),
            "nn.ZeroPad2d" => {
                // args[0] 是 [l, r, t, b]（NestedList）
                let pads = match args.first() {
                    Some(YamlVal::NestedList(l)) if !l.is_empty() => l[0].clone(),
                    Some(YamlVal::List(l)) if l.len() == 4 => l.clone(),
                    _ => vec![1, 1, 1, 1],
                };
                (
                    Module::ZeroPad2d(pads),
                    ch_f(f[0])?,
                    "torch.nn.ZeroPad2d".into(),
                )
            }
            "nn.MaxPool2d" => {
                let k = arg_i(args, 0, 2);
                let s = arg_i(args, 1, k);
                let pad = arg_i(args, 2, 0);
                (
                    Module::MaxPool2d(k, s, pad),
                    ch_f(f[0])?,
                    "torch.nn.MaxPool2d".into(),
                )
            }
            "Index" => {
                let idx = arg_i(args, 0, 0);
                let c2m = ch_f(f[0])?;
                (Module::Index(idx), c2m, "models.yolo.conv.Index".into())
            }
            "Detect" | "Segment" | "Segment26" | "Pose" | "Pose26" | "OBB" | "OBB26" | "v10Detect" => {
                let chs: Vec<i64> = f.iter().map(|fi| ch_f(*fi)).collect::<Result<Vec<_>, _>>()?;
                // v10Detect 模块名即 e2end 语义（官方 parse: end2end = d.end2end or m == "v10Detect"）
                let end2end = end2end || m == "v10Detect";
                let (kind, nm, npr, ne) = match m {
                    "Detect" | "v10Detect" => (ExtraKind::None, 0, 0, 0),
                    "Segment" | "Segment26" => {
                        let nm_default = YamlVal::Int(32);
                let nm_arg = args.get(1).unwrap_or(&nm_default);
                let nm_ = scaled(nm_arg, width, max_channels);
                        (ExtraKind::Mask, nm_, arg_i(args, 2, 256), 0)
                    }
                    "Pose" | "Pose26" => (if m == "Pose26" { ExtraKind::Kpt26 } else { ExtraKind::Kpt }, 0, 0, 0),
                    "OBB" | "OBB26" => (if m == "OBB26" { ExtraKind::Angle26 } else { ExtraKind::Angle }, 0, 0, arg_i(args, 1, 1)),
                    _ => unreachable!(),
                };
                let kpt = if kind == ExtraKind::Kpt || kind == ExtraKind::Kpt26 { kpt_shape } else { (17, 3) };
                let head_nc = nc;
                let mut head = DetectHead::new(
                    &root.child(&fmt(i)),
                    head_nc,
                    &chs,
                    reg_max,
                    end2end,
                    legacy,
                    kind,
                    nm,
                    npr,
                    ne,
                    kpt,
                    m == "v10Detect",
                );
                if end2end {
                    head.set_end2end(&root.child(&fmt(i)), &chs);
                }
                // proto
                let proto = match m {
                    "Segment" => {
                        let c_ = npr;
                        Some(ProtoVariant::P1(crate::special::Proto::init(&root.child(&fmt(i)).child("proto"), chs[0], c_, nm)))
                    }
                    "Segment26" => Some(ProtoVariant::P26(Box::new(crate::special::Proto26::init(
                        &root.child(&fmt(i)).child("proto"),
                        &chs,
                        npr,
                        nm,
                        nc,
                    )))),
                    _ => None,
                };
                let mut head = head;
                head.proto = proto;
                (Module::Head(HeadMod::Detect(head)), nc, format!("models.yolo.detect.{}", m))
            }
            "Classify" => {
                let c1 = ch_f(f[0])?;
                (
                    Module::Head(HeadMod::Classify(ClassifyHead::init(&root.child(&fmt(i)), c1, nc))),
                    nc,
                    "models.yolo.classify.Classify".into(),
                )
            }
            "SemanticSegment" => {
                let chs: Vec<i64> = f.iter().map(|fi| ch_f(*fi)).collect::<Result<Vec<_>, _>>()?;
                (
                    Module::Head(HeadMod::SemSeg(SemSegHead::init(&root.child(&fmt(i)), nc, &chs))),
                    chs[0],
                    "models.yolo.segment.SemanticSegment".into(),
                )
            }
            other => {
                return Err(UlError::Msg(format!(
                    "module '{other}' is not yet supported by ultralytics-rust (rt-detr/yolo-world/yoloe/depth on roadmap)"
                )))
            }
        };

        // 记录 save（层 0 无前序可引）
        if i > 0 {
            for x in &f {
                let xi = x.rem_euclid(i as i64) as usize;
                if *x != -1 {
                    save.push(xi);
                }
            }
        }
        if verbose {
            log::info!("{i:>3} {f:?} {n_:>3}  {m:<20} -> out_ch={c2}");
        }
        layers.push(Layer { i, from: f.clone(), module, type_name, params: 0 });
        if i == 0 {
            ch.clear();
        }
        ch.push(c2);
    }

    save.sort_unstable();
    save.dedup();

    let task = guess_task(&layers);
    Ok(Parsed { layers, save, task })
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn fmt(i: usize) -> String {
    format!("model/{}", i)
}

fn scaled(v: &YamlVal, width: f64, max_channels: f64) -> i64 {
    let c = v.as_i64().unwrap_or(0) as f64;
    make_divisible(c.min(max_channels) * width, 8)
}

fn arg_i(args: &[YamlVal], idx: usize, default: i64) -> i64 {
    args.get(idx).and_then(|a| a.as_i64()).unwrap_or(default)
}

fn arg_f(args: &[YamlVal], idx: usize, default: f64) -> f64 {
    args.get(idx).and_then(|a| a.as_f64()).unwrap_or(default)
}

fn arg_b(args: &[YamlVal], idx: usize, default: bool) -> bool {
    args.get(idx).and_then(|a| a.as_bool()).unwrap_or(default)
}

/// 从层列表推断任务（guess_model_task 的 cfg2task 语义）。
fn guess_task(layers: &[Layer]) -> Task {
    match layers.last().map(|l| l.type_name.as_str()) {
        Some(t) if t.contains("Classify") => Task::Classify,
        Some(t) if t.contains("Segment") => Task::Segment,
        Some(t) if t.contains("Pose") => Task::Pose,
        Some(t) if t.contains("OBB") => Task::Obb,
        Some(t) if t.contains("SemanticSegment") => Task::SemSeg,
        Some(t) if t.contains("Detect") => Task::Detect,
        _ => Task::Detect,
    }
}
