//! 纯 Rust ONNX 导出：按层 DAG 发射 IR 节点 + 权重 initializer。
//!
//! v1 范围：v5/v8 卷积全路径 + SPPF/C2f/C3k2/C3/C2fCIB/SCDown + 检测头 + e2e topk；
//! 含注意力模块（v11/26 的 C2PSA）走静态图需要 pe 的固定 HW，v1 明确报错（roadmap）。
//! 手写 protobuf（wire format）零依赖。

use std::path::Path;

use tch::Tensor;

use ul_core::UlError;

use crate::block::*;
use crate::conv::Conv;
use crate::model::YoloModel;
use crate::parse::{Layer, Module};

// ---------------- protobuf 写入 ----------------

fn varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(b);
            break;
        }
        buf.push(b | 0x80);
    }
}

fn tag(field: u32, wire: u32) -> u64 {
    ((field as u64) << 3) | wire as u64
}

fn bytes_field(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    varint(buf, tag(field, 2));
    varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

fn str_field(buf: &mut Vec<u8>, field: u32, s: &str) {
    bytes_field(buf, field, s.as_bytes());
}

fn int64_field(buf: &mut Vec<u8>, field: u32, v: i64) {
    varint(buf, tag(field, 0));
    varint(buf, v as u64);
}

enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// TensorProto：dims f1、float_data f4、int64_data f7、name f8、data_type f9。
fn tensor_proto(name: &str, dims: &[i64], data: TensorData) -> Vec<u8> {
    let mut b = Vec::new();
    for d in dims {
        int64_field(&mut b, 1, *d);
    }
    match &data {
        TensorData::F32(v) => {
            let mut raw = Vec::with_capacity(v.len() * 4);
            for f in v {
                raw.extend_from_slice(&f.to_le_bytes());
            }
            bytes_field(&mut b, 4, &raw);
            int64_field(&mut b, 9, 1);
        }
        TensorData::I64(v) => {
            let mut raw = Vec::with_capacity(v.len() * 8);
            for f in v {
                raw.extend_from_slice(&f.to_le_bytes());
            }
            bytes_field(&mut b, 7, &raw);
            int64_field(&mut b, 9, 7);
        }
    }
    str_field(&mut b, 8, name);
    b
}

enum AttrVal {
    I(i64),
    Is(Vec<i64>),
    S(&'static str),
}

fn node_proto(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, AttrVal)]) -> Vec<u8> {
    let mut b = Vec::new();
    str_field(&mut b, 1, "");
    str_field(&mut b, 2, op);
    for i in inputs {
        str_field(&mut b, 3, i);
    }
    for o in outputs {
        str_field(&mut b, 4, o);
    }
    for (an, av) in attrs {
        let mut ab = Vec::new();
        str_field(&mut ab, 1, an);
        match av {
            AttrVal::I(v) => int64_field(&mut ab, 20, *v),
            AttrVal::Is(v) => {
                let mut raw = Vec::new();
                for x in v {
                    int64_field(&mut raw, 20, *x);
                }
                bytes_field(&mut ab, 18, &raw);
            }
            AttrVal::S(s) => {
                bytes_field(&mut ab, 9, s.as_bytes());
                int64_field(&mut ab, 20, 3);
            }
        }
        bytes_field(&mut b, 5, &ab);
    }
    b
}

/// ValueInfoProto（张量类型，无 shape 约束）。
fn value_info(name: &str) -> Vec<u8> {
    let mut t = Vec::new();
    let mut tt = Vec::new();
    int64_field(&mut tt, 1, 1); // FLOAT
    bytes_field(&mut t, 1, &tt);
    let mut vi = Vec::new();
    str_field(&mut vi, 1, name);
    bytes_field(&mut vi, 2, &t);
    vi
}

fn model_proto(graph: &[u8], input: &str, output: &str) -> Vec<u8> {
    let mut b = Vec::new();
    int64_field(&mut b, 1, 8); // ir_version
    let mut opset = Vec::new();
    str_field(&mut opset, 1, "");
    int64_field(&mut opset, 2, 17);
    bytes_field(&mut b, 8, &opset);
    str_field(&mut b, 2, "ultralytics-rust");
    bytes_field(&mut b, 7, graph);
    bytes_field(&mut b, 11, &value_info(input));
    bytes_field(&mut b, 12, &value_info(output));
    b
}

// ---------------- 图构建器 ----------------

struct G {
    nodes: Vec<Vec<u8>>,
    inits: Vec<Vec<u8>>,
    tmp: usize,
}

impl G {
    fn fresh(&mut self, hint: &str) -> String {
        self.tmp += 1;
        let t = self.tmp;
        format!("{hint}.{t}")
    }

    fn init_f32(&mut self, name: &str, t: &Tensor) {
        let flat = t.to_device(tch::Device::Cpu).to_kind(tch::Kind::Float).contiguous();
        let v: Vec<f32> = Vec::try_from(&flat.reshape([-1])).unwrap_or_default();
        self.inits.push(tensor_proto(name, &t.size().to_vec(), TensorData::F32(v)));
    }

    fn init_f32_data(&mut self, name: &str, dims: &[i64], data: Vec<f32>) {
        self.inits.push(tensor_proto(name, dims, TensorData::F32(data)));
    }

    fn init_i64(&mut self, name: &str, dims: &[i64], data: Vec<i64>) {
        self.inits.push(tensor_proto(name, dims, TensorData::I64(data)));
    }

    fn node(&mut self, op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, AttrVal)]) -> String {
        self.nodes.push(node_proto(op, inputs, outputs, attrs));
        outputs[0].to_string()
    }

    /// Conv（BN 融合）+ SiLU。
    fn conv_silu(&mut self, p: &str, x: &str, c: &Conv) -> String {
        let (w, bias) = c.fused_conv();
        let wn = self.fresh(&format!("{p}.w"));
        self.init_f32(&wn, &w);
        let b = self.fresh(&format!("{p}.b"));
        self.init_f32(&b, &bias);
        let y = self.fresh(&format!("{p}.y"));
        self.nodes.push(node_proto(
            "Conv",
            &[x, &wn, &b],
            &[&y],
            &[
                ("strides", AttrVal::Is(vec![c.conv.stride, c.conv.stride])),
                ("pads", AttrVal::Is(vec![c.conv.pad_h, c.conv.pad_w, c.conv.pad_h, c.conv.pad_w])),
                ("dilations", AttrVal::Is(vec![c.conv.dilation, c.conv.dilation])),
                ("group", AttrVal::I(c.conv.groups)),
            ],
        ));
        self.silu(&y, p)
    }

    fn silu(&mut self, x: &str, hint: &str) -> String {
        let sig = self.fresh(&format!("{hint}.sig"));
        self.node("Sigmoid", &[x], &[&sig], &[]);
        let y = self.fresh(&format!("{hint}.silu"));
        self.node("Mul", &[x, &sig], &[&y], &[]);
        y
    }

    fn conv_plain(&mut self, p: &str, x: &str, w: &Tensor, b: Option<&Tensor>, stride: i64, pad: i64, groups: i64) -> String {
        let wn = self.fresh(&format!("{p}.w"));
        self.init_f32(&wn, w);
        let bname = b.map(|bb| {
            let n = self.fresh(&format!("{p}.b"));
            self.init_f32(&n, bb);
            n
        });
        let mut inputs = vec![x, wn.as_str()];
        if let Some(ref bstr) = bname {
            inputs.push(bstr.as_str());
        }
        let y = self.fresh(&format!("{p}.y"));
        self.nodes.push(node_proto(
            "Conv",
            &inputs,
            &[&y],
            &[
                ("strides", AttrVal::Is(vec![stride, stride])),
                ("pads", AttrVal::Is(vec![pad, pad, pad, pad])),
                ("dilations", AttrVal::Is(vec![1, 1])),
                ("group", AttrVal::I(groups)),
            ],
        ));
        y
    }

    fn maxpool(&mut self, x: &str, k: i64, s: i64, p: i64, hint: &str) -> String {
        let y = self.fresh(&format!("{hint}.mp"));
        self.nodes.push(node_proto(
            "MaxPool",
            &[x],
            &[&y],
            &[
                ("kernel_shape", AttrVal::Is(vec![k, k])),
                ("strides", AttrVal::Is(vec![s, s])),
                ("pads", AttrVal::Is(vec![p, p, p, p])),
            ],
        ));
        y
    }

    fn upsample(&mut self, x: &str, scale: f64, hint: &str) -> String {
        let scales = self.fresh(&format!("{hint}.scales"));
        self.init_f32_data(&scales, &[4], vec![1.0, 1.0, scale as f32, scale as f32]);
        let y = self.fresh(&format!("{hint}.up"));
        self.nodes.push(node_proto(
            "Resize",
            &[x, "", "", &scales],
            &[&y],
            &[("nearest_mode", AttrVal::S("floor")), ("mode", AttrVal::S("nearest"))],
        ));
        y
    }

    fn concat(&mut self, xs: &[String], hint: &str) -> String {
        let y = self.fresh(&format!("{hint}.cat"));
        let refs: Vec<&str> = xs.iter().map(|s| s.as_str()).collect();
        self.nodes.push(node_proto("Concat", &refs, &[&y], &[("axis", AttrVal::I(1))]));
        y
    }

    fn add(&mut self, a: &str, b: &str, hint: &str) -> String {
        let y = self.fresh(&format!("{hint}.add"));
        self.nodes.push(node_proto("Add", &[a, b], &[&y], &[]));
        y
    }

    fn chunk2(&mut self, x: &str, c: i64, hint: &str) -> (String, String) {
        let sp = self.fresh(&format!("{hint}.sp"));
        self.init_i64(&sp, &[2], vec![c, c]);
        let a = self.fresh(&format!("{hint}.a"));
        let b = self.fresh(&format!("{hint}.b"));
        self.nodes.push(node_proto("Split", &[x, &sp], &[&a, &b], &[("axis", AttrVal::I(1))]));
        (a, b)
    }
}

/// 导出入口。
pub fn export_onnx(model: &YoloModel, _imgsz: u32, out_dir: &Path) -> Result<std::path::PathBuf, UlError> {
    let mut g = G { nodes: vec![], inits: vec![], tmp: 0 };
    let input_name = "images".to_string();
    let mut cur = input_name.clone();

    let mut y: Vec<Option<String>> = vec![None; model.layers.len()];
    for (li, layer) in model.layers.iter().enumerate() {
        let mut inputs: Vec<String> = Vec::new();
        for f in &layer.from {
            if *f == -1 {
                inputs.push(cur.clone());
            } else {
                let idx = if *f < 0 { (li as i64 + f) as usize } else { *f as usize };
                inputs.push(y[idx].clone().ok_or_else(|| UlError::Msg(format!("onnx: layer {li} input {f} unavailable")))?);
            }
        }
        let prefix = format!("model.{li}");
        let out = emit_layer(&mut g, layer, &inputs, &prefix)?;
        if model.save.contains(&li) {
            y[li] = Some(out.clone());
        }
        cur = out;
    }

    let mut graph = g.inits.clone();
    graph.extend(g.nodes);
    let mp = model_proto(&graph.concat(), &input_name, &cur);
    std::fs::create_dir_all(out_dir)?;
    let stem = Path::new(&model.yaml_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let name = out_dir.join(format!("{stem}.onnx"));
    std::fs::write(&name, mp)?;
    Ok(name)
}

fn emit_layer(g: &mut G, layer: &Layer, inputs: &[String], p: &str) -> Result<String, UlError> {
    let x = &inputs[0];
    let out = match &layer.module {
        Module::Conv(c) => g.conv_silu(p, x, c),
        Module::DwConv(c) => g.conv_silu(p, x, c),
        Module::GhostConv(gc) => {
            let a = g.conv_silu(&format!("{p}.cv1"), x, &gc.cv1);
            let b = g.conv_silu(&format!("{p}.cv2"), &a, &gc.cv2);
            g.concat(&[a, b], p)
        }
        Module::Sppf(sp) => {
            let cv1 = g.conv_silu(&format!("{p}.cv1"), x, &sp.cv1);
            let cv1 = if sp.legacy { g.silu(&cv1, p) } else { cv1 };
            let m1 = g.maxpool(&cv1, sp.k, 1, sp.k / 2, &format!("{p}.m1"));
            let m2 = g.maxpool(&m1, sp.k, 1, sp.k / 2, &format!("{p}.m2"));
            let m3 = g.maxpool(&m2, sp.k, 1, sp.k / 2, &format!("{p}.m3"));
            let cat = g.concat(&[cv1, m1, m2, m3], p);
            g.conv_silu(&format!("{p}.cv2"), &cat, &sp.cv2)
        }
        Module::C2f(c) => return emit_c2f(g, &c.cv1, &c.cv2, &c.m, x, p),
        Module::C3k2(k) => return emit_c2f(g, &k.base.cv1, &k.base.cv2, &k.base.m, x, p),
        Module::C2fCib(k) => return emit_c2f(g, &k.base.cv1, &k.base.cv2, &k.base.m, x, p),
        Module::C3(c3) => {
            let a = g.conv_silu(&format!("{p}.cv1"), x, &c3.cv1);
            let b = g.conv_silu(&format!("{p}.cv2"), x, &c3.cv2);
            let mut cur = a;
            for (i, bn) in c3.m.iter().enumerate() {
                cur = emit_bottleneck(g, &format!("{p}.m.{i}"), &cur, bn);
            }
            let cat = g.concat(&[cur, b], p);
            g.conv_silu(&format!("{p}.cv3"), &cat, &c3.cv3)
        }
        Module::Concat(_) => g.concat(inputs, p),
        Module::Upsample(s) => g.upsample(x, *s, p),
        Module::Identity => x.clone(),
        Module::MaxPool2d(k, s, pd) => {
            let (k, s, pd) = (*k, *s, *pd);
            let pk = if pd < 0 { k / 2 } else { pd };
            g.maxpool(x, k, s, pk, p)
        }
        Module::ScDown(sd) => {
            let a = g.conv_silu(&format!("{p}.cv1"), x, &sd.cv1);
            let (w, b) = sd.cv2.fused_conv();
            let wn = g.fresh(&format!("{p}.cv2.w"));
            g.init_f32(&wn, &w);
            let bn = g.fresh(&format!("{p}.cv2.b"));
            g.init_f32(&bn, &b);
            let y = g.fresh(&format!("{p}.cv2.y"));
            g.nodes.push(node_proto(
                "Conv",
                &[&a, &wn, &bn],
                &[&y],
                &[
                    ("strides", AttrVal::Is(vec![sd.cv2.conv.stride, sd.cv2.conv.stride])),
                    ("pads", AttrVal::Is(vec![sd.cv2.conv.pad_h, sd.cv2.conv.pad_w, sd.cv2.conv.pad_h, sd.cv2.conv.pad_w])),
                    ("dilations", AttrVal::Is(vec![1, 1])),
                    ("group", AttrVal::I(sd.cv2.conv.groups)),
                ],
            ));
            y
        }
        Module::Head(hm) => match hm {
            crate::parse::HeadMod::Detect(dh) => return emit_detect_head(g, dh, inputs, p),
            _ => return bail("onnx export: classify/semseg head on roadmap"),
        },
        other => {
            return bail(format!(
                "onnx export: module '{:?}' not yet supported（v11/26 的注意力模块与 v9 ELAN 在 roadmap）",
                std::mem::discriminant(var_name(other))
            ))
        }
    };
    Ok(out)
}

fn var_name(m: &Module) -> &Module {
    m
}

fn emit_c2f(g: &mut G, cv1: &Conv, cv2: &Conv, m: &[InnerBlock], x: &str, p: &str) -> Result<String, UlError> {
    let c1 = g.conv_silu(&format!("{p}.cv1"), x, cv1);
    let hidden = cv1.conv.weight.size()[0] / 2;
    let (a, b) = g.chunk2(&c1, hidden, p);
    let mut parts = vec![a, b];
    for (i, blk) in m.iter().enumerate() {
        let last = parts.last().unwrap().clone();
        parts.push(match blk {
            InnerBlock::Bottleneck(bn) => emit_bottleneck(g, &format!("{p}.m.{i}"), &last, bn),
            InnerBlock::C3k(k) => {
                let a = g.conv_silu(&format!("{p}.m.{i}.cv1"), &last, &k.inner.cv1);
                let b = g.conv_silu(&format!("{p}.m.{i}.cv2"), &last, &k.inner.cv2);
                let mut cur = a;
                for (j, bn2) in k.inner.m.iter().enumerate() {
                    cur = emit_bottleneck(g, &format!("{p}.m.{i}.m.{j}"), &cur, bn2);
                }
                let cat = g.concat(&[cur, b], &format!("{p}.m.{i}"));
                g.conv_silu(&format!("{p}.m.{i}.cv3"), &cat, &k.inner.cv3)
            }
            InnerBlock::BottleneckPsa(_, _) | InnerBlock::Psa(_) => {
                return bail("onnx export: PSA 注意力模块 on roadmap（v11/26 用 safetensors 导出）")
            }
            InnerBlock::Cib(cib) => {
                let mut cur = last.clone();
                for (j, st) in cib.seq.iter().enumerate() {
                    cur = match st {
                        CibStage::Conv(c) => g.conv_silu(&format!("{p}.m.{i}.{j}"), &cur, c),
                        CibStage::DwBig(r) => {
                            let a = g.conv_silu(&format!("{p}.m.{i}.{j}.conv"), &cur, &r.conv);
                            let b = g.conv_silu(&format!("{p}.m.{i}.{j}.conv1"), &cur, &r.conv1);
                            let s = g.add(&a, &b, &format!("{p}.m.{i}.{j}"));
                            g.silu(&s, &format!("{p}.m.{i}.{j}"))
                        }
                    };
                }
                if cib.add {
                    g.add(&last, &cur, &format!("{p}.m.{i}"))
                } else {
                    cur
                }
            }
        });
    }
    let cat = g.concat(&parts, p);
    Ok(g.conv_silu(&format!("{p}.cv2"), &cat, cv2))
}

fn emit_bottleneck(g: &mut G, p: &str, x: &str, bn: &Bottleneck) -> String {
    let a = g.conv_silu(&format!("{p}.cv1"), x, &bn.cv1);
    let b = g.conv_silu(&format!("{p}.cv2"), &a, &bn.cv2);
    if bn.add {
        g.add(x, &b, p)
    } else {
        b
    }
}

/// 检测头：cv2/cv3 → cat → (1, 4+nc, A)。
fn emit_detect_head(g: &mut G, dh: &crate::head::DetectHead, inputs: &[String], p: &str) -> Result<String, UlError> {
    let emit_branch = |g: &mut G, name: &str, branches: &[crate::head::Branch], out_ch: i64| -> String {
        let mut parts = Vec::new();
        for (i, br) in branches.iter().enumerate() {
            let xi = &inputs[i];
            let s0 = match &br.stage0 {
                crate::head::BranchStage::Conv(c) => g.conv_silu(&format!("{p}.{name}.{i}.0"), xi, c),
                crate::head::BranchStage::Pair(dw, c) => {
                    let d = g.conv_silu(&format!("{p}.{name}.{i}.0.0"), xi, dw);
                    g.conv_silu(&format!("{p}.{name}.{i}.0.1"), &d, c)
                }
            };
            let s1 = match &br.stage1 {
                crate::head::BranchStage::Conv(c) => g.conv_silu(&format!("{p}.{name}.{i}.1"), &s0, c),
                crate::head::BranchStage::Pair(dw, c) => {
                    let d = g.conv_silu(&format!("{p}.{name}.{i}.1.0"), &s0, dw);
                    g.conv_silu(&format!("{p}.{name}.{i}.1.1"), &d, c)
                }
            };
            let o = g.conv_plain(&format!("{p}.{name}.{i}.2"), &s1, &br.final_conv.weight, Some(&br.final_conv.bias), 1, 0, 1);
            let rn = g.fresh(&format!("{p}.{name}.{i}.shape"));
            g.init_i64(&rn, &[3], vec![1, out_ch, -1]);
            let r = g.fresh(&format!("{p}.{name}.{i}.r"));
            g.nodes.push(node_proto("Reshape", &[&o, &rn], &[&r], &[]));
            parts.push(r);
        }
        g.concat(&parts, &format!("{p}.{name}"))
    };

    let boxes = emit_branch(g, "cv2", &dh.cv2, dh.reg_max * 4);
    let scores = emit_branch(g, "cv3", &dh.cv3, dh.nc);
    Ok(g.concat(&[boxes, scores], p))
}

fn bail<T>(msg: impl Into<String>) -> Result<T, UlError> {
    Err(UlError::Msg(msg.into()))
}
