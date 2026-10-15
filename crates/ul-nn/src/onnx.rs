//! 纯 Rust ONNX 导出：按层 DAG 发射 IR 节点 + 权重 initializer。
//!
//! 范围：v5/v8/v9/v10 卷积全路径 + SPPF/C2f/C3k2/C3/C2fCIB/SCDown + 检测头（图内解码）
//! + PSA 注意力族（Psa/C2PSA/C2fPSA，固定输入尺寸静态图）。
//! A2C2f（v12 Area-Attention）与 classify/semseg 头显式报错（roadmap）。
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
            int64_field(&mut b, 2, 1); // FLOAT
            let mut raw = Vec::with_capacity(v.len() * 4);
            for f in v {
                raw.extend_from_slice(&f.to_le_bytes());
            }
            bytes_field(&mut b, 4, &raw);
        }
        TensorData::I64(v) => {
            int64_field(&mut b, 2, 7); // INT64
            let mut raw = Vec::with_capacity(v.len() * 10);
            for f in v {
                varint(&mut raw, *f as u64); // packed varint（含负数补码）
            }
            bytes_field(&mut b, 7, &raw);
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
    node_proto_named(op, inputs, outputs, attrs, "")
}

fn node_proto_named(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, AttrVal)], name: &str) -> Vec<u8> {
    // NodeProto: input=1, output=2, name=3, op_type=4, attribute=5
    let mut b = Vec::new();
    for i in inputs {
        str_field(&mut b, 1, i);
    }
    for o in outputs {
        str_field(&mut b, 2, o);
    }
    str_field(&mut b, 3, name);
    str_field(&mut b, 4, op);
    for (an, av) in attrs {
        let mut ab = Vec::new();
        str_field(&mut ab, 1, an);
        match av {
            AttrVal::I(v) => {
                int64_field(&mut ab, 20, 2); // type = INT
                int64_field(&mut ab, 3, *v); // i = 3
            }
            AttrVal::Is(v) => {
                int64_field(&mut ab, 20, 7); // type = INTS
                let mut raw = Vec::new();
                for x in v {
                    varint(&mut raw, *x as u64); // packed：仅值，无 tag
                }
                bytes_field(&mut ab, 8, &raw);
            }
            AttrVal::S(s) => {
                int64_field(&mut ab, 20, 3); // type = STRING
                bytes_field(&mut ab, 4, s.as_bytes()); // s = 4
            }
        }
        bytes_field(&mut b, 5, &ab);
    }
    b
}

/// ValueInfoProto：动态 batch 输入 [1,3,S,S]；输出 [1,4+nc,A] 由运行时决定。
fn value_info(name: &str, dims: &[i64]) -> Vec<u8> {
    // TypeProto { tensor_type=1: Tensor { elem_type=1, shape=2: { dim=1: { dim_value=1 } } } }
    let mut tt = Vec::new();
    let mut tensor = Vec::new();
    int64_field(&mut tensor, 1, 1); // FLOAT
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        if *d < 0 {
            // 未知维：dim_param 字符串（field 2）
            str_field(&mut dim, 2, "dyn");
        } else {
            int64_field(&mut dim, 1, *d); // Dimension.dim_value
        }
        bytes_field(&mut shape, 1, &dim);
    }
    bytes_field(&mut tensor, 2, &shape);
    bytes_field(&mut tt, 1, &tensor);
    let mut vi = Vec::new();
    str_field(&mut vi, 1, name);
    bytes_field(&mut vi, 2, &tt);
    vi
}

fn model_proto(graph: &[u8], input: &str, output: &str, out_dims: &[i64]) -> Vec<u8> {
    let mut b = Vec::new();
    int64_field(&mut b, 1, 8); // ir_version
    let mut opset = Vec::new();
    str_field(&mut opset, 1, "");
    int64_field(&mut opset, 2, 17);
    bytes_field(&mut b, 8, &opset);
    str_field(&mut b, 2, "ultralytics-rust");
    bytes_field(&mut b, 7, graph);
    bytes_field(&mut b, 11, &value_info(input, &[1, 3, 320, 320]));
    bytes_field(&mut b, 12, &value_info(output, out_dims));
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
        let name = outputs[0].to_string();
        self.nodes.push(node_proto_named(op, inputs, outputs, attrs, &name));
        outputs[0].to_string()
    }

    /// Conv（BN 融合）+ SiLU。
    fn conv_silu(&mut self, p: &str, x: &str, c: &Conv) -> String {
        let y = self.conv_bn(p, x, c);
        self.silu(&y, p)
    }

    /// Conv（BN 融合），不带激活（SPPF cv1 / SCDown cv2 等 act=False 的卷积）。
    fn conv_bn(&mut self, p: &str, x: &str, c: &Conv) -> String {
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
        y
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
            &[x, "", &scales],
            &[&y],
            &[
                // torch nn.Upsample nearest 语义 = asymmetric + floor（默认 half_pixel 会错位采样）
                ("coordinate_transformation_mode", AttrVal::S("asymmetric")),
                ("nearest_mode", AttrVal::S("floor")),
                ("mode", AttrVal::S("nearest")),
            ],
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

    /// 沿 axis 按给定尺寸拆分（Split-2）。
    fn split2(&mut self, x: &str, sizes: [i64; 2], axis: i64, hint: &str) -> (String, String) {
        let sp = self.fresh(&format!("{hint}.sp"));
        self.init_i64(&sp, &[2], sizes.to_vec());
        let a = self.fresh(&format!("{hint}.a"));
        let b = self.fresh(&format!("{hint}.b"));
        self.nodes.push(node_proto("Split", &[x, &sp], &[&a, &b], &[("axis", AttrVal::I(axis))]));
        (a, b)
    }

    /// Reshape（目标形状为 initializer）。
    fn reshape(&mut self, x: &str, shape: &[i64], hint: &str) -> String {
        let rs = self.fresh(&format!("{hint}.shape"));
        self.init_i64(&rs, &[shape.len() as i64], shape.to_vec());
        let y = self.fresh(&format!("{hint}.rs"));
        self.nodes.push(node_proto("Reshape", &[x, &rs], &[&y], &[]));
        y
    }
}

/// 固定输入尺寸下的锚点表（官方 make_anchors(feats, strides, 0.5)）：
/// 返回 ((1,2,A) anchor xy、(1,4,A) 每锚 stride、A)。二者均按 (1,C,A) 行主序布局填充。
fn anchor_tables(imgsz: i64, strides: &[f64]) -> (Vec<f32>, Vec<f32>, i64) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut per_anchor = Vec::new();
    for s in strides {
        let n = imgsz / (*s as i64);
        for y in 0..n {
            for x in 0..n {
                xs.push(x as f32 + 0.5);
                ys.push(y as f32 + 0.5);
            }
        }
        for _ in 0..n * n {
            per_anchor.push(*s as f32);
        }
    }
    let a = xs.len() as i64;
    let mut anc = xs;
    anc.extend(ys);
    // (1,4,A)：通道维在外，每通道重复整张每锚 stride 表
    let mut st4 = Vec::with_capacity(4 * per_anchor.len());
    for _ in 0..4 {
        st4.extend_from_slice(&per_anchor);
    }
    (anc, st4, a)
}

/// 图内解码（官方默认导出语义，Detect._inference）：DFL 积分 + dist2bbox×strides +
/// sigmoid 分数。标准模型输出 (1, 4+nc, A)（xywh）；e2e 输出 (1, 300, 6)（xyxy + topk）。
fn emit_decode(
    g: &mut G,
    raw: &str,
    reg_max: i64,
    nc: i64,
    imgsz: i64,
    strides: &[f64],
    e2e: bool,
    extra_n: i64,
) -> Result<String, UlError> {
    let (anc_data, st_data, a) = anchor_tables(imgsz, strides);

    // 拆原始头：[boxes (4*reg_max), scores (nc), extra (nm/nk/ne)]
    let (boxes, scores, extra) = if extra_n > 0 {
        let sp3 = g.fresh("head.sp3");
        g.init_i64(&sp3, &[3], vec![reg_max * 4, nc, extra_n]);
        let b = g.fresh("head.box");
        let c = g.fresh("head.cls");
        let e = g.fresh("head.extra");
        g.nodes
            .push(node_proto("Split", &[raw, &sp3], &[&b, &c, &e], &[("axis", AttrVal::I(1))]));
        (b, c, Some(e))
    } else {
        let (b, c) = g.split2(raw, [reg_max * 4, nc], 1, "head");
        (b, c, None)
    };

    // DFL 积分：view(1,4,rm,A) → transpose(0,2,1,3) → softmax(1) → Conv proj(0..rm-1) → (1,4,A)
    let ltrb = if reg_max > 1 {
        let r4 = g.reshape(&boxes, &[1, 4, reg_max, a], "dfl.v");
        let tp = g.fresh("dfl.t");
        g.nodes.push(node_proto(
            "Transpose",
            &[&r4],
            &[&tp],
            &[("perm", AttrVal::Is(vec![0, 2, 1, 3]))],
        ));
        let sm = g.fresh("dfl.sm");
        g.nodes.push(node_proto("Softmax", &[&tp], &[&sm], &[("axis", AttrVal::I(1))]));
        let proj = g.fresh("dfl.proj");
        let pw: Vec<f32> = (0..reg_max).map(|i| i as f32).collect();
        g.init_f32_data(&proj, &[1, reg_max, 1, 1], pw);
        let cv = g.fresh("dfl.cv");
        g.nodes.push(node_proto(
            "Conv",
            &[&sm, &proj],
            &[&cv],
            &[
                ("strides", AttrVal::Is(vec![1, 1])),
                ("pads", AttrVal::Is(vec![0, 0, 0, 0])),
                ("dilations", AttrVal::Is(vec![1, 1])),
                ("group", AttrVal::I(1)),
            ],
        ));
        g.reshape(&cv, &[1, 4, a], "dfl.out")
    } else {
        boxes
    };

    // dist2bbox(xywh=True)：x1y1 = anc − lt，x2y2 = anc + rb
    let anc = g.fresh("dec.anc");
    g.init_f32_data(&anc, &[1, 2, a], anc_data);
    let st = g.fresh("dec.str");
    g.init_f32_data(&st, &[1, 4, a], st_data);
    let (lt, rb) = g.split2(&ltrb, [2, 2], 1, "dec");
    let x1y1 = g.fresh("dec.x1y1");
    g.nodes.push(node_proto("Sub", &[&anc, &lt], &[&x1y1], &[]));
    let x2y2 = g.fresh("dec.x2y2");
    g.nodes.push(node_proto("Add", &[&anc, &rb], &[&x2y2], &[]));
    let s = g.fresh("dec.sum");
    g.nodes.push(node_proto("Add", &[&x1y1, &x2y2], &[&s], &[]));
    let half = g.fresh("dec.half");
    g.init_f32_data(&half, &[], vec![0.5]);
    let cxy = g.fresh("dec.cxy");
    g.nodes.push(node_proto("Mul", &[&s, &half], &[&cxy], &[]));
    let wh = g.fresh("dec.wh");
    g.nodes.push(node_proto("Sub", &[&x2y2, &x1y1], &[&wh], &[]));
    let dbox = g.fresh("dec.dbox");
    let dbox = if e2e {
        // end2end：xyxy（官方 decode_bboxes xywh=not end2end → False）
        g.concat(&[x1y1, x2y2], "dec")
    } else {
        g.concat(&[cxy, wh], "dec")
    };
    let dbox_px = g.fresh("dec.dbox_px");
    g.nodes.push(node_proto("Mul", &[&dbox, &st], &[&dbox_px], &[]));

    let sig = g.fresh("dec.sig");
    g.nodes.push(node_proto("Sigmoid", &[&scores], &[&sig], &[]));
    if !e2e {
        let out = if let Some(e) = extra.as_ref() {
            let e2 = g.fresh("dec.extra");
            g.nodes.push(node_proto("Identity", &[e], &[&e2], &[]));
            let o = g.fresh("output");
            g.nodes
                .push(node_proto("Concat", &[&dbox_px, &sig, &e2], &[&o], &[("axis", AttrVal::I(1))]));
            o
        } else {
            let o = g.fresh("output");
            g.nodes.push(node_proto("Concat", &[&dbox_px, &sig], &[&o], &[("axis", AttrVal::I(1))]));
            o
        };
        return Ok(out);
    }

    // ---- e2e topk 后处理（官方 postprocess + get_topk_index，k=max_det=300）----
    let k = 300i64;
    let zero_ax = g.fresh("e2e.ax");
    g.init_i64(&zero_ax, &[1], vec![2]);
    // scores (1,nc,A) → max over cls → (1,A)
    let mx = g.fresh("e2e.max");
    g.nodes
        .push(node_proto("ReduceMax", &[&sig], &[&mx], &[("axes", AttrVal::Is(vec![1])), ("keepdims", AttrVal::I(0))]));
    // ori_index = TopK(max, k)
    let kv = g.fresh("e2e.k");
    g.init_i64(&kv, &[1], vec![k]);
    let (_sv, ori) = {
        let v = g.fresh("e2e.top1_v");
        let i = g.fresh("e2e.top1_i");
        g.nodes
            .push(node_proto("TopK", &[&mx, &kv], &[&v, &i], &[("axis", AttrVal::I(1)), ("largest", AttrVal::I(1)), ("sorted", AttrVal::I(1))]));
        (v, i)
    };
    // scores_sel = sig[:, :, ori] → (1,nc,k) → 转置 (1,k,nc)
    let ori_f = g.reshape(&ori, &[-1], "e2e.ori_f"); // Gather 索引须一维（2-D 会升秩）
    let sel = g.fresh("e2e.sel");
    g.nodes.push(node_proto("Gather", &[&sig, &ori_f], &[&sel], &[("axis", AttrVal::I(2))]));
    let sel_t = g.fresh("e2e.sel_t");
    g.nodes
        .push(node_proto("Transpose", &[&sel], &[&sel_t], &[("perm", AttrVal::Is(vec![0, 2, 1]))]));
    let flat = g.reshape(&sel_t, &[1, k * nc], "e2e.flat");
    let (s2, idx) = {
        let v = g.fresh("e2e.top2_v");
        let i = g.fresh("e2e.top2_i");
        g.nodes
            .push(node_proto("TopK", &[&flat, &kv], &[&v, &i], &[("axis", AttrVal::I(1)), ("largest", AttrVal::I(1)), ("sorted", AttrVal::I(1))]));
        (v, i)
    };
    // cls_idx = idx % nc；box_row = idx / nc
    let ncc = g.fresh("e2e.nc");
    g.init_i64(&ncc, &[1], vec![nc]);
    let cls_i = g.fresh("e2e.cls_i");
    g.nodes.push(node_proto("Mod", &[&idx, &ncc], &[&cls_i], &[("fmod", AttrVal::I(0))]));
    let row = g.fresh("e2e.row");
    g.nodes.push(node_proto("Div", &[&idx, &ncc], &[&row], &[]));
    // ori 行选择 → (1,k)
    let row_f = g.reshape(&row, &[-1], "e2e.row_f");
    let ori_k = g.fresh("e2e.ori_k");
    g.nodes.push(node_proto("Gather", &[&ori, &row_f], &[&ori_k], &[("axis", AttrVal::I(1))]));
    // boxes (1,4,A) gather ori_k → (1,4,k) → 转置 (1,k,4)
    let ori_kf = g.reshape(&ori_k, &[-1], "e2e.ori_kf");
    let bg = g.fresh("e2e.bg");
    g.nodes.push(node_proto("Gather", &[&dbox_px, &ori_kf], &[&bg], &[("axis", AttrVal::I(2))]));
    let bt = g.fresh("e2e.bt");
    g.nodes
        .push(node_proto("Transpose", &[&bg], &[&bt], &[("perm", AttrVal::Is(vec![0, 2, 1]))]));
    let s2u = g.fresh("e2e.s2u");
    g.nodes
        .push(node_proto("Unsqueeze", &[&s2, &zero_ax], &[&s2u], &[]));
    let clu = g.fresh("e2e.clu");
    g.nodes.push(node_proto("Unsqueeze", &[&cls_i, &zero_ax], &[&clu], &[]));
    let cl_f = g.fresh("e2e.cl_f");
    g.nodes.push(node_proto("Cast", &[&clu], &[&cl_f], &[("to", AttrVal::I(1))]));
    let out = g.fresh("output");
    g.nodes.push(node_proto("Concat", &[&bt, &s2u, &cl_f], &[&out], &[("axis", AttrVal::I(2))]));
    Ok(out)
}

/// 导出入口。输出为官方默认导出语义的解码结果 (1, 4+nc, A)：xywh 像素 + sigmoid 分数。
pub fn export_onnx(model: &YoloModel, imgsz: u32, out_dir: &Path) -> Result<std::path::PathBuf, UlError> {
    let mut g = G { nodes: vec![], inits: vec![], tmp: 0 };
    let input_name = "images".to_string();
    let mut cur = input_name.clone();

    // 每层输出空间尺寸（PSA 的 reshape 需要具体 H/W）
    let mut hws: Vec<(i64, i64)> = vec![(imgsz as i64, imgsz as i64); model.layers.len()];
    let mut y: Vec<Option<String>> = vec![None; model.layers.len()];
    let mut cur_hw = (imgsz as i64, imgsz as i64);
    for (li, layer) in model.layers.iter().enumerate() {
        let mut inputs: Vec<String> = Vec::new();
        let mut in_hw = cur_hw;
        for f in &layer.from {
            if *f == -1 {
                inputs.push(cur.clone());
                in_hw = cur_hw;
            } else {
                let idx = if *f < 0 { (li as i64 + f) as usize } else { *f as usize };
                inputs.push(y[idx].clone().ok_or_else(|| UlError::Msg(format!("onnx: layer {li} input {f} unavailable")))?);
                in_hw = hws[idx];
            }
        }
        let prefix = format!("model.{li}");
        let out = emit_layer(&mut g, layer, &inputs, &prefix, in_hw)?;
        let ohw = module_hw(&layer.module, in_hw);
        hws[li] = ohw;
        cur_hw = ohw;
        if model.save.contains(&li) {
            y[li] = Some(out.clone());
        }
        cur = out;
    }

    // 标准检测头：追加图内解码（官方默认导出为解码后输出）
    let last_is_detect = matches!(
        &model.layers.last().expect("model has layers").module,
        Module::Head(crate::parse::HeadMod::Detect(_))
    );
    let extra_n = last_is_detect
        .then(|| model.layers.last().unwrap())
        .and_then(|l| match &l.module {
            Module::Head(crate::parse::HeadMod::Detect(dh)) => match dh.kind {
                crate::head::ExtraKind::Mask => Some(dh.nm),
                crate::head::ExtraKind::Kpt => Some(dh.nk()),
                crate::head::ExtraKind::Angle | crate::head::ExtraKind::Angle26 => Some(dh.ne),
                _ => Some(0),
            },
            _ => Some(0),
        })
        .unwrap_or(0);
    let out_name = if last_is_detect {
        emit_decode(&mut g, &cur, model.reg_max, model.nc, imgsz as i64, &model.stride, model.end2end, extra_n)?
    } else {
        cur.clone()
    };

    // GraphProto 子消息需要各自的 tag：node=1 / initializer=5 / name=2
    let mut graph = Vec::new();
    for node in &g.nodes {
        bytes_field(&mut graph, 1, node);
    }
    str_field(&mut graph, 2, "yolov");
    for init in &g.inits {
        bytes_field(&mut graph, 5, init);
    }
    // GraphProto.input(11) / output(12) 在 graph 内部
    let a = model.stride.iter().map(|s| (imgsz as f64 / s).powi(2) as i64).sum::<i64>();
    let out_dims: Vec<i64> = if model.end2end {
        vec![1, 300, 6]
    } else {
        vec![1, 4 + model.nc + extra_n, a]
    };
    bytes_field(&mut graph, 11, &value_info(&input_name, &[1, 3, imgsz as i64, imgsz as i64]));
    bytes_field(&mut graph, 12, &value_info(&out_name, &out_dims));
    let mp = model_proto(&graph, &input_name, &out_name, &out_dims);
    std::fs::create_dir_all(out_dir)?;
    let stem = Path::new(&model.yaml_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let name = out_dir.join(format!("{stem}.onnx"));
    std::fs::write(&name, mp)?;
    Ok(name)
}

fn emit_layer(g: &mut G, layer: &Layer, inputs: &[String], p: &str, hw: (i64, i64)) -> Result<String, UlError> {
    let x = &inputs[0];
    let _ = hw;
    let out = match &layer.module {
        Module::Conv(c) => g.conv_silu(p, x, c),
        Module::DwConv(c) => g.conv_silu(p, x, c),
        Module::GhostConv(gc) => {
            let a = g.conv_silu(&format!("{p}.cv1"), x, &gc.cv1);
            let b = g.conv_silu(&format!("{p}.cv2"), &a, &gc.cv2);
            g.concat(&[a, b], p)
        }
        Module::Sppf(sp) => {
            // 官方 SPPF cv1 act=False（legacy 时其后补一次 SiLU）；cv2 带激活
            let cv1 = g.conv_bn(&format!("{p}.cv1"), x, &sp.cv1);
            let cv1 = if sp.legacy { g.silu(&cv1, p) } else { cv1 };
            let m1 = g.maxpool(&cv1, sp.k, 1, sp.k / 2, &format!("{p}.m1"));
            let m2 = g.maxpool(&m1, sp.k, 1, sp.k / 2, &format!("{p}.m2"));
            let m3 = g.maxpool(&m2, sp.k, 1, sp.k / 2, &format!("{p}.m3"));
            let cat = g.concat(&[cv1, m1, m2, m3], p);
            g.conv_silu(&format!("{p}.cv2"), &cat, &sp.cv2)
        }
        Module::C2f(c) => return emit_c2f(g, &c.cv1, &c.cv2, &c.m, x, p, hw),
        Module::C3k2(k) => return emit_c2f(g, &k.base.cv1, &k.base.cv2, &k.base.m, x, p, hw),
        Module::C2fCib(k) => return emit_c2f(g, &k.base.cv1, &k.base.cv2, &k.base.m, x, p, hw),
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
        Module::Psa(m) => emit_psa(g, m, x, hw, p),
        Module::C2Psa(m) => emit_c2psa(g, m, x, hw, p),
        Module::C2fPsa(m) => return emit_c2f(g, &m.base.cv1, &m.base.cv2, &m.base.m, x, p, hw),
        Module::Concat(_) => g.concat(inputs, p),
        Module::Upsample(s) => g.upsample(x, *s, p),
        Module::Identity => x.clone(),
        Module::MaxPool2d(k, s, pd) => {
            let (k, s, pd) = (*k, *s, *pd);
            let pk = if pd < 0 { k / 2 } else { pd };
            g.maxpool(x, k, s, pk, p)
        }
        Module::ScDown(sd) => {
            // 官方 SCDown：cv1 带 SiLU；cv2 act=False（深度卷积下采样）
            let a = g.conv_silu(&format!("{p}.cv1"), x, &sd.cv1);
            g.conv_bn(&format!("{p}.cv2"), &a, &sd.cv2)
        }
        Module::Head(hm) => match hm {
            crate::parse::HeadMod::Detect(dh) => return emit_detect_head(g, dh, inputs, p, hw.0),
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

/// 估算模块输出空间尺寸（静态图尺寸跟踪用）。
fn module_hw(m: &Module, (h, w): (i64, i64)) -> (i64, i64) {
    let conv = |x: i64, k: i64, st: i64, pd: i64| (x + 2 * pd - k) / st + 1;
    match m {
        Module::Conv(c) | Module::DwConv(c) => {
            let k = c.conv.weight.size()[2];
            (conv(h, k, c.conv.stride, c.conv.pad_h), conv(w, k, c.conv.stride, c.conv.pad_w))
        }
        Module::ScDown(sd) => {
            let k = sd.cv2.conv.weight.size()[2];
            (conv(h, k, sd.cv2.conv.stride, sd.cv2.conv.pad_h), conv(w, k, sd.cv2.conv.stride, sd.cv2.conv.pad_w))
        }
        Module::Upsample(s) => (h * *s as i64, w * *s as i64),
        Module::MaxPool2d(k, st, pd) => {
            let pd = if *pd < 0 { k / 2 } else { *pd };
            (conv(h, *k, *st, pd), conv(w, *k, *st, pd))
        }
        _ => (h, w),
    }
}

fn emit_c2f(
    g: &mut G,
    cv1: &Conv,
    cv2: &Conv,
    m: &[InnerBlock],
    x: &str,
    p: &str,
    hw: (i64, i64),
) -> Result<String, UlError> {
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
            InnerBlock::Psa(psb) => emit_psa_block(g, psb, &last, hw, p),
            InnerBlock::BottleneckPsa(b, psb) => {
                let bo = emit_bottleneck(g, &format!("{p}.m.{i}.0"), &last, b);
                emit_psa_block(g, psb, &bo, hw, &format!("{p}.m.{i}.1"))
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

/// Attention 发射：qkv → 按头 reshape → split q/k/v → softmax(q^T k)·scale → v@attn^T
/// → reshape + DW pe → proj。固定 HW 静态图。
fn emit_attention(
    g: &mut G,
    attn: &crate::attention::Attention,
    x: &str,
    hw: (i64, i64),
    p: &str,
) -> String {
    let (h, w) = hw;
    let n = h * w;
    let nh = attn.num_heads;
    let kd = attn.key_dim;
    let hd = attn.head_dim;
    let qkv = g.conv_bn(&format!("{p}.qkv"), x, &attn.qkv);
    let r = g.reshape(&qkv, &[1, nh, kd * 2 + hd, n], &format!("{p}.qkv.v"));
    let sp = g.fresh(&format!("{p}.sp"));
    g.init_i64(&sp, &[3], vec![kd, kd, hd]);
    let q = g.fresh(&format!("{p}.q"));
    let k = g.fresh(&format!("{p}.k"));
    let v = g.fresh(&format!("{p}.v"));
    g.nodes
        .push(node_proto("Split", &[&r, &sp], &[&q, &k, &v], &[("axis", AttrVal::I(2))]));
    // attn = softmax((q·scale)^T @ k)
    let sc = g.fresh(&format!("{p}.scale"));
    g.init_f32_data(&sc, &[], vec![(kd as f64).powf(-0.5) as f32]);
    let qs = g.fresh(&format!("{p}.qs"));
    g.nodes.push(node_proto("Mul", &[&q, &sc], &[&qs], &[]));
    let qt = g.fresh(&format!("{p}.qt"));
    g.nodes
        .push(node_proto("Transpose", &[&qs], &[&qt], &[("perm", AttrVal::Is(vec![0, 1, 3, 2]))]));
    let am = g.fresh(&format!("{p}.mm"));
    g.nodes.push(node_proto("MatMul", &[&qt, &k], &[&am], &[]));
    let sm = g.fresh(&format!("{p}.sm"));
    g.nodes
        .push(node_proto("Softmax", &[&am], &[&sm], &[("axis", AttrVal::I(-1))]));
    let smt = g.fresh(&format!("{p}.smt"));
    g.nodes
        .push(node_proto("Transpose", &[&sm], &[&smt], &[("perm", AttrVal::Is(vec![0, 1, 3, 2]))]));
    let y = g.fresh(&format!("{p}.y"));
    g.nodes.push(node_proto("MatMul", &[&v, &smt], &[&y], &[]));
    let ch = nh * hd;
    let yim = g.reshape(&y, &[1, ch, h, w], &format!("{p}.yim"));
    let vim = g.reshape(&v, &[1, ch, h, w], &format!("{p}.vim"));
    let pe = g.conv_bn(&format!("{p}.pe"), &vim, &attn.pe);
    let add = g.add(&yim, &pe, p);
    g.conv_bn(&format!("{p}.proj"), &add, &attn.proj)
}

/// PSABlock：x + attn(x) → +ffn。
fn emit_psa_block(
    g: &mut G,
    psb: &crate::attention::PsaBlock,
    x: &str,
    hw: (i64, i64),
    p: &str,
) -> String {
    let a = emit_attention(g, &psb.attn, x, hw, &format!("{p}.attn"));
    let x1 = g.add(x, &a, &format!("{p}.a"));
    let f0 = g.conv_silu(&format!("{p}.ffn.0"), &x1, &psb.ffn.0);
    let f1 = g.conv_bn(&format!("{p}.ffn.1"), &f0, &psb.ffn.1);
    g.add(&x1, &f1, &format!("{p}.f"))
}

/// PSA：cv1 → 半通道过 attn+ffn → cat → cv2。
fn emit_psa(g: &mut G, m: &crate::attention::Psa, x: &str, hw: (i64, i64), p: &str) -> String {
    let c = m.c;
    let cv1 = g.conv_silu(&format!("{p}.cv1"), x, &m.cv1);
    let (x1, x2) = g.split2(&cv1, [c, c], 1, &format!("{p}.sp"));
    let a = emit_attention(g, &m.attn, &x2, hw, &format!("{p}.attn"));
    let b = g.add(&x2, &a, &format!("{p}.b"));
    let f0 = g.conv_silu(&format!("{p}.ffn.0"), &b, &m.ffn.0);
    let f1 = g.conv_bn(&format!("{p}.ffn.1"), &f0, &m.ffn.1);
    let bo = g.add(&b, &f1, &format!("{p}.bo"));
    let cat = g.concat(&[x1, bo], p);
    g.conv_silu(&format!("{p}.cv2"), &cat, &m.cv2)
}

/// C2PSA：cv1 → 半通道过 n 个 PSABlock → cat → cv2。
fn emit_c2psa(g: &mut G, m: &crate::attention::C2Psa, x: &str, hw: (i64, i64), p: &str) -> String {
    let c = m.c;
    let cv1 = g.conv_silu(&format!("{p}.cv1"), x, &m.cv1);
    let (x1, mut b) = g.split2(&cv1, [c, c], 1, &format!("{p}.sp"));
    for (i, blk) in m.m.iter().enumerate() {
        b = emit_psa_block(g, blk, &b, hw, &format!("{p}.m.{i}"));
    }
    let cat = g.concat(&[x1, b], p);
    g.conv_silu(&format!("{p}.cv2"), &cat, &m.cv2)
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

/// 检测头：cv2/cv3 → cat → (1, 4+nc, A)。e2e 模型发射 one2one 分支（官方导出语义）。
fn emit_detect_head(g: &mut G, dh: &crate::head::DetectHead, inputs: &[String], p: &str, _imgsz: i64) -> Result<String, UlError> {
    let empty: Vec<crate::head::Branch> = Vec::new();
    let (cv2r, cv3r) = if dh.end2end {
        (dh.o2o_cv2.as_ref().unwrap_or(&dh.cv2), dh.o2o_cv3.as_ref().unwrap_or(&dh.cv3))
    } else {
        (&dh.cv2, &dh.cv3)
    };
    let _ = empty;
    let cv2r: &[crate::head::Branch] = cv2r;
    let cv3r: &[crate::head::Branch] = cv3r;
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
        // 官方：三层展平后沿锚点维（dim=-1）拼接
        let y = g.fresh(&format!("{p}.{name}.cat"));
        let refs: Vec<&str> = parts.iter().map(|x| x.as_str()).collect();
        g.nodes.push(node_proto("Concat", &refs, &[&y], &[("axis", AttrVal::I(-1))]));
        y
    };

    let boxes = emit_branch(g, "cv2", cv2r, dh.reg_max * 4);
    let scores = emit_branch(g, "cv3", cv3r, dh.nc);
    let main = g.concat(&[boxes, scores], p);
    // 附加通道（seg coef / pose kpt / obb angle）
    if !dh.cv4.is_empty() {
        let mode = match dh.kind {
            crate::head::ExtraKind::Kpt => Cv4Mode::Pose { k: dh.kpt_shape.0, ndim: dh.kpt_shape.1 },
            crate::head::ExtraKind::Angle => Cv4Mode::Angle,
            _ => Cv4Mode::Raw,
        };
        return Ok(emit_cv4_concat(g, &dh.cv4, inputs, &main, match dh.kind {
            crate::head::ExtraKind::Mask => dh.nm,
            crate::head::ExtraKind::Kpt => dh.nk(),
            crate::head::ExtraKind::Angle | crate::head::ExtraKind::Angle26 => dh.ne,
            _ => 0,
        }, p, mode, &dh.stride, _imgsz));
    }
    Ok(main)
}

/// cv4 附加分支（seg coef / pose kpt / obb angle）：与 cv2/cv3 同构发射 → cat 到主输出。
/// mode：None=原样拼接；Pose=按层解码 kpts；Angle=(sigmoid−0.25)·π。
fn emit_cv4_concat(
    g: &mut G,
    cv4: &[crate::head::Branch],
    inputs: &[String],
    main: &str,
    out_ch: i64,
    p: &str,
    mode: Cv4Mode,
    strides: &[f64],
    imgsz: i64,
) -> String {
    let mut parts = vec![main.to_string()];
    let mut cv4_parts = Vec::new();
    for (i, br) in cv4.iter().enumerate() {
        let xi_input = inputs.get(i).cloned().unwrap_or_default();
        let s0 = match &br.stage0 {
            crate::head::BranchStage::Conv(c) => g.conv_silu(&format!("{p}.cv4.{i}.0"), &xi_input, c),
            crate::head::BranchStage::Pair(dw, c) => {
                let d = g.conv_silu(&format!("{p}.cv4.{i}.0.0"), &xi_input, dw);
                g.conv_silu(&format!("{p}.cv4.{i}.0.1"), &d, c)
            }
        };
        let s1 = match &br.stage1 {
            crate::head::BranchStage::Conv(c) => g.conv_silu(&format!("{p}.cv4.{i}.1"), &s0, c),
            crate::head::BranchStage::Pair(dw, c) => {
                let d = g.conv_silu(&format!("{p}.cv4.{i}.1.0"), &s0, dw);
                g.conv_silu(&format!("{p}.cv4.{i}.1.1"), &d, c)
            }
        };
        let o = g.conv_plain(&format!("{p}.cv4.{i}.2"), &s1, &br.final_conv.weight, Some(&br.final_conv.bias), 1, 0, 1);
        let rn = g.fresh(&format!("{p}.cv4.{i}.shape"));
        g.init_i64(&rn, &[3], vec![1, out_ch, -1]);
        let r = g.fresh(&format!("{p}.cv4.{i}.r"));
        g.nodes.push(node_proto("Reshape", &[&o, &rn], &[&r], &[]));
        cv4_parts.push(r);
    }
    // 官方：三层沿锚点维拼接
    let cat = g.fresh(&format!("{p}.cv4.cat"));
    let refs: Vec<&str> = cv4_parts.iter().map(|x| x.as_str()).collect();
    g.nodes.push(node_proto("Concat", &refs, &[&cat], &[("axis", AttrVal::I(-1))]));
    let cat_ref = cat.clone();
    parts.push(cat);
    // Angle：变换 (sigmoid−0.25)·π
    if let Cv4Mode::Angle = mode {
        let sig = g.fresh(&format!("{p}.ang.sig"));
        g.nodes.push(node_proto("Sigmoid", &[&cat_ref], &[&sig], &[]));
        let c25 = g.fresh(&format!("{p}.ang.c"));
        g.init_f32_data(&c25, &[], vec![0.25]);
        let shifted = g.fresh(&format!("{p}.ang.sh"));
        g.nodes.push(node_proto("Sub", &[&sig, &c25], &[&shifted], &[]));
        let piv = g.fresh(&format!("{p}.ang.pi"));
        g.init_f32_data(&piv, &[], vec![std::f32::consts::PI]);
        let out_a = g.fresh(&format!("{p}.ang.out"));
        g.nodes.push(node_proto("Mul", &[&shifted, &piv], &[&out_a], &[]));
        let out = g.fresh(&format!("{p}.with_extra"));
        let pr: Vec<&str> = parts.iter().map(|x| x.as_str()).collect();
        let o = g.fresh(&format!("{p}.with_extra"));
        g.nodes.push(node_proto("Concat", &pr, &[&o], &[("axis", AttrVal::I(1))]));
        return o;
    }
    let out = g.fresh(&format!("{p}.with_extra"));
    let pr: Vec<&str> = parts.iter().map(|x| x.as_str()).collect();
    g.nodes.push(node_proto("Concat", &pr, &[&out], &[("axis", AttrVal::I(1))]));
    out
}

#[derive(Clone, Copy)]
enum Cv4Mode {
    Raw,
    Pose { k: i64, ndim: i64 },
    Angle,
}

fn bail<T>(msg: impl Into<String>) -> Result<T, UlError> {
    Err(UlError::Msg(msg.into()))
}
