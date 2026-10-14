//! 特殊模块：Proto / Proto26（分割原型）、RealNVP（YOLO26-pose 流模型）、SwiGLU。

use tch::{Kind, Tensor};

use crate::conv::{Conv, RawConv};
use crate::module::{Init, VarPath};

/// ConvTranspose2d 裸层（Proto 的上采样；权重名 `upsample.weight/bias`）。
pub struct RawConvTranspose {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl RawConvTranspose {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64) -> Self {
        let bound = (1.0 / (c1 * k * k) as f64).sqrt();
        let weight = p.var_conv_transpose_weight("upsample.weight", c1, c2, k, bound);
        let bias = p.var_1d("upsample.bias", c2, Init::UniformSigned(((c1 * k * k) as f64).sqrt()));
        let _ = s;
        RawConvTranspose { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.f_conv_transpose2d(&self.weight, Some(&self.bias), [2, 2], [0, 0], [0, 0], 1, [1, 1]).unwrap()
    }
}

/// Proto：分割原型生成（P3 特征 → 2x 上采样原型）。
pub struct Proto {
    pub cv1: Conv,
    pub upsample: RawConvTranspose,
    pub cv2: Conv,
    pub cv3: Conv,
}

impl Proto {
    pub fn init(p: &VarPath, c1: i64, c_: i64, c2: i64) -> Self {
        Proto {
            cv1: Conv::init(&p.child("cv1"), c1, c_, 3, 1, None, 1, 1, true),
            upsample: RawConvTranspose::init(p, c_, c_, 2, 2),
            cv2: Conv::init(&p.child("cv2"), c_, c_, 3, 1, None, 1, 1, true),
            cv3: Conv::init(&p.child("cv3"), c_, c2, 1, 1, None, 1, 1, true),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.cv1.forward(x, train);
        let y = self.upsample.forward(&y);
        let y = self.cv2.forward(&y, train);
        self.cv3.forward(&y, train)
    }
}

/// Proto26（YOLO26-seg）：多尺度特征精修融合 + 原型 + 语义分割辅助头。
pub struct Proto26 {
    pub base: Proto,
    pub feat_refine: Vec<Conv>, // Conv(x, ch[0], 1) for x in ch[1:]
    pub feat_fuse: Conv,
    pub semseg: Option<(Conv, Conv, RawConv)>,
    pub nc: i64,
}

impl Proto26 {
    pub fn init(p: &VarPath, ch: &[i64], c_: i64, c2: i64, nc: i64) -> Self {
        let feat_refine = ch[1..]
            .iter()
            .enumerate()
            .map(|(i, &c)| Conv::init(&p.child("feat_refine").idx(i), c, ch[0], 1, 1, None, 1, 1, true))
            .collect();
        let semseg = (
            Conv::init(&p.child("semseg").idx(0), ch[0], c_, 3, 1, None, 1, 1, true),
            Conv::init(&p.child("semseg").idx(1), c_, c_, 3, 1, None, 1, 1, true),
            {
                let bound = (1.0 / c_ as f64).sqrt();
                RawConv {
                    weight: p.var_conv_weight("semseg.2.weight", nc, c_, 1, bound),
                    bias: Some(p.var_1d("semseg.2.bias", nc, Init::UniformSigned((c_ as f64).sqrt()))),
                    stride: 1,
                    padding: 0,
                    pad_h: 0,
                    pad_w: 0,
                    dilation: 1,
                    groups: 1,
                }
            },
        );
        Proto26 {
            base: Proto::init(p, c_, c_, c2),
            feat_refine,
            feat_fuse: Conv::init(&p.child("feat_fuse"), ch[0], c_, 3, 1, None, 1, 1, true),
            semseg: Some(semseg),
            nc,
        }
    }

    /// 输入 x：多尺度特征（P3 在前）。推理时返回原型；训练时另返回语义 logits。
    pub fn forward(&self, x: &[Tensor], train: bool) -> (Tensor, Option<Tensor>) {
        let mut feat = x[0].shallow_clone();
        for (i, f) in self.feat_refine.iter().enumerate() {
            let up = f.forward(&x[i + 1], train);
            let scale = 2f64.powi(i as i32 + 1);
            let oh = up.size()[2] * scale as i64;
            let ow = up.size()[3] * scale as i64;
            let up = crate::ops::up_nearest(&up, [oh, ow]);
            feat = feat + up;
        }
        let fused = self.feat_fuse.forward(&feat, train);
        let p = self.base.forward(&fused, train);
        let sem = if train {
            self.semseg
                .as_ref()
                .map(|(a, b, c)| c.forward(&b.forward(&a.forward(&feat, train), train)))
        } else {
            None
        };
        (p, sem)
    }

    /// 推理融合：移除训练期语义头。
    pub fn fuse(&mut self) {
        self.semseg = None;
    }
}

// ---------------- RealNVP（YOLO26-pose RLE 损失的流模型） ----------------

fn xavier_uniform(p: &VarPath, name: &str, out: i64, c_in: i64) -> Tensor {
    let bound = (6.0 / (c_in + out) as f64).sqrt() * 0.01;
    p.var(name, &[out, c_in], Init::UniformSigned(bound))
}

fn realnvp_net(p: &VarPath, tanh: bool) -> Vec<(Tensor, Tensor)> {
    // Sequential(Linear(2,64), SiLU, Linear(64,64), SiLU, Linear(64,2)[, Tanh])
    let mut v = Vec::new();
    for (i, (a, b)) in [(2, 64), (64, 64), (64, 2)].iter().enumerate() {
        let idx = i * 2;
        let w = xavier_uniform(p, &format!("{idx}.weight"), *b, *a);
        let bias = p.var_1d(&format!("{idx}.bias"), *b, Init::Const(0.0));
        v.push((w, bias));
    }
    let _ = tanh;
    v
}

/// RealNVP：6 层耦合层流模型（仅训练期 RLE 损失使用）。
pub struct RealNvp {
    pub mask: Tensor,
    pub s: Vec<Vec<(Tensor, Tensor)>>,
    pub t: Vec<Vec<(Tensor, Tensor)>>,
}

impl RealNvp {
    pub fn init(p: &VarPath, device: tch::Device) -> Self {
        let mask = p.var_nograd("mask", &[6, 2], Init::Const(0.0));
        tch::no_grad(|| {
            let mut mask = mask.shallow_clone();
            let _ = mask.copy_(&Tensor::from_slice(&[0f32, 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0.]).reshape([6, 2]).to_device(device));
        });
        let mut s = Vec::new();
        let mut t = Vec::new();
        for i in 0..6 {
            s.push(realnvp_net(&p.child(&format!("s.{i}")), true));
            t.push(realnvp_net(&p.child(&format!("t.{i}")), false));
        }
        RealNvp { mask, s, t }
    }

    fn run_net(net: &[(Tensor, Tensor)], x: &Tensor) -> Tensor {
        let mut h = x.matmul(&net[0].0.transpose(0, 1)) + net[0].1.reshape([1, -1]);
        h = h.silu();
        h = h.matmul(&net[1].0.transpose(0, 1)) + net[1].1.reshape([1, -1]);
        h = h.silu();
        h.matmul(&net[2].0.transpose(0, 1)) + net[2].1.reshape([1, -1])
    }

    /// log N(z; 0, I)：closed-form。
    pub fn log_prob(&self, x: &Tensor) -> Tensor {
        let n = x.size()[0];
        let mut z = x.shallow_clone();
        let mut log_det = Tensor::zeros([n], (Kind::Float, x.device()));
        for i in (0..6usize).rev() {
            let m = self.mask.get(i as i64);
            let m_inv = 1.0 - &m;
            let z_ = &m * z.shallow_clone();
            let s_out: Tensor = Self::run_net(&self.s[i], &z_) * &m_inv;
            let t_out: Tensor = Self::run_net(&self.t[i], &z_) * &m_inv;
            z = &m_inv * ((z - t_out) * (-(&s_out)).exp()) + z_;
            log_det = log_det - s_out.sum_dim_intlist(Some(&[1i64][..]), true, None);
        }
        -0.5 * z.pow_tensor_scalar(2.0).sum_dim_intlist(Some(&[1i64][..]), true, None)
            - (2.0 * std::f64::consts::PI).ln()
            + log_det
    }
}

/// SwiGLU FFN（YOLOE 用，预留）。
pub struct SwiGluFfn {
    pub w12: (Tensor, Tensor),
    pub w3: (Tensor, Tensor),
}

impl SwiGluFfn {
    pub fn init(p: &VarPath, gc: i64, ec: i64, e: i64) -> Self {
        let (w, b) = p.var_linear("w12.weight", e * ec, gc);
        let (w3, b3) = p.var_linear("w3.weight", ec, e * ec / 2);
        let _ = b;
        let _ = b3;
        SwiGluFfn { w12: (w, b), w3: (w3, b3) }
    }
}
