//! 卷积族模块：Conv / DWConv / Conv2 / ConvTranspose / Focus / GhostConv / LightConv / RepConv。
//!
//! 参数名与 ultralytics state_dict 逐字对齐（`conv.weight`、`bn.running_mean`、
//! `cv2.0.conv.weight`……），使官方 .pt 权重可以直接按名字加载。
//! 初始化遵循 PyTorch Conv2d 默认（kaiming_uniform a=√5）+ ultralytics BN（eps 1e-3 / momentum 0.03）。

use tch::{Kind, Tensor};

use crate::module::{Init, VarPath};

/// SiLU：x · σ(x)。
pub fn silu(x: &Tensor) -> Tensor {
    x * x.sigmoid()
}

/// ultralytics autopad(k, p, d)。
pub fn autopad(k: i64, p: Option<i64>, d: i64) -> i64 {
    let k_eff = if d > 1 { d * (k - 1) + 1 } else { k };
    p.unwrap_or(k_eff / 2)
}

/// 批归一化（手写，精确控制 torch native_batch_norm 语义与 running 统计更新）。
pub struct BatchNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub eps: f64,
    pub momentum: f64,
}

impl BatchNorm {
    /// ultralytics：eps=1e-3，momentum=0.03；weight=1 / bias=0 / running_var=1。
    pub fn init(p: &VarPath, c: i64) -> Self {
        let weight = p.var_1d("bn.weight", c, Init::Const(1.0));
        let bias = p.var_1d("bn.bias", c, Init::Const(0.0));
        let running_mean = p.var_1d_nograd("bn.running_mean", c, Init::Const(0.0));
        let running_var = p.var_1d_nograd("bn.running_var", c, Init::Const(1.0));
        BatchNorm { weight, bias, running_mean, running_var, eps: 1e-3, momentum: 0.03 }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        x.f_batch_norm(
            Some(&self.weight),
            Some(&self.bias),
            Some(&self.running_mean),
            Some(&self.running_var),
            train,
            self.momentum,
            self.eps,
            false,
        )
        .unwrap()
    }

    /// 推理融合用：返回 (scale=γ/σ, shift=β−μγ/σ)。
    pub fn fuse_params(&self) -> (Tensor, Tensor) {
        let std = (&self.running_var + self.eps).sqrt();
        let scale = &self.weight / &std;
        let shift = &self.bias - &self.running_mean * (&self.weight / &std);
        (scale, shift)
    }
}

/// 标准卷积：Conv2d(+bias 可选) —— 不含 BN/激活（由 Conv 组合）。
pub struct RawConv {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: i64,
    pub padding: i64,
    pub pad_h: i64,
    pub pad_w: i64,
    pub dilation: i64,
    pub groups: i64,
}

impl RawConv {
    /// PyTorch Conv2d 默认初始化：kaiming_uniform(a=√5) + 偏置 U(±1/√fan_in)。
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, bias: bool, groups: i64) -> Self {
        let fan_in = c1 * k * k / groups.max(1);
        let bound = (1.0 / fan_in as f64).sqrt();
        let weight = p.var_conv_weight("weight", c2, c1 / groups.max(1), k, bound);
        let bias = if bias { Some(p.var_1d("bias", c2, Init::UniformSigned(1.0 / (fan_in as f64).sqrt()))) } else { None };
        RawConv { weight, bias, stride: 1, padding: 0, pad_h: 0, pad_w: 0, dilation: 1, groups }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.f_conv2d(
            &self.weight,
            self.bias.as_ref(),
            &[self.stride, self.stride],
            &[self.pad_h, self.pad_w],
            &[self.dilation, self.dilation],
            self.groups,
        )
        .unwrap()
    }
}

/// ultralytics `Conv` = Conv2d + BN + SiLU。
pub struct Conv {
    pub conv: RawConv,
    pub bn: BatchNorm,
    pub act: bool,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64, p_pad: Option<i64>, g: i64, d: i64, act: bool) -> Self {
        let pad = autopad(k, p_pad, d);
        let conv = RawConv::init(&p.child("conv"), c1, c2, k, false, g);
        let mut conv = conv;
        conv.stride = s;
        conv.pad_h = pad;
        conv.pad_w = pad;
        conv.padding = pad;
        conv.dilation = d;
        conv.pad_h = pad;
        conv.pad_w = pad;
        let bn = BatchNorm::init(p, c2);
        Conv { conv, bn, act }
    }

    /// 矩形 kernel 构造（C3x 等）：k=(kh,kw)，pad=(ph,pw)。
    #[allow(clippy::too_many_arguments)]
    pub fn init_rect(
        p: &VarPath,
        c1: i64,
        c2: i64,
        kh: i64,
        kw: i64,
        s: i64,
        pad: (i64, i64),
        g: i64,
        act: bool,
    ) -> Self {
        let conv_p = p.child("conv");
        let fan_in = c1 * kh * kw / g.max(1);
        let bound = (1.0 / fan_in as f64).sqrt();
        let weight = conv_p.var("conv.weight", &[c2, c1 / g.max(1), kh, kw], crate::module::Init::UniformSigned(bound));
        let mut conv = RawConv { weight, bias: None, stride: s, padding: 0, pad_h: pad.0, pad_w: pad.1, dilation: 1, groups: g };
        let _ = &mut conv;
        let bn = BatchNorm::init(p, c2);
        Conv { conv, bn, act }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.conv.forward(x);
        let y = self.bn.forward(&y, train);
        if self.act {
            silu(&y)
        } else {
            y
        }
    }

    /// 推理模式融合（conv+BN 合一）：返回融合后的 (weight, bias)。
    pub fn fused_conv(&self) -> (Tensor, Tensor) {
        let w = &self.conv.weight;
        let (scale, shift) = self.bn.fuse_params();
        let w = w * scale.reshape([scale.size()[0], 1, 1, 1]);
        (w, shift)
    }
}

/// `DWConv`：depthwise（groups = gcd(c1,c2)，标准用法下 = c1）。
pub fn dwconv_init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64, d: i64, act: bool) -> Conv {
    let g = gcd(c1, c2);
    Conv::init(p, c1, c2, k, s, None, g, d, act)
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// `Conv2`：RepConv 简化版（3x3 + 并行 1x1 加和）。
pub struct Conv2 {
    pub conv: RawConv, // 3x3
    pub cv2: RawConv,  // 1x1
    pub bn: BatchNorm,
    pub act: bool,
}

impl Conv2 {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64, g: i64, d: i64, act: bool) -> Self {
        let conv_p = p.child("conv");
        let mut conv = RawConv::init(&conv_p, c1, c2, k, false, g);
        conv.stride = s;
        conv.padding = autopad(k, None, d);
        conv.dilation = d;
        let mut cv2 = RawConv::init(&p.child("cv2"), c1, c2, 1, false, g);
        cv2.stride = s;
        cv2.padding = autopad(1, None, d);
        cv2.pad_h = cv2.padding;
        cv2.pad_w = cv2.padding;
        cv2.dilation = d;
        let bn = BatchNorm::init(p, c2);
        Conv2 { conv, cv2, bn, act }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.conv.forward(x) + self.cv2.forward(x);
        let y = self.bn.forward(&y, train);
        if self.act {
            silu(&y)
        } else {
            y
        }
    }
}

/// `ConvTranspose` = ConvTranspose2d + BN + SiLU。
pub struct ConvTranspose {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub bn: Option<BatchNorm>,
    pub act: bool,
    pub k: i64,
    pub s: i64,
    pub p: i64,
}

impl ConvTranspose {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64, pad: i64, bn: bool, act: bool) -> Self {
        // torch ConvTranspose2d 权重形状 (c1, c2, k, k)（输入通道在前）
        let weight = p.var_conv_transpose_weight("conv_transpose.weight", c1, c2, k, (1.0 / (c1 * k * k) as f64).sqrt());
        let bias = if bn { None } else { Some(p.var_1d("conv_transpose.bias", c2, Init::UniformSigned(((c1 * k * k) as f64).sqrt()))) };
        let bn_m = if bn { Some(BatchNorm::init(p, c2)) } else { None };
        ConvTranspose { weight, bias, bn: bn_m, act, k, s, p: pad }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = x
            .f_conv_transpose2d(
                &self.weight,
                self.bias.as_ref(),
                [self.s, self.s],
                [self.p, self.p],
                [0, 0],
                1,
                [1, 1],
            )
            .unwrap();
        let y = match &self.bn {
            Some(bn) => bn.forward(&y, train),
            None => y,
        };
        if self.act {
            silu(&y)
        } else {
            y
        }
    }
}

/// `Focus`：空间切片 4 份拼接后过 Conv。
pub struct Focus {
    pub conv: Conv,
}

impl Focus {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64, g: i64, act: bool) -> Self {
        Focus { conv: Conv::init(p, c1 * 4, c2, k, s, None, g, 1, act) }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        // x[..., ::2, ::2] 等 4 份
        let w = x.size()[3];
        let h = x.size()[2];
        let a = x.narrow(3, 0, (w + 1) / 2).narrow(2, 0, (h + 1) / 2);
        let a = stride2_slice(&a, 3, 0);
        let a = stride2_slice(&a, 2, 0);
        let b = stride2_slice(x, 3, 1);
        let b = stride2_slice(&b.narrow(2, 0, (h + 1) / 2), 2, 0);
        let c = stride2_slice(x, 3, 0);
        let c = stride2_slice(&c.narrow(2, 1, h / 2), 2, 0);
        let d = stride2_slice(x, 3, 1);
        let d = stride2_slice(&d.narrow(2, 1, h / 2), 2, 0);
        let cat = Tensor::cat(&[&a, &b, &c, &d], 1);
        self.conv.forward(&cat, train)
    }
}

/// 按 stride 2 抽取切片（x[..., start::2, :] 已 narrow 后调用）。
fn stride2_slice(x: &Tensor, dim: usize, start: i64) -> Tensor {
    let n = x.size()[dim];
    let cnt = (n - start + 1) / 2;
    let idx = Tensor::arange_start_step(start, n, 2, (Kind::Int64, x.device()));
    x.index_select(dim as i64, &idx).narrow(dim as i64, 0, cnt)
}

/// `GhostConv`：廉价卷积对。
pub struct GhostConv {
    pub cv1: Conv,
    pub cv2: Conv,
}

impl GhostConv {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64, g: i64, act: bool) -> Self {
        let c_ = c2 / 2;
        GhostConv {
            cv1: Conv::init(&p.child("cv1"), c1, c_, k, s, None, g, 1, act),
            cv2: Conv::init(&p.child("cv2"), c_, c_, 5, 1, None, c_, 1, act),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.cv1.forward(x, train);
        Tensor::cat(&[&y, &self.cv2.forward(&y, train)], 1)
    }
}

/// `LightConv`：1x1 Conv + DWConv（HGNetV2）。
pub struct LightConv {
    pub conv1: Conv,
    pub conv2: Conv,
}

impl LightConv {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64) -> Self {
        LightConv {
            conv1: Conv::init(&p.child("conv1"), c1, c2, 1, 1, None, 1, 1, false),
            conv2: dwconv_init(&p.child("conv2"), c2, c2, k, 1, 1, true),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.conv1.forward(x, train);
        self.conv2.forward(&y, train)
    }
}

/// `RepConv`（RT-DETR/v9 用）：conv1(3x3) + conv2(1x1) [+ bn 恒等]。
pub struct RepConv {
    pub conv1: Conv, // act=False
    pub conv2: Conv, // act=False
    pub bn: Option<BatchNorm>,
    pub act: bool,
}

impl RepConv {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64, g: i64, d: i64, act: bool, bn_identity: bool) -> Self {
        assert_eq!(k, 3);
        let mut conv1 = Conv::init(&p.child("conv1"), c1, c2, 3, s, Some(1), g, d, false);
        conv1.conv.pad_h = 1;
        conv1.conv.pad_w = 1;
        let mut conv2 = Conv::init(&p.child("conv2"), c1, c2, 1, s, None, g, d, false);
        conv2.conv.pad_h = 0;
        conv2.conv.pad_w = 0;
        let bn = if bn_identity && c2 == c1 && s == 1 { Some(BatchNorm::init(p, c1)) } else { None };
        RepConv { conv1, conv2, bn, act }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = self.conv1.forward(x, train) + self.conv2.forward(x, train);
        if let Some(bn) = &self.bn {
            y = y + bn.forward(x, train);
        }
        if self.act {
            silu(&y)
        } else {
            y
        }
    }
}
