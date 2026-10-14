//! 模块动物园（block 层）：与 ultralytics/nn/modules/block.py 一一对应。
//!
//! 命名铁律：每个子模块的属性名（cv1/cv2/m/attn/ffn…）与 Python 版逐字一致，
//! 保证官方 .pt state_dict 可以按键名直接加载。

use tch::Tensor;

use crate::attention::PsaBlock;
use crate::conv::{dwconv_init, silu, BatchNorm, Conv, LightConv, RawConv};
use crate::module::{Init, VarPath};

// ---------------- DFL ----------------

/// Distribution Focal Loss 积分模块：固定 arange 权重的 1x1 卷积。
pub struct Dfl {
    pub conv: Tensor, // (1, c1, 1, 1) 固定 arange
    pub c1: i64,
}

impl Dfl {
    pub fn new(p: &VarPath, c1: i64) -> Self {
        let conv = p.var_nograd("conv.weight", &[1, c1, 1, 1], Init::Const(0.0));
        tch::no_grad(|| {
            let arange = Tensor::arange(c1, (tch::Kind::Float, conv.device())).reshape([1, c1, 1, 1]);
            let mut conv = conv.shallow_clone();
            let _ = conv.copy_(&arange);
        });
        let conv = conv;
        Dfl { conv, c1 }
    }

    /// (B, 4·c1, A) → (B, 4, A)：softmax 加权积分。
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let b = x.size()[0];
        let a = x.size()[2];
        let x = x.reshape([b, 4, self.c1, a]).transpose(2, 1).softmax(1, tch::Kind::Float);
        let y = x.f_conv2d(&self.conv, None::<&Tensor>, &[1, 1], &[0, 0], &[1, 1], 1).unwrap();
        y.reshape([b, 4, a])
    }
}

// ---------------- 通用内块 ----------------

/// C2f/C3 系列内部块的统一枚举（覆盖 Bottleneck / C3k / PSA / CIB 组合）。
pub enum InnerBlock {
    Bottleneck(Bottleneck),
    C3k(C3k),
    /// C3k2 attn 分支：Sequential(Bottleneck, PSABlock)。
    BottleneckPsa(Bottleneck, PsaBlock),
    /// C2fPSA：直接 PSABlock。
    Psa(PsaBlock),
    Cib(Cib),
}

impl InnerBlock {
    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        match self {
            InnerBlock::Bottleneck(b) => b.forward(x, train),
            InnerBlock::C3k(b) => b.forward(x, train),
            InnerBlock::BottleneckPsa(b, p) => p.forward(&b.forward(x, train), train),
            InnerBlock::Psa(p) => p.forward(x, train),
            InnerBlock::Cib(b) => b.forward(x, train),
        }
    }
}

// ---------------- 基础 bottleneck ----------------

/// 标准 Bottleneck：cv1 → cv2 [+shortcut]。
pub struct Bottleneck {
    pub cv1: Conv,
    pub cv2: Conv,
    pub add: bool,
}

impl Bottleneck {
    pub fn init(p: &VarPath, c1: i64, c2: i64, shortcut: bool, g: i64, k: (i64, i64), e: f64) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        Bottleneck {
            cv1: Conv::init(&p.child("cv1"), c1, c_, k.0, 1, None, g, 1, true),
            cv2: Conv::init(&p.child("cv2"), c_, c2, k.1, 1, None, g, 1, true),
            add: shortcut && c1 == c2,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.cv2.forward(&self.cv1.forward(x, train), train);
        if self.add {
            x + y
        } else {
            y
        }
    }
}

// ---------------- C2f 家族 ----------------

/// C2f：快速 CSP（chunk 两半 + n 个内块逐级拼接）。
pub struct C2f {
    pub c: i64,
    pub cv1: Conv,
    pub cv2: Conv,
    pub m: Vec<InnerBlock>,
}

impl C2f {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, shortcut: bool, g: i64, e: f64) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mut out = C2f {
            c: c_,
            cv1: Conv::init(&p.child("cv1"), c1, 2 * c_, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), (2 + n) * c_, c2, 1, 1, None, 1, 1, true),
            m: vec![],
        };
        let mp = p.child("m");
        for i in 0..n {
            out.m.push(InnerBlock::Bottleneck(Bottleneck::init(&mp.idx(i as usize), c_, c_, shortcut, g, (3, 3), 1.0)));
        }
        out
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let cv1 = self.cv1.forward(x, train);
        let mut y: Vec<Tensor> = cv1.chunk(2, 1);
        for b in &self.m {
            let last = y.last().unwrap().shallow_clone();
            y.push(b.forward(&last, train));
        }
        self.cv2.forward(&Tensor::cat(&y, 1), train)
    }
}

// ---------------- C3 家族 ----------------

/// C3/C3x 共用主体（m 内块的 kernel 由构造方决定）。
pub struct C3 {
    pub cv1: Conv,
    pub cv2: Conv,
    pub cv3: Conv,
    /// Sequential(Bottleneck × n)：每块独立参数 m.0..m.n-1。
    pub m: Vec<Bottleneck>,
}

impl C3 {
    /// `k=(1,3)` 标准 C3；`k=(1,3)` 交叉即 C3x 亦同形（C3x 用 (1,3)？官方 C3x 用 k=((1,3),(3,1))）。
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, shortcut: bool, g: i64, e: f64, k: (i64, i64)) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mp = p.child("m");
        C3 {
            cv1: Conv::init(&p.child("cv1"), c1, c_, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), c1, c_, 1, 1, None, 1, 1, true),
            cv3: Conv::init(&p.child("cv3"), 2 * c_, c2, 1, 1, None, 1, 1, true),
            m: (0..n).map(|i| Bottleneck::init(&mp.idx(i as usize), c_, c_, shortcut, g, k, 1.0)).collect(),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = self.cv1.forward(x, train);
        for b in &self.m {
            y = b.forward(&y, train);
        }
        let y2 = self.cv2.forward(x, train);
        self.cv3.forward(&Tensor::cat(&[&y, &y2], 1), train)
    }
}

/// C3k：C3 with 自定义 kernel（(k,k)）。
pub struct C3k {
    pub inner: C3,
}

impl C3k {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, shortcut: bool, g: i64, e: f64, k: i64) -> Self {
        C3k { inner: C3::init(p, c1, c2, n, shortcut, g, e, (k, k)) }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        self.inner.forward(x, train)
    }
}

/// C3f：cv1/cv2 并行 + n bottleneck 串（注意顺序：[cv2, cv1, m...]）。
pub struct C3f {
    pub cv1: Conv,
    pub cv2: Conv,
    pub cv3: Conv,
    pub m: Vec<Bottleneck>,
}

impl C3f {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, shortcut: bool, g: i64, e: f64) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mut out = C3f {
            cv1: Conv::init(&p.child("cv1"), c1, c_, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), c1, c_, 1, 1, None, 1, 1, true),
            cv3: Conv::init(&p.child("cv3"), (2 + n) * c_, c2, 1, 1, None, 1, 1, true),
            m: vec![],
        };
        let mp = p.child("m");
        for i in 0..n {
            out.m.push(Bottleneck::init(&mp.idx(i as usize), c_, c_, shortcut, g, (3, 3), 1.0));
        }
        out
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = vec![self.cv2.forward(x, train), self.cv1.forward(x, train)];
        for b in &self.m {
            let last = y.last().unwrap().shallow_clone();
            y.push(b.forward(&last, train));
        }
        self.cv3.forward(&Tensor::cat(&y, 1), train)
    }
}

/// C3k2 = C2f 变体（内块 Bottleneck / C3k / Bottleneck+PSABlock）。
pub struct C3k2 {
    pub base: C2f,
}

impl C3k2 {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, c3k: bool, e: f64, attn: bool, g: i64, shortcut: bool) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mut out = C3k2 {
            base: C2f {
                c: c_,
                cv1: Conv::init(&p.child("cv1"), c1, 2 * c_, 1, 1, None, 1, 1, true),
                cv2: Conv::init(&p.child("cv2"), (2 + n) * c_, c2, 1, 1, None, 1, 1, true),
                m: vec![],
            },
        };
        let mp = p.child("m");
        for i in 0..n {
            let pi = mp.idx(i as usize);
            let blk = if attn {
                let b = Bottleneck::init(&pi.child("0"), c_, c_, shortcut, g, (3, 3), 1.0);
                let psa = PsaBlock::init(&pi.child("1"), c_, 0.5, (c_ / 64).max(1));
                InnerBlock::BottleneckPsa(b, psa)
            } else if c3k {
                InnerBlock::C3k(C3k::init(&pi, c_, c_, 2, shortcut, g, 0.5, 3))
            } else {
                // 官方 C3k2：Bottleneck(self.c, self.c, shortcut, g) —— e 用默认 0.5
                InnerBlock::Bottleneck(Bottleneck::init(&pi, c_, c_, shortcut, g, (3, 3), 0.5))
            };
            out.base.m.push(blk);
        }
        out
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        self.base.forward(x, train)
    }
}

// ---------------- SPP 家族 ----------------

/// SPP：多 kernel maxpool 拼接。
pub struct Spp {
    pub cv1: Conv,
    pub cv2: Conv,
    pub k: Vec<i64>,
}

impl Spp {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: Vec<i64>) -> Self {
        let c_ = c1 / 2;
        Spp {
            cv1: Conv::init(&p.child("cv1"), c1, c_, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), c_ * (k.len() as i64 + 1), c2, 1, 1, None, 1, 1, true),
            k,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let x = self.cv1.forward(x, train);
        let mut outs = vec![x.shallow_clone()];
        for &k in &self.k {
            outs.push(x.f_max_pool2d([k, k], [1, 1], [k / 2, k / 2], [1, 1], false).unwrap());
        }
        self.cv2.forward(&Tensor::cat(&outs, 1), train)
    }
}

/// SPPF：串行 maxpool（k=5×3 等价 SPP(5,9,13)）。YOLO26 新签名带 n/shortcut；
/// legacy YAML（args<=3）时 cv1 需补激活（parse_model 置 legacy=true）。
pub struct Sppf {
    pub cv1: Conv,
    pub cv2: Conv,
    pub k: i64,
    pub n: i64,
    pub add: bool,
    pub legacy: bool,
}

impl Sppf {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, n: i64, shortcut: bool) -> Self {
        let c_ = c1 / 2;
        Sppf {
            cv1: Conv::init(&p.child("cv1"), c1, c_, 1, 1, None, 1, 1, false),
            cv2: Conv::init(&p.child("cv2"), c_ * (n + 1), c2, 1, 1, None, 1, 1, true),
            k,
            n,
            add: shortcut && c1 == c2,
            legacy: false,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let cv1 = self.cv1.forward(x, train);
        let cv1 = if self.legacy { silu(&cv1) } else { cv1 };
        let mut y = vec![cv1.shallow_clone()];
        for _ in 0..self.n {
            let last = y.last().unwrap();
            y.push(last.f_max_pool2d(&[self.k, self.k], &[1, 1], &[self.k / 2, self.k / 2], &[1, 1], false).unwrap());
        }
        let out = self.cv2.forward(&Tensor::cat(&y, 1), train);
        if self.add {
            x + out
        } else {
            out
        }
    }
}

// ---------------- ELAN（YOLOv9） ----------------

/// RepConv（训练态）：conv1(3x3)+conv2(1x1)，bn_identity 仅 RepConv(bn=True) 时存在。
pub struct RepConv {
    pub conv1: Conv,
    pub conv2: Conv,
    pub bn: Option<BatchNorm>,
    pub act: bool,
}

impl RepConv {
    pub fn init(p: &VarPath, c1: i64, c2: i64, g: i64, bn_identity: bool) -> Self {
        let conv1 = Conv::init(&p.child("conv1"), c1, c2, 3, 1, Some(1), g, 1, false);
        let conv2 = Conv::init(&p.child("conv2"), c1, c2, 1, 1, None, g, 1, false);
        let bn = if bn_identity && c2 == c1 { Some(BatchNorm::init(p, c1)) } else { None };
        RepConv { conv1, conv2, bn, act: true }
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

/// RepC3：C3 with RepConv 串（YOLOv9/RT-DETR）。
pub struct RepC3 {
    pub cv1: Conv,
    pub cv2: Conv,
    pub m: Vec<RepConv>,
    pub cv3: Option<Conv>,
}

impl RepC3 {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, e: f64) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mp = p.child("m");
        RepC3 {
            cv1: Conv::init(&p.child("cv1"), c1, c_, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), c1, c_, 1, 1, None, 1, 1, true),
            m: (0..n).map(|i| RepConv::init(&mp.idx(i as usize), c_, c_, 1, false)).collect(),
            cv3: if c_ != c2 { Some(Conv::init(&p.child("cv3"), c_, c2, 1, 1, None, 1, 1, true)) } else { None },
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = self.cv1.forward(x, train);
        for m in &self.m {
            y = m.forward(&y, train);
        }
        let y = y + self.cv2.forward(x, train);
        match &self.cv3 {
            Some(cv3) => cv3.forward(&y, train),
            None => y,
        }
    }
}

/// ELAN 分支：RepNCSPELAN4 用 Sequential(RepCSP, Conv)；ELAN1 用裸 Conv。
pub enum ElanBranch {
    Rep(RepC3, Conv),
    Plain(Conv),
}

impl ElanBranch {
    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        match self {
            ElanBranch::Rep(rc, c) => c.forward(&rc.forward(x, train), train),
            ElanBranch::Plain(c) => c.forward(x, train),
        }
    }
}

/// RepNCSPELAN4 / ELAN1：CSP-ELAN（YOLOv9 主体）。
pub struct RepNcspeLan4 {
    pub cv1: Conv,
    pub cv2: ElanBranch,
    pub cv3: ElanBranch,
    pub cv4: Conv,
}

impl RepNcspeLan4 {
    pub fn init(p: &VarPath, c1: i64, c2: i64, c3: i64, c4: i64, n: i64, elan1: bool) -> Self {
        let branch = |p: &VarPath, cin: i64| {
            if elan1 {
                ElanBranch::Plain(Conv::init(p, cin, c4, 3, 1, None, 1, 1, true))
            } else {
                ElanBranch::Rep(
                    RepC3::init(&p.child("0"), cin, c4, n, 1.0),
                    Conv::init(&p.child("1"), c4, c4, 3, 1, None, 1, 1, true),
                )
            }
        };
        RepNcspeLan4 {
            cv1: Conv::init(&p.child("cv1"), c1, c3, 1, 1, None, 1, 1, true),
            cv2: branch(&p.child("cv2"), c3 / 2),
            cv3: branch(&p.child("cv3"), c4),
            cv4: Conv::init(&p.child("cv4"), c3 + 2 * c4, c2, 1, 1, None, 1, 1, true),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let cv1 = self.cv1.forward(x, train);
        let halves = cv1.chunk(2, 1);
        let a = halves[0].shallow_clone();
        let y2 = self.cv2.forward(&halves[1], train);
        let y3 = self.cv3.forward(&y2, train);
        self.cv4.forward(&Tensor::cat(&[a, halves[1].shallow_clone(), y2, y3], 1), train)
    }
}

/// AConv：avg_pool(2,1) + Conv 3x3 s2。
pub struct AConv {
    pub cv1: Conv,
}

impl AConv {
    pub fn init(p: &VarPath, c1: i64, c2: i64) -> Self {
        AConv { cv1: Conv::init(&p.child("cv1"), c1, c2, 3, 2, Some(1), 1, 1, true) }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let x = x.f_avg_pool2d(&[2, 2], &[1, 1], &[0, 0], false, true, None).unwrap();
        self.cv1.forward(&x, train)
    }
}

/// ADown：avg_pool 后通道两半分别 conv/maxpool 处理。
pub struct ADown {
    pub cv1: Conv,
    pub cv2: Conv,
}

impl ADown {
    pub fn init(p: &VarPath, c1: i64, c2: i64) -> Self {
        let c_ = c2 / 2;
        ADown {
            cv1: Conv::init(&p.child("cv1"), c1 / 2, c_, 3, 2, Some(1), 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), c1 / 2, c_, 1, 1, Some(0), 1, 1, true),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let x = x.f_avg_pool2d(&[2, 2], &[1, 1], &[0, 0], false, true, None).unwrap();
        let halves = x.chunk(2, 1);
        let x1 = self.cv1.forward(&halves[0], train);
        let x2 = halves[1].f_max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false).unwrap();
        let x2 = self.cv2.forward(&x2, train);
        Tensor::cat(&[x1, x2], 1)
    }
}

/// SPPELAN：SPP-ELAN。
pub struct SppeLan {
    pub cv1: Conv,
    pub cv5: Conv,
    pub k: i64,
    pub c3: i64,
}

impl SppeLan {
    pub fn init(p: &VarPath, c1: i64, c2: i64, c3: i64, k: i64) -> Self {
        SppeLan {
            cv1: Conv::init(&p.child("cv1"), c1, c3, 1, 1, None, 1, 1, true),
            cv5: Conv::init(&p.child("cv5"), 4 * c3, c2, 1, 1, None, 1, 1, true),
            k,
            c3,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y0 = self.cv1.forward(x, train);
        let pool = |t: &Tensor| t.f_max_pool2d(self.k, &[1, 1], &[self.k / 2, self.k / 2], &[1, 1], false).unwrap();
        let y1 = pool(&y0);
        let y2 = pool(&y1);
        let y3 = pool(&y2);
        self.cv5.forward(&Tensor::cat(&[y0, y1, y2, y3], 1), train)
    }
}

// ---------------- YOLOv9 渐进式特征（CBLinear/CBFuse） ----------------

/// CBLinear：输出 sum(c2s) 通道，按 c2s 切分（bias=True）。
pub struct CbLinear {
    pub weight: Tensor,
    pub bias: Tensor,
    pub c2s: Vec<i64>,
    pub k: i64,
    pub s: i64,
    pub padding: i64,
    pub groups: i64,
}

impl CbLinear {
    pub fn init(p: &VarPath, c1: i64, c2s: Vec<i64>, k: i64, s: i64, g: i64) -> Self {
        let c_out: i64 = c2s.iter().sum();
        let bound = (1.0 / (c1 * k * k / g.max(1)) as f64).sqrt();
        let weight = p.var_conv_weight("conv.weight", c_out, c1 / g.max(1), k, bound);
        let bias = p.var_1d("conv.bias", c_out, Init::UniformSigned(1.0 / (c1 * k * k / g.max(1)) as f64));
        CbLinear { weight, bias, c2s, k, s, padding: crate::conv::autopad(k, None, 1), groups: g }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.f_conv2d(&self.weight, Some(&self.bias), &[self.s, self.s], &[self.padding, self.padding], &[1, 1], self.groups)
            .unwrap()
    }
}

/// CBFuse：把各 CBLinear 输出的指定分段插值到目标尺寸后求和（含最后的原图分支）。
/// `split_c2s[i]` 是第 i 个 CBLinear 输入的切分表（parse 时填入）。
pub struct CbFuse {
    pub idx: Vec<i64>,
    pub split_c2s: Vec<Vec<i64>>,
}

impl CbFuse {
    pub fn new(idx: Vec<i64>) -> Self {
        CbFuse { idx, split_c2s: vec![] }
    }

    /// xs：from 列表对应的输入，最后一个元素是目标特征图。
    pub fn forward(&self, xs: &[Tensor]) -> Tensor {
        let target_h = xs[xs.len() - 1].size()[2];
        let target_w = xs[xs.len() - 1].size()[3];
        let mut acc: Option<Tensor> = None;
        for (i, &sel) in self.idx.iter().enumerate() {
            let full = &xs[i];
            // 按 split_c2s[i] 前缀和取第 sel 段
            let starts: Vec<i64> = self.split_c2s[i]
                .iter()
                .scan(0i64, |st, &c| {
                    let s = *st;
                    *st += c;
                    Some(s)
                })
                .collect();
            let seg = full.narrow(1, starts[sel as usize], self.split_c2s[i][sel as usize]);
            let up = crate::ops::up_nearest(&seg, [target_h, target_w]);
            acc = Some(match acc {
                Some(a) => a + up,
                None => up,
            });
        }
        let target = &xs[xs.len() - 1];
        match acc {
            Some(a) => a + target.shallow_clone(),
            None => target.shallow_clone(),
        }
    }
}

// ---------------- RepVGGDW / CIB / C2fCIB ----------------

/// RepVGGDW：深度可分离重参数化块（7x7 + 3x3 并行）。
pub struct RepVggDw {
    pub conv: Conv,
    pub conv1: Conv,
    pub act: bool,
}

impl RepVggDw {
    pub fn init(p: &VarPath, ed: i64) -> Self {
        RepVggDw {
            conv: Conv::init(&p.child("conv"), ed, ed, 7, 1, Some(3), ed, 1, false),
            conv1: Conv::init(&p.child("conv1"), ed, ed, 3, 1, Some(1), ed, 1, false),
            act: true,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.conv.forward(x, train) + self.conv1.forward(x, train);
        if self.act {
            silu(&y)
        } else {
            y
        }
    }
}

/// CIB：紧凑倒置块（YOLOv10）。
pub struct Cib {
    pub seq: Vec<CibStage>,
    pub add: bool,
}

pub enum CibStage {
    Conv(Conv),
    DwBig(RepVggDw),
}

impl Cib {
    pub fn init(p: &VarPath, c1: i64, c2: i64, shortcut: bool, e: f64, lk: bool) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mut seq = vec![
            CibStage::Conv(dwconv_init(&p.idx(0), c1, c1, 3, 1, 1, true)),
            CibStage::Conv(Conv::init(&p.idx(1), c1, 2 * c_, 1, 1, None, 1, 1, true)),
        ];
        if lk {
            seq.push(CibStage::DwBig(RepVggDw::init(&p.idx(2), 2 * c_)));
        } else {
            seq.push(CibStage::Conv(dwconv_init(&p.idx(2), 2 * c_, 2 * c_, 3, 1, 1, true)));
        }
        seq.push(CibStage::Conv(Conv::init(&p.idx(3), 2 * c_, c2, 1, 1, None, 1, 1, true)));
        seq.push(CibStage::Conv(dwconv_init(&p.idx(4), c2, c2, 3, 1, 1, true)));
        Cib { seq, add: shortcut && c1 == c2 }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = x.shallow_clone();
        for s in &self.seq {
            y = match s {
                CibStage::Conv(c) => c.forward(&y, train),
                CibStage::DwBig(r) => r.forward(&y, train),
            };
        }
        if self.add {
            x + y
        } else {
            y
        }
    }
}

/// C2fCIB：C2f with CIB 内块。
pub struct C2fCib {
    pub base: C2f,
}

impl C2fCib {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, shortcut: bool, lk: bool, g: i64, e: f64) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mut out = C2fCib {
            base: C2f {
                c: c_,
                cv1: Conv::init(&p.child("cv1"), c1, 2 * c_, 1, 1, None, 1, 1, true),
                cv2: Conv::init(&p.child("cv2"), (2 + n) * c_, c2, 1, 1, None, 1, 1, true),
                m: vec![],
            },
        };
        let mp = p.child("m");
        for i in 0..n {
            out.base.m.push(InnerBlock::Cib(Cib::init(&mp.idx(i as usize), c_, c_, shortcut, 1.0, lk)));
        }
        out
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        self.base.forward(x, train)
    }
}

// ---------------- 其他基础块 ----------------

/// C1：单卷积 + 3x3 串。
pub struct C1 {
    pub cv1: Conv,
    pub m: Vec<Conv>,
}

impl C1 {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64) -> Self {
        let mp = p.child("m");
        C1 {
            cv1: Conv::init(&p.child("cv1"), c1, c2, 1, 1, None, 1, 1, true),
            m: (0..n).map(|i| Conv::init(&mp.idx(i as usize), c2, c2, 3, 1, None, 1, 1, true)).collect(),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.cv1.forward(x, train);
        let mut y = y.shallow_clone();
        for c in &self.m {
            y = c.forward(&y, train);
        }
        y + self.cv1.forward(x, train)
    }
}

/// C2：CSP bottleneck（chunk 两半，m 只走一半）。
pub struct C2 {
    pub c: i64,
    pub cv1: Conv,
    pub cv2: Conv,
    pub m: Vec<Bottleneck>,
}

impl C2 {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, shortcut: bool, g: i64, e: f64) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mp = p.child("m");
        C2 {
            c: c_,
            cv1: Conv::init(&p.child("cv1"), c1, 2 * c_, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), 2 * c_, c2, 1, 1, None, 1, 1, true),
            m: (0..n).map(|i| Bottleneck::init(&mp.idx(i as usize), c_, c_, shortcut, g, (3, 3), 1.0)).collect(),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let halves = self.cv1.forward(x, train).chunk(2, 1);
        let mut a = halves[0].shallow_clone();
        for b in &self.m {
            a = b.forward(&a, train);
        }
        self.cv2.forward(&Tensor::cat(&[a, halves[1].shallow_clone()], 1), train)
    }
}

/// GhostBottleneck。
pub struct GhostBottleneck {
    pub conv: Vec<GhostStage>,
    pub shortcut: GhostShortcut,
}

pub enum GhostStage {
    Ghost(crate::conv::GhostConv),
    Dw(Conv),
}

pub enum GhostShortcut {
    Identity,
    Proj(Conv, Conv),
}

impl GhostBottleneck {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64) -> Self {
        let c_ = c2 / 2;
        let mut conv = vec![
            GhostStage::Ghost(crate::conv::GhostConv::init(&p.idx(0), c1, c_, 1, 1, 1, true)),
        ];
        let shortcut = if s == 2 {
            conv.push(GhostStage::Dw(dwconv_init(&p.idx(1), c_, c_, k, s, 1, false)));
            GhostShortcut::Proj(
                dwconv_init(&p.child("shortcut").idx(0), c1, c1, k, s, 1, false),
                Conv::init(&p.child("shortcut").idx(1), c1, c2, 1, 1, None, 1, 1, false),
            )
        } else {
            GhostShortcut::Identity
        };
        conv.push(GhostStage::Ghost(crate::conv::GhostConv::init(&p.idx(if s == 2 { 2 } else { 1 }), c_, c2, 1, 1, 1, false)));
        GhostBottleneck { conv, shortcut }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = x.shallow_clone();
        for s in &self.conv {
            y = match s {
                GhostStage::Ghost(g) => g.forward(&y, train),
                GhostStage::Dw(c) => c.forward(&y, train),
            };
        }
        match &self.shortcut {
            GhostShortcut::Identity => y,
            GhostShortcut::Proj(dw, c) => {
                let sc = c.forward(&dw.forward(x, train), train);
                y + sc
            }
        }
    }
}

/// C3Ghost：C3 with GhostBottleneck。
pub struct C3Ghost {
    pub cv1: Conv,
    pub cv2: Conv,
    pub cv3: Conv,
    pub m: Vec<GhostBottleneck>,
}

impl C3Ghost {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, e: f64) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mp = p.child("m");
        C3Ghost {
            cv1: Conv::init(&p.child("cv1"), c1, c_, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), c1, c_, 1, 1, None, 1, 1, true),
            cv3: Conv::init(&p.child("cv3"), 2 * c_, c2, 1, 1, None, 1, 1, true),
            m: (0..n).map(|i| GhostBottleneck::init(&mp.idx(i as usize), c_, c_, 3, 1)).collect(),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = self.cv1.forward(x, train);
        for b in &self.m {
            y = b.forward(&y, train);
        }
        let y2 = self.cv2.forward(x, train);
        self.cv3.forward(&Tensor::cat(&[&y, &y2], 1), train)
    }
}

/// BottleneckCSP。
pub struct BottleneckCsp {
    pub cv1: Conv,
    pub cv2: RawConv,
    pub cv3: RawConv,
    pub cv4: Conv,
    pub bn: BatchNorm,
    pub m: Vec<Bottleneck>,
    pub c_: i64,
}

impl BottleneckCsp {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, shortcut: bool, g: i64, e: f64) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let bound = (1.0 / (c1 * 1 * 1) as f64).sqrt();
        let cv2_w = p.var_conv_weight("cv2.weight", c_, c1, 1, bound);
        let cv3_w = p.var_conv_weight("cv3.weight", c_, c_, 1, (1.0 / (c_ * 1 * 1) as f64).sqrt());
        BottleneckCsp {
            c_,
            cv1: Conv::init(&p.child("cv1"), c1, c_, 1, 1, None, 1, 1, true),
            cv2: RawConv { weight: cv2_w, bias: None, stride: 1, padding: 0, pad_h: 0, pad_w: 0, dilation: 1, groups: 1 },
            cv3: RawConv { weight: cv3_w, bias: None, stride: 1, padding: 0, pad_h: 0, pad_w: 0, dilation: 1, groups: 1 },
            cv4: Conv::init(&p.child("cv4"), 2 * c_, c2, 1, 1, None, 1, 1, true),
            bn: BatchNorm::init(p, 2 * c_),
            m: {
                let mp = p.child("m");
                (0..n).map(|i| Bottleneck::init(&mp.idx(i as usize), c_, c_, shortcut, g, (3, 3), 1.0)).collect()
            },
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y1 = self.cv1.forward(x, train);
        for b in &self.m {
            y1 = b.forward(&y1, train);
        }
        let y1 = self.cv3.forward(&y1);
        let y2 = self.cv2.forward(x);
        let cat = Tensor::cat(&[y1, y2], 1);
        let cat = silu(&self.bn.forward(&cat, train));
        self.cv4.forward(&cat, train)
    }
}

// ---------------- HGNetV2（RT-DETR 主干，暂供 PPHGNet 系 YAML 使用） ----------------

pub struct HgStem {
    pub stem1: Conv,
    pub stem2a: Conv,
    pub stem2b: Conv,
    pub stem3: Conv,
    pub stem4: Conv,
}

impl HgStem {
    pub fn init(p: &VarPath, c1: i64, cm: i64, c2: i64) -> Self {
        let conv_relu = |p: &VarPath, a: i64, b: i64, k: i64, s: i64, pad: i64| {
            let mut c = Conv::init(p, a, b, k, s, Some(pad), 1, 1, false);
            c.act = true; // ReLU —— forward 里我们用 relu 覆盖
            c
        };
        HgStem {
            stem1: conv_relu(&p.child("stem1"), c1, cm, 3, 2, 1),
            stem2a: conv_relu(&p.child("stem2a"), cm, cm / 2, 2, 1, 0),
            stem2b: conv_relu(&p.child("stem2b"), cm / 2, cm, 2, 1, 0),
            stem3: conv_relu(&p.child("stem3"), cm * 2, cm, 3, 2, 1),
            stem4: conv_relu(&p.child("stem4"), cm, c2, 1, 1, 0),
        }
    }

    /// HGStem 用 ReLU 而非 SiLU：forward 内手动处理（各 stem conv act=false 构造后 relu 外挂）。
    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let act = |t: Tensor| t.relu();
        let bn = |c: &Conv, t: &Tensor| c.bn.forward(&c.conv.forward(t), train);
        let x1 = act(bn(&self.stem1, x));
        let x1 = x1.f_constant_pad_nd(&[0, 1, 0, 1]).unwrap();
        let x2 = act(bn(&self.stem2a, &x1));
        let x2 = x2.f_constant_pad_nd(&[0, 1, 0, 1]).unwrap();
        let x2 = act(bn(&self.stem2b, &x2));
        let x1 = x1.f_max_pool2d(&[2, 2], &[1, 1], &[0, 0], &[1, 1], true).unwrap();
        let x = Tensor::cat(&[x1, x2], 1);
        let x = act(bn(&self.stem3, &x));
        act(bn(&self.stem4, &x))
    }
}

pub struct HgBlock {
    pub m: Vec<HgStage>,
    pub sc: Conv,
    pub ec: Conv,
    pub add: bool,
}

pub enum HgStage {
    Conv(Conv),
    Light(LightConv),
}

impl HgBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn init(p: &VarPath, c1: i64, cm: i64, c2: i64, k: i64, n: i64, lightconv: bool, shortcut: bool) -> Self {
        let mp = p.child("m");
        let m = (0..n)
            .map(|i| {
                let ci = if i == 0 { c1 } else { cm };
                if lightconv {
                    HgStage::Light(LightConv::init(&mp.idx(i as usize), ci, cm, k))
                } else {
                    HgStage::Conv(Conv::init(&mp.idx(i as usize), ci, cm, k, 1, None, 1, 1, true))
                }
            })
            .collect();
        HgBlock {
            m,
            sc: Conv::init(&p.child("sc"), c1 + n * cm, c2 / 2, 1, 1, None, 1, 1, true),
            ec: Conv::init(&p.child("ec"), c2 / 2, c2, 1, 1, None, 1, 1, true),
            add: shortcut && c1 == c2,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = vec![x.shallow_clone()];
        for m in &self.m {
            let last = y.last().unwrap().shallow_clone();
            y.push(match m {
                HgStage::Conv(c) => c.forward(&last, train),
                HgStage::Light(l) => l.forward(&last, train),
            });
        }
        let out = self.ec.forward(&self.sc.forward(&Tensor::cat(&y, 1), train), train);
        if self.add {
            x + out
        } else {
            out
        }
    }
}

// ---------------- ResNet（YOLO26-depth / torchvision 桥） ----------------

pub struct ResNetBlock {
    pub cv1: Conv,
    pub cv2: Conv,
    pub cv3: Conv,
    pub shortcut: Option<Conv>,
    pub c3: i64,
}

impl ResNetBlock {
    pub fn init(p: &VarPath, c1: i64, c2: i64, s: i64, e: i64) -> Self {
        let c3 = e * c2;
        let shortcut = if s != 1 || c1 != c3 {
            Some(Conv::init(&p.child("shortcut").idx(0), c1, c3, 1, s, None, 1, 1, false))
        } else {
            None
        };
        ResNetBlock {
            cv1: Conv::init(&p.child("cv1"), c1, c2, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), c2, c2, 3, s, Some(1), 1, 1, true),
            cv3: Conv::init(&p.child("cv3"), c2, c3, 1, 1, None, 1, 1, false),
            shortcut,
            c3,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.cv3.forward(&self.cv2.forward(&self.cv1.forward(x, train), train), train);
        let y = match &self.shortcut {
            Some(sc) => y + sc.forward(x, train),
            None => y + x.shallow_clone(),
        };
        y.relu()
    }
}

pub struct ResNetLayer {
    pub stem: Option<(Conv, i64)>, // (conv, maxpool) is_first
    pub blocks: Vec<ResNetBlock>,
}

impl ResNetLayer {
    pub fn init(p: &VarPath, c1: i64, c2: i64, s: i64, is_first: bool, n: i64, e: i64) -> Self {
        if is_first {
            ResNetLayer {
                stem: Some((Conv::init(&p.child("layer").idx(0), c1, c2, 7, 2, Some(3), 1, 1, true), 0)),
                blocks: vec![],
            }
        } else {
            let lp = p.child("layer");
            let mut blocks = vec![ResNetBlock::init(&lp.idx(0), c1, c2, s, e)];
            for i in 1..n {
                blocks.push(ResNetBlock::init(&lp.idx(i as usize), e * c2, c2, 1, e));
            }
            ResNetLayer { stem: None, blocks }
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        match &self.stem {
            Some((conv, _)) => {
                let y = conv.forward(x, train);
                y.f_max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false).unwrap()
            }
            None => {
                let mut y = x.shallow_clone();
                for b in &self.blocks {
                    y = b.forward(&y, train);
                }
                y
            }
        }
    }
}

// ---------------- SCDown ----------------

pub struct ScDown {
    pub cv1: Conv,
    pub cv2: Conv,
}

impl ScDown {
    pub fn init(p: &VarPath, c1: i64, c2: i64, k: i64, s: i64) -> Self {
        ScDown {
            cv1: Conv::init(&p.child("cv1"), c1, c2, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), c2, c2, k, s, None, c2, 1, false),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        self.cv2.forward(&self.cv1.forward(x, train), train)
    }
}


/// C3x：C3 with 交叉卷积 bottleneck（官方 k=((1,3),(3,1))）。
pub struct C3x {
    pub cv1: Conv,
    pub cv2: Conv,
    pub cv3: Conv,
    pub m: Vec<BottleneckRect>,
}

/// 非对称 kernel bottleneck（C3x 内部）。
pub struct BottleneckRect {
    pub cv1: Conv,
    pub cv2: Conv,
    pub add: bool,
}

impl BottleneckRect {
    pub fn init(p: &VarPath, c1: i64, c2: i64, shortcut: bool, g: i64) -> Self {
        let c_ = c2; // e=1.0
        let cv1 = Conv::init_rect(&p.child("cv1"), c1, c_, 1, 3, 1, (0, 1), g, true);
        let cv2 = Conv::init_rect(&p.child("cv2"), c_, c2, 3, 1, 1, (1, 0), g, true);
        BottleneckRect { cv1, cv2, add: shortcut && c1 == c2 }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.cv2.forward(&self.cv1.forward(x, train), train);
        if self.add {
            x + y
        } else {
            y
        }
    }
}

impl C3x {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, shortcut: bool, g: i64, e: f64) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        let mp = p.child("m");
        C3x {
            cv1: Conv::init(&p.child("cv1"), c1, c_, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), c1, c_, 1, 1, None, 1, 1, true),
            cv3: Conv::init(&p.child("cv3"), 2 * c_, c2, 1, 1, None, 1, 1, true),
            m: (0..n).map(|i| BottleneckRect::init(&mp.idx(i as usize), c_, c_, shortcut, g)).collect(),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = self.cv1.forward(x, train);
        for b in &self.m {
            y = b.forward(&y, train);
        }
        let y2 = self.cv2.forward(x, train);
        self.cv3.forward(&Tensor::cat(&[y, y2], 1), train)
    }
}
