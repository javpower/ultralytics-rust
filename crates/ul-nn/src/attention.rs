//! 注意力族模块（Attention / PSA / C2PSA / Area-Attention A2C2f）。

use tch::Tensor;

use crate::block::{Bottleneck, C3k, InnerBlock};
use crate::conv::{dwconv_init, Conv};
use crate::module::VarPath;

/// 多头自注意力（YOLO PSA 系）。
pub struct Attention {
    pub num_heads: i64,
    pub head_dim: i64,
    pub key_dim: i64,
    pub scale: f64,
    pub qkv: Conv,  // act=False
    pub proj: Conv, // act=False
    pub pe: Conv,   // DW 3x3 act=False
}

impl Attention {
    pub fn init(p: &VarPath, dim: i64, num_heads: i64, attn_ratio: f64) -> Self {
        let head_dim = dim / num_heads;
        let key_dim = (head_dim as f64 * attn_ratio) as i64;
        let scale = (key_dim as f64).powf(-0.5);
        let h = dim + key_dim * num_heads * 2;
        Attention {
            num_heads,
            head_dim,
            key_dim,
            scale,
            qkv: Conv::init(&p.child("qkv"), dim, h, 1, 1, None, 1, 1, false),
            proj: Conv::init(&p.child("proj"), dim, dim, 1, 1, None, 1, 1, false),
            pe: dwconv_init(&p.child("pe"), dim, dim, 3, 1, 1, false),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let b = x.size()[0];
        let c = x.size()[1];
        let h = x.size()[2];
        let w = x.size()[3];
        let n = h * w;
        let qkv = self.qkv.forward(x, train);
        // (B, nh, kd*2+hd, N)
        let qkv = qkv.reshape([b, self.num_heads, self.key_dim * 2 + self.head_dim, n]);
        let parts = qkv.split_with_sizes(&[self.key_dim, self.key_dim, self.head_dim], 2);
        let (q, k, v) = (&parts[0], &parts[1], &parts[2]);
        // attn = softmax((q*scale)^T @ k)
        let qs = q * self.scale;
        let attn = qs.transpose(-2, -1).matmul(k).softmax(-1, tch::Kind::Float);
        // x = v @ attn^T：(B,nh,hd,N) @ (B,nh,N,N) → (B,nh,hd,N)
        let y = v.matmul(&attn.transpose(-2, -1));
        let y = y.reshape([b, c, h, w]);
        let v_im = v.reshape([b, c, h, w]);
        let y = y + self.pe.forward(&v_im, train);
        self.proj.forward(&y, train)
    }
}

/// PSABlock：Attention + FFN（带 shortcut）。
pub struct PsaBlock {
    pub attn: Attention,
    pub ffn: (Conv, Conv),
    pub add: bool,
}

impl PsaBlock {
    pub fn init(p: &VarPath, c: i64, attn_ratio: f64, num_heads: i64) -> Self {
        PsaBlock {
            attn: Attention::init(&p.child("attn"), c, num_heads, attn_ratio),
            ffn: (
                Conv::init(&p.child("ffn").idx(0), c, c * 2, 1, 1, None, 1, 1, true),
                Conv::init(&p.child("ffn").idx(1), c * 2, c, 1, 1, None, 1, 1, false),
            ),
            add: true,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let y = self.attn.forward(x, train);
        let x = if self.add { x + y } else { y };
        let ffn = self.ffn.1.forward(&self.ffn.0.forward(&x, train), train);
        if self.add {
            x + ffn
        } else {
            ffn
        }
    }
}

/// PSA：Position-Sensitive Attention（split 一半过 attn+ffn）。
pub struct Psa {
    pub c: i64,
    pub cv1: Conv,
    pub cv2: Conv,
    pub attn: Attention,
    pub ffn: (Conv, Conv),
}

impl Psa {
    pub fn init(p: &VarPath, c1: i64, c2: i64, e: f64) -> Self {
        assert_eq!(c1, c2);
        let c = (c1 as f64 * e) as i64;
        Psa {
            c,
            cv1: Conv::init(&p.child("cv1"), c1, 2 * c, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), 2 * c, c1, 1, 1, None, 1, 1, true),
            attn: Attention::init(&p.child("attn"), c, (c / 64).max(1), 0.5),
            ffn: (
                Conv::init(&p.child("ffn").idx(0), c, c * 2, 1, 1, None, 1, 1, true),
                Conv::init(&p.child("ffn").idx(1), c * 2, c, 1, 1, None, 1, 1, false),
            ),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let sp = self.cv1.forward(x, train).split_with_sizes(&[self.c, self.c], 1);
        let b0 = sp[1].shallow_clone();
        let b = &b0 + self.attn.forward(&b0, train);
        let b1 = self.ffn.1.forward(&self.ffn.0.forward(&b, train), train);
        let b_out = &b + b1;
        self.cv2.forward(&Tensor::cat(&[sp[0].shallow_clone(), b_out], 1), train)
    }
}

/// C2PSA：CSP + n 个 PSABlock。
pub struct C2Psa {
    pub c: i64,
    pub cv1: Conv,
    pub cv2: Conv,
    pub m: Vec<PsaBlock>,
}

impl C2Psa {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, e: f64) -> Self {
        assert_eq!(c1, c2);
        let c = (c1 as f64 * e) as i64;
        let mp = p.child("m");
        C2Psa {
            c,
            cv1: Conv::init(&p.child("cv1"), c1, 2 * c, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), 2 * c, c1, 1, 1, None, 1, 1, true),
            m: (0..n).map(|i| PsaBlock::init(&mp.idx(i as usize), c, 0.5, (c / 64).max(1))).collect(),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let sp = self.cv1.forward(x, train).split_with_sizes(&[self.c, self.c], 1);
        let mut b = sp[1].shallow_clone();
        for blk in &self.m {
            b = blk.forward(&b, train);
        }
        self.cv2.forward(&Tensor::cat(&[sp[0].shallow_clone(), b], 1), train)
    }
}

/// C2fPSA：C2f with PSABlock 内块。
pub struct C2fPsa {
    pub base: crate::block::C2f,
}

impl C2fPsa {
    pub fn init(p: &VarPath, c1: i64, c2: i64, n: i64, e: f64) -> Self {
        assert_eq!(c1, c2);
        let c_ = (c2 as f64 * e) as i64;
        let mut out = C2fPsa {
            base: crate::block::C2f {
                c: c_,
                cv1: Conv::init(&p.child("cv1"), c1, 2 * c_, 1, 1, None, 1, 1, true),
                cv2: Conv::init(&p.child("cv2"), (2 + n) * c_, c2, 1, 1, None, 1, 1, true),
                m: vec![],
            },
        };
        let mp = p.child("m");
        for i in 0..n {
            out.base.m.push(InnerBlock::Psa(PsaBlock::init(&mp.idx(i as usize), c_, 0.5, (c_ / 64).max(1))));
        }
        out
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        self.base.forward(x, train)
    }
}

// ---------------- Area-Attention（YOLOv12） ----------------

/// AAttn：Area-attention。
pub struct AAttn {
    pub area: i64,
    pub num_heads: i64,
    pub head_dim: i64,
    pub all_head_dim: i64,
    pub qkv: Conv,
    pub proj: Conv,
    pub pe: Conv,
}

impl AAttn {
    pub fn init(p: &VarPath, dim: i64, num_heads: i64, area: i64) -> Self {
        let head_dim = dim / num_heads;
        let all_head_dim = head_dim * num_heads;
        AAttn {
            area,
            num_heads,
            head_dim,
            all_head_dim,
            qkv: Conv::init(&p.child("qkv"), dim, all_head_dim * 3, 1, 1, None, 1, 1, false),
            proj: Conv::init(&p.child("proj"), all_head_dim, dim, 1, 1, None, 1, 1, false),
            pe: dwconv_init(&p.child("pe"), all_head_dim, all_head_dim, 7, 1, 1, false),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let bs = x.size()[0];
        let h = x.size()[2];
        let w = x.size()[3];
        let n = h * w;
        // (B, N, 3*ahd)
        let mut qkv = self.qkv.forward(x, train).flatten(2, -1).transpose(1, 2);
        let mut b = bs;
        let mut n2 = n;
        if self.area > 1 {
            qkv = qkv.reshape([bs * self.area, n / self.area, self.all_head_dim * 3]);
            b = bs * self.area;
            n2 = n / self.area;
        }
        // (B, nh, hd*3, N) → split q/k/v (B, nh, hd, N)
        let qkv = qkv
            .reshape([b, n2, self.num_heads, self.head_dim * 3])
            .permute([0, 2, 3, 1]);
        let parts = qkv.split_with_sizes(&[self.head_dim, self.head_dim, self.head_dim], 2);
        let (q, k, v) = (&parts[0], &parts[1], &parts[2]);
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q * scale).transpose(-2, -1).matmul(k).softmax(-1, tch::Kind::Float);
        // (B, nh, hd, N) → (B, N, nh, hd)：与 Python permute(0,3,1,2) 严格一致
        let mut y = v.matmul(&attn.transpose(-2, -1)).permute([0, 3, 1, 2]);
        let mut v2 = v.permute([0, 3, 1, 2]);
        if self.area > 1 {
            y = y.reshape([bs, n, self.all_head_dim]);
            v2 = v2.reshape([bs, n, self.all_head_dim]);
        }
        // (B, N, ahd) → (B, H, W, ahd) → (B, ahd, H, W)
        let y = y.reshape([bs, h, w, self.all_head_dim]).permute([0, 3, 1, 2]).contiguous();
        let v2 = v2.reshape([bs, h, w, self.all_head_dim]).permute([0, 3, 1, 2]).contiguous();
        let y = y + self.pe.forward(&v2, train);
        self.proj.forward(&y, train)
    }
}

/// ABlock：AAttn + MLP。
pub struct ABlock {
    pub attn: AAttn,
    pub mlp: (Conv, Conv),
}

impl ABlock {
    pub fn init(p: &VarPath, dim: i64, num_heads: i64, mlp_ratio: f64, area: i64) -> Self {
        let mlp_hidden = (dim as f64 * mlp_ratio) as i64;
        ABlock {
            attn: AAttn::init(&p.child("attn"), dim, num_heads, area),
            mlp: (
                Conv::init(&p.child("mlp").idx(0), dim, mlp_hidden, 1, 1, None, 1, 1, true),
                Conv::init(&p.child("mlp").idx(1), mlp_hidden, dim, 1, 1, None, 1, 1, false),
            ),
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let x2 = x + self.attn.forward(x, train);
        &x2 + self.mlp.1.forward(&self.mlp.0.forward(&x2, train), train)
    }
}

/// A2C2f：Area-Attention C2f（YOLOv12 主体）。
pub struct A2C2f {
    pub cv1: Conv,
    pub cv2: Conv,
    pub gamma: Option<Tensor>,
    pub m: Vec<A2Inner>,
    pub c_: i64,
}

pub enum A2Inner {
    Blocks(Vec<ABlock>),
    C3k(C3k),
}

impl A2C2f {
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        p: &VarPath,
        c1: i64,
        c2: i64,
        n: i64,
        a2: bool,
        area: i64,
        residual: bool,
        mlp_ratio: f64,
        e: f64,
        g: i64,
        shortcut: bool,
    ) -> Self {
        let c_ = (c2 as f64 * e) as i64;
        assert!(c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32.");
        let mp = p.child("m");
        let m = (0..n)
            .map(|i| {
                if a2 {
                    let bp = mp.idx(i as usize);
                    A2Inner::Blocks(vec![
                        ABlock::init(&bp.idx(0), c_, c_ / 32, mlp_ratio, area),
                        ABlock::init(&bp.idx(1), c_, c_ / 32, mlp_ratio, area),
                    ])
                } else {
                    A2Inner::C3k(C3k::init(&mp.idx(i as usize), c_, c_, 2, shortcut, g, 0.5, 3))
                }
            })
            .collect();
        A2C2f {
            cv1: Conv::init(&p.child("cv1"), c1, c_, 1, 1, None, 1, 1, true),
            cv2: Conv::init(&p.child("cv2"), (1 + n) * c_, c2, 1, 1, None, 1, 1, true),
            gamma: if a2 && residual { Some(p.var_1d("gamma", c2, crate::module::Init::Const(0.01))) } else { None },
            m,
            c_,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut y = vec![self.cv1.forward(x, train)];
        for blk in &self.m {
            let last = y.last().unwrap().shallow_clone();
            y.push(match blk {
                A2Inner::Blocks(blocks) => {
                    let mut cur = last;
                    for b in blocks {
                        cur = b.forward(&cur, train);
                    }
                    cur
                }
                A2Inner::C3k(c) => c.forward(&last, train),
            });
        }
        let out = self.cv2.forward(&Tensor::cat(&y, 1), train);
        match &self.gamma {
            Some(g) => {
                let g = g.reshape([1, g.size()[0], 1, 1]);
                x + g * out
            }
            None => out,
        }
    }
}

/// Bottleneck in C3（A2C2f C3k 分支引用用）。
pub use crate::block::Bottleneck as A2Bottleneck;
