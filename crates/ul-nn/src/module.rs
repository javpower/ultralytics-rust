//! VarStore 路径辅助层。
//!
//! tch 的 VarStore 禁止参数名含 `.`，因此路径用 `/` 分隔；ultralytics state_dict
//! 键全部用 `.`（如 `model.12.cv3.0.1.conv.weight`）。两边一一对应：
//! 加载 .pt 时把键的 `.` 替换为 `/` 即可按名对位（见 pt.rs）。

use tch::nn::VarStore;
use tch::{Kind, Tensor};

/// 参数初始化方式。
#[derive(Clone, Copy, Debug)]
pub enum Init {
    Const(f64),
    /// U(−bound, +bound)（等于 torch kaiming_uniform_(a=√5) 的卷积默认）。
    UniformSigned(f64),
    Normal(f64, f64),
}

impl Init {
    fn to_tch(self) -> tch::nn::Init {
        match self {
            Init::Const(c) => tch::nn::Init::Const(c),
            Init::UniformSigned(b) => tch::nn::Init::Uniform { lo: -b, up: b },
            Init::Normal(m, s) => tch::nn::Init::Randn { mean: m, stdev: s },
        }
    }
}

/// 变量路径（点分），绑定一个 VarStore（Arc 共享）。
#[derive(Clone)]
pub struct VarPath {
    vs: std::sync::Arc<VarStore>,
    path: String,
}

impl VarPath {
    pub fn root(vs: &std::sync::Arc<VarStore>) -> Self {
        VarPath { vs: std::sync::Arc::clone(vs), path: String::new() }
    }

    pub fn child(&self, name: &str) -> VarPath {
        let path = if self.path.is_empty() { name.to_string() } else { format!("{}/{}", self.path, name) };
        VarPath { vs: self.vs.clone(), path }
    }

    /// 层索引子路径：`p.idx(i)` == `p.child("0")`。
    pub fn idx(&self, i: usize) -> VarPath {
        self.child(&i.to_string())
    }

    /// VarStore 内部全名（/ 分隔；叶子名中的 `.` 也转为 `/`，
    /// 如 "conv.weight" → "…/conv/weight"）。
    pub fn full_name(&self, name: &str) -> String {
        let leaf = name.replace('.', "/");
        if self.path.is_empty() {
            leaf
        } else {
            format!("{}/{}", self.path, leaf)
        }
    }

    /// 对应 ultralytics state_dict 键（. 分隔）。
    pub fn pt_name(&self, name: &str) -> String {
        self.full_name(name).replace('/', ".")
    }

    /// 声明一个可训练 float 变量。
    pub fn var(&self, name: &str, dims: &[i64], init: Init) -> Tensor {
        self.vs.as_ref().root().var(self.full_name(name).as_str(), dims, init.to_tch())
    }

    /// 声明变量后关闭梯度（BN running 统计、DFL 固定权重）。
    pub fn var_nograd(&self, name: &str, dims: &[i64], init: Init) -> Tensor {
        let t = self.var(name, dims, init);
        t.set_requires_grad(false);
        t
    }

    pub fn var_1d(&self, name: &str, n: i64, init: Init) -> Tensor {
        self.var(name, &[n], init)
    }

    pub fn var_1d_nograd(&self, name: &str, n: i64, init: Init) -> Tensor {
        self.var_nograd(name, &[n], init)
    }

    /// 卷积权重 (out, in/groups, k, k)。
    pub fn var_conv_weight(&self, name: &str, out: i64, in_g: i64, k: i64, bound: f64) -> Tensor {
        self.var(name, &[out, in_g, k, k], Init::UniformSigned(bound))
    }

    /// 转置卷积权重 (in, out/groups, k, k)。
    pub fn var_conv_transpose_weight(&self, name: &str, c_in: i64, c_out: i64, k: i64, bound: f64) -> Tensor {
        self.var(name, &[c_in, c_out, k, k], Init::UniformSigned(bound))
    }

    /// Linear 权重 (out, in)：name 形如 "linear.weight"，ultralytics
    /// initialize_weights 的 xavier*gain(0.01) + 零偏置（bias 名由 weight 推导）。
    pub fn var_linear(&self, name: &str, out: i64, c_in: i64) -> (Tensor, Tensor) {
        let bound = 0.01 * (6.0 / (c_in + out) as f64).sqrt();
        let w = self.var(name, &[out, c_in], Init::UniformSigned(bound));
        let bias_name = name.replace("weight", "bias");
        let b = self.var(bias_name.as_str(), &[out], Init::Const(0.0));
        (w, b)
    }
}
