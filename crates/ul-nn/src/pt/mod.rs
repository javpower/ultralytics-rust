//! 官方 `.pt` 权重加载：zip 容器 → pickle 对象树 → state_dict → 按名写入 VarStore。
//!
//! 零 Python 依赖（对比：aegis-vision 需要 Python 侧车转换 safetensors）。

pub mod pickle;

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use tch::{Kind, Tensor};

use ul_core::UlError;

use self::pickle::PValue;

/// state_dict 键 → 张量（CPU，float/long 原始 dtype）。
pub type StateDict = HashMap<String, Tensor>;

/// dtype 名 → Kind。
fn dtype_kind(name: &str) -> Option<Kind> {
    Some(match name {
        "FloatStorage" | "torch.float32" | "float32" => Kind::Float,
        "HalfStorage" | "torch.float16" | "float16" => Kind::Half,
        "BFloat16Storage" | "torch.bfloat16" => Kind::BFloat16,
        "DoubleStorage" | "torch.float64" => Kind::Double,
        "LongStorage" | "torch.int64" => Kind::Int64,
        "IntStorage" | "torch.int32" => Kind::Int,
        "ShortStorage" | "torch.int16" => Kind::Int16,
        "CharStorage" | "torch.int8" => Kind::Int8,
        "ByteStorage" | "torch.uint8" => Kind::Uint8,
        "BoolStorage" | "torch.bool" => Kind::Bool,
        _ => return None,
    })
}

fn dtype_size(kind: Kind) -> usize {
    match kind {
        Kind::Double => 8,
        Kind::Float | Kind::Int => 4,
        Kind::Half | Kind::BFloat16 | Kind::Int16 => 2,
        Kind::Int64 => 8,
        Kind::Int8 | Kind::Uint8 | Kind::Bool => 1,
        _ => 4,
    }
}

/// 加载 .pt：返回 (state_dict, checkpoint 元数据)。
pub fn load_pt(path: &Path) -> Result<(StateDict, PValue), UlError> {
    let zw = |e: zip::result::ZipError| UlError::Msg(format!("zip: {e}"));
    let file = std::fs::File::open(path)?;
    let mut zip = zip::ZipArchive::new(file).map_err(zw)?;
    // 找 data.pkl（可能带前缀目录，如 archive/data.pkl）
    let mut pkl_name = String::new();
    let mut names: Vec<String> = Vec::new();
    for i in 0..zip.len() {
        let n = zip.by_index(i).map_err(zw)?.name().to_string();
        names.push(n.clone());
        if n.ends_with("data.pkl") {
            pkl_name = n;
        }
    }
    if pkl_name.is_empty() {
        return Err(UlError::Msg(format!("{}: not a torch zip checkpoint (no data.pkl); entries={:?}", path.display(), &names[..names.len().min(8)])));
    }
    let prefix = pkl_name.strip_suffix("data.pkl").unwrap_or("").to_string();

    let mut pkl_file = zip.by_name(&pkl_name).map_err(zw)?;
    let mut pkl_bytes = Vec::new();
    pkl_file.read_to_end(&mut pkl_bytes)?;
    drop(pkl_file);

    let root = pickle::loads(&pkl_bytes)?;

    // 收集张量元数据（memo 化后同一 storage 只存一次）
    let mut tensors: Vec<(String, PValue)> = Vec::new(); // (pt key, Tensor repr)
    collect_tensors(&root, "", &mut tensors);

    // 读取 storage 数据并重建张量
    let mut storages: HashMap<String, (Kind, Vec<u8>)> = HashMap::new();
    let mut sd: StateDict = HashMap::new();
    for (key, tv) in &tensors {
        let (storage_key, offset, size, stride, dtype_name) = match tv {
            PValue::Tensor { storage_key, offset, size, stride, dtype } => {
                (storage_key.clone(), *offset, size.clone(), stride.clone(), dtype.clone())
            }
            _ => continue,
        };
        let kind = dtype_name
            .as_deref()
            .and_then(dtype_kind)
            .unwrap_or(Kind::Float);
        if !storages.contains_key(&storage_key) {
            let data_name = format!("{prefix}data/{storage_key}");
            let mut f = zip
                .by_name(&data_name)
                .map_err(|_| UlError::Msg(format!("checkpoint missing storage data '{data_name}'")))?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            storages.insert(storage_key.clone(), (kind, buf));
        }
        let (_k, bytes) = &storages[&storage_key];
        let tensor = rebuild_tensor(kind, bytes, offset, &size, &stride)?;
        sd.insert(key.clone(), tensor);
    }

    // checkpoint 顶层元数据（dict）
    let meta = match &root {
        PValue::Dict(_) => root.clone(),
        _ => PValue::Dict(vec![]),
    };
    Ok((sd, meta))
}

/// 从对象树收集 (键名, 张量)。
fn collect_tensors(node: &PValue, prefix: &str, out: &mut Vec<(String, PValue)>) {
    match node {
        PValue::Object { attrs, .. } => {
            for (k, v) in attrs {
                let key = k.as_str().unwrap_or("");
                match key {
                    "_parameters" | "_buffers" => {
                        if let PValue::Dict(pairs) = v {
                            for (pk, pv) in pairs {
                                let name = pk.as_str().unwrap_or("");
                                if name.is_empty() {
                                    continue;
                                }
                                let full = if prefix.is_empty() { name.to_string() } else { format!("{prefix}.{name}") };
                                if matches!(pv, PValue::Tensor { .. }) {
                                    out.push((full, pv.clone()));
                                } else if matches!(pv, PValue::None) {
                                    // None 参数跳过
                                }
                            }
                        }
                    }
                    "_modules" => {
                        if let PValue::Dict(pairs) = v {
                            for (pk, pv) in pairs {
                                let name = pk.as_str().unwrap_or("");
                                if name.is_empty() {
                                    continue;
                                }
                                let full = if prefix.is_empty() { name.to_string() } else { format!("{prefix}.{name}") };
                                collect_tensors(pv, &full, out);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // Sequential / ModuleList 表现为 List[Object]
        PValue::List(items) => {
            for (i, item) in items.iter().enumerate() {
                let full = if prefix.is_empty() { i.to_string() } else { format!("{prefix}.{i}") };
                collect_tensors(item, &full, out);
            }
        }
        // checkpoint 顶层 dict / 其它 dict 容器：递归值（键名不并入张量前缀）
        PValue::Dict(pairs) => {
            for (_k, v) in pairs {
                collect_tensors(v, prefix, out);
            }
        }
        _ => {}
    }
}

/// 从 storage 字节重建张量视图（as_strided 语义）。
fn rebuild_tensor(kind: Kind, bytes: &[u8], offset: i64, size: &[i64], stride: &[i64]) -> Result<Tensor, UlError> {
    let dsize = dtype_size(kind);
    let total: i64 = size.iter().product::<i64>().max(0);
    let max_off: i64 = if stride.is_empty() || total == 0 {
        total
    } else {
        let mut m = 0i64;
        for (sz, st) in size.iter().zip(stride.iter()) {
            if *sz > 0 {
                m += (sz - 1) * st;
            }
        }
        m + 1
    };
    let need = (offset.max(0) + max_off) as usize * dsize;
    if bytes.len() < need {
        return Err(UlError::Msg(format!(
            "storage too small: have {} bytes, need {need}",
            bytes.len()
        )));
    }
    let total_prod: i64 = size.iter().product::<i64>();
    if total_prod == 0 {
        // 空张量（如 Detect.anchors 缓存 buffer）
        return Ok(Tensor::empty(size, (kind, tch::Device::Cpu)));
    }
    let data = &bytes[offset.max(0) as usize * dsize..];
    // 以 1D 连续张量读入，再 as_strided 成目标视图
    let n = (max_off.max(1)) as usize;
    let flat = Tensor::from_data_size(&data[..n * dsize], &[n as i64], kind);
    if size.is_empty() {
        return Ok(flat); // 标量按 1 元素处理（加载时只关心数据）
    }
    // 连续步长直接 reshape；否则 as_strided
    let contiguous: Vec<i64> = {
        let mut s = vec![1i64; size.len()];
        for i in (0..size.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * size[i + 1];
        }
        s
    };
    let view = if stride == contiguous {
        flat.reshape(size)
    } else {
        flat.f_as_strided(size, stride, 0)?
    };
    Ok(view.contiguous())
}

/// 加载结果报告。
#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    pub loaded: usize,
    pub shape_mismatch: Vec<String>,
    pub missing: Vec<String>,    // 模型有、权重无
    pub unexpected: Vec<String>, // 权重有、模型无
}

impl LoadReport {
    pub fn summary(&self) -> String {
        format!(
            "loaded={}, shape_mismatch={}, missing={}, unexpected={}",
            self.loaded,
            self.shape_mismatch.len(),
            self.missing.len(),
            self.unexpected.len()
        )
    }
}

/// 把 state_dict 按名写入模型的 VarStore（float 模型自动降/升精度）。
pub fn load_state_dict_into(vs: &tch::nn::VarStore, sd: &StateDict, strict: bool) -> Result<LoadReport, UlError> {
    let mut report = LoadReport::default();
    // 噪声键：BN num_batches_tracked、Detect 缓存 buffer —— 不参与对位
    let noisy = |k: &str| k.ends_with("num_batches_tracked") || k.ends_with(".anchors") || k.ends_with(".strides");
    let sd: StateDict = sd
        .iter()
        .filter(|(k, _)| !noisy(k))
        .map(|(k, v)| (k.clone(), v.shallow_clone()))
        .collect();
    let vars = vs.variables();
    let mut pt_used = vec![false; sd.len()];
    let keys: Vec<&String> = sd.keys().collect();
    for (name, var) in &vars {
        let pt_key = name.replace('/', ".");
        match sd.get(&pt_key) {
            Some(w) => {
                let w = w.shallow_clone();
                if w.size() == var.size() {
                    tch::no_grad(|| {
                        let mut v = var.shallow_clone();
                        let _ = v.copy_(&w.to_device(var.device()).to_kind(var.kind()));
                    });
                    report.loaded += 1;
                    if let Some(i) = keys.iter().position(|k| *k == &pt_key) {
                        pt_used[i] = true;
                    }
                } else {
                    report.shape_mismatch.push(format!(
                        "{pt_key}: model {:?} vs ckpt {:?}",
                        var.size(),
                        w.size()
                    ));
                }
            }
            None => report.missing.push(pt_key),
        }
    }
    for (i, k) in keys.iter().enumerate() {
        if !pt_used[i] {
            report.unexpected.push((*k).clone());
        }
    }
    if strict && (!report.missing.is_empty() || !report.shape_mismatch.is_empty()) {
        return Err(UlError::Msg(format!(
            "strict load failed: {}",
            report.summary()
        )));
    }
    Ok(report)
}

/// 从 checkpoint 元数据里取训练/任务相关信息（guess 用）。
pub fn ckpt_get_str(meta: &PValue, key: &str) -> Option<String> {
    match meta.get(key) {
        Some(PValue::Str(s)) => Some(s.clone()),
        _ => None,
    }
}
