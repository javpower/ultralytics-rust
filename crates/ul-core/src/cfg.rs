//! 模型 YAML 与数据集 YAML 解析。
//!
//! 模型定义文件直接嵌入官方 `ultralytics/cfg/models/**/*.yaml`（见 build.rs），
//! 由 parse_model 消费——与 Python 版共用同一份架构定义，保证全系列一致。

use serde_yaml::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// YAML 标量/列表的中间表示（模型层 args 里会出现 int/float/bool/str/嵌套 list）。
#[derive(Debug, Clone, PartialEq)]
pub enum YamlVal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    List(Vec<i64>),
    NestedList(Vec<Vec<i64>>),
    None,
}

impl YamlVal {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            YamlVal::Int(i) => Some(*i),
            YamlVal::Float(f) => Some(*f as i64),
            YamlVal::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            YamlVal::Int(i) => Some(*i as f64),
            YamlVal::Float(f) => Some(*f),
            YamlVal::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            YamlVal::Bool(b) => Some(*b),
            YamlVal::Int(i) => Some(*i != 0),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            YamlVal::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_list(&self) -> Option<&[i64]> {
        match self {
            YamlVal::List(l) => Some(l),
            _ => None,
        }
    }
    pub fn is_list(&self) -> bool {
        matches!(self, YamlVal::List(_))
    }
}

/// 模型 YAML 中的一层：`[from, repeats, module, args]`。
#[derive(Debug, Clone)]
pub struct LayerRow {
    /// from：-1 为上一层；正数为层索引；列表为多输入。
    pub from: Vec<i64>,
    pub repeats: i64,
    pub module: String,
    pub args: Vec<YamlVal>,
}

/// 完整模型配置（含按名字面推断的 scale）。
#[derive(Debug, Clone)]
pub struct ModelYaml {
    pub nc: i64,
    pub reg_max: i64,
    pub end2end: bool,
    pub activation: Option<String>,
    pub kpt_shape: Option<(i64, i64)>,
    pub depth_multiple: f64,
    pub width_multiple: f64,
    pub max_channels: f64,
    pub scale: String,
    pub backbone: Vec<LayerRow>,
    pub head: Vec<LayerRow>,
    pub yaml_file: String,
}

fn value_to_yamlval(v: &Value) -> YamlVal {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                YamlVal::Int(i)
            } else {
                YamlVal::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::Bool(b) => YamlVal::Bool(*b),
        Value::String(s) => YamlVal::Str(s.clone()),
        Value::Sequence(seq) => {
            if seq.iter().all(|x| x.as_sequence().is_some()) {
                YamlVal::NestedList(
                    seq.iter()
                        .map(|x| x.as_sequence().unwrap().iter().filter_map(|v| v.as_i64()).collect())
                        .collect(),
                )
            } else {
                YamlVal::List(seq.iter().filter_map(|x| x.as_i64()).collect())
            }
        }
        Value::Null => YamlVal::None,
        other => YamlVal::Str(format!("{other:?}")),
    }
}

fn parse_rows(seq: &[Value]) -> Vec<LayerRow> {
    seq.iter()
        .map(|row| {
            let r = row.as_sequence().expect("layer row must be a list [from, repeats, module, args]");
            let from_v = &r[0];
            let from = match from_v {
                Value::Sequence(s) => s.iter().filter_map(|x| x.as_i64()).collect(),
                Value::Number(n) => vec![n.as_i64().unwrap_or(-1)],
                _ => vec![-1],
            };
            let repeats = r.get(1).and_then(|x| x.as_i64()).unwrap_or(1);
            let module = r.get(2).and_then(|x| x.as_str()).unwrap_or("").to_string();
            let args = r
                .get(3)
                .and_then(|x| x.as_sequence())
                .map(|s| s.iter().map(value_to_yamlval).collect())
                .unwrap_or_default();
            LayerRow { from, repeats, module, args }
        })
        .collect()
}

/// 从 YAML 文本解析模型配置；`scale` 由调用方按文件名推断后传入（可为空）。
pub fn parse_model_yaml_text(text: &str, scale_hint: &str, yaml_file: &str) -> Result<ModelYaml, crate::UlError> {
    let v: Value = serde_yaml::from_str(text)?;
    let nc = v.get("nc").and_then(|x| x.as_i64()).unwrap_or(80);
    let reg_max = v.get("reg_max").and_then(|x| x.as_i64()).unwrap_or(16);
    let end2end = v.get("end2end").and_then(|x| x.as_bool()).unwrap_or(false);
    let activation = v.get("activation").and_then(|x| x.as_str()).map(|s| s.to_string());
    let kpt_shape = v.get("kpt_shape").and_then(|x| x.as_sequence()).and_then(|s| {
        Some((s.first()?.as_i64()?, s.get(1)?.as_i64()?))
    });
    let depth_multiple = v
        .get("depth_multiple")
        .and_then(|x| x.as_f64())
        .unwrap_or(1.0);
    let width_multiple = v
        .get("width_multiple")
        .and_then(|x| x.as_f64())
        .unwrap_or(1.0);

    let scales: Option<BTreeMap<String, [f64; 3]>> = v.get("scales").and_then(|s| {
        let m = s.as_mapping()?;
        let mut out = BTreeMap::new();
        for (k, val) in m {
            let name = k.as_str()?.to_string();
            let arr = val.as_sequence()?;
            let f = |i: usize| arr.get(i).and_then(|x| x.as_f64()).unwrap_or(1.0);
            out.insert(name, [f(0), f(1), f(2)]);
        }
        Some(out)
    });

    let (scale, depth_multiple, width_multiple, max_channels) = match (&scales, scale_hint) {
        (Some(scales), hint) if !hint.is_empty() && scales.contains_key(hint) => {
            let [d, w, m] = scales[hint];
            (hint.to_string(), d, w, m)
        }
        (Some(scales), _) => {
            let first = scales.keys().next().cloned().unwrap_or_default();
            log::warn!("no model scale passed. Assuming scale='{first}'.");
            let [d, w, m] = scales[&first];
            (first, d, w, m)
        }
        (None, _) => (String::new(), depth_multiple, width_multiple, f64::INFINITY),
    };

    let empty = vec![];
    let backbone = parse_rows(v.get("backbone").and_then(|x| x.as_sequence()).unwrap_or(&empty));
    let head = parse_rows(v.get("head").and_then(|x| x.as_sequence()).unwrap_or(&empty));

    Ok(ModelYaml {
        nc,
        reg_max,
        end2end,
        activation,
        kpt_shape,
        depth_multiple,
        width_multiple,
        max_channels,
        scale,
        backbone,
        head,
        yaml_file: yaml_file.to_string(),
    })
}

/// `yolo11n.yaml` → scale `n`（与 ultralytics guess_model_scale 同规则）。
pub fn guess_model_scale(stem: &str) -> String {
    let re = regex::Regex::new(r"yolo(e-)?[v]?\d+([nslmx])").unwrap();
    re.captures(stem).and_then(|c| c.get(2)).map(|m| m.as_str().to_string()).unwrap_or_default()
}

/// 嵌入的官方 cfg 文件索引（build.rs 生成）。
pub mod embedded {
    include!(concat!(env!("OUT_DIR"), "/cfg_index.rs"));
}

/// 解析模型 yaml 路径/名字 → ModelYaml。
///
/// 支持三种输入：显式路径（读磁盘）、内嵌注册表全名（`11/yolo11.yaml`）、
/// 惯用名（`yolo11n.yaml`，自动剥离 scale 字符并推断 scale）。
pub fn load_model_yaml(path_or_name: &str) -> Result<ModelYaml, crate::UlError> {
    let p = Path::new(path_or_name);
    if p.exists() {
        let text = std::fs::read_to_string(p)?;
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let scale = guess_model_scale(stem);
        return parse_model_yaml_text(&text, &scale, path_or_name);
    }
    // 惯用名 → 内嵌注册表
    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or(path_or_name);
    let stem = name.strip_suffix(".yaml").unwrap_or(name);
    let scale = guess_model_scale(stem);
    // 优先整名（v10/v9 的 per-scale 文件如 yolov10n.yaml 本身存在）
    if let Some(rel) = resolve_embedded_named(stem) {
        let text = embedded::get(&rel)
            .ok_or_else(|| crate::UlError::Msg(format!("embedded cfg '{rel}' missing")))?;
        return parse_model_yaml_text(text, &scale, &rel);
    }
    // 再剥 scale：yolo11n → yolo11
    let bare = strip_scale(stem, &scale);
    let rel = resolve_embedded(&bare).ok_or_else(|| {
        crate::UlError::Msg(format!(
            "model yaml '{path_or_name}' not found on disk or embedded registry"
        ))
    })?;
    let text = embedded::get(&rel)
        .ok_or_else(|| crate::UlError::Msg(format!("embedded cfg '{rel}' missing")))?;
    parse_model_yaml_text(text, &scale, &rel)
}

/// 剥离 scale 字符：yolo11n → yolo11，yolov8n-pose → yolov8-pose，yolo26n-seg → yolo26-seg。
fn strip_scale(stem: &str, scale: &str) -> String {
    if scale.is_empty() {
        return stem.to_string();
    }
    // 只替换 yolo<N><scale> 中的 scale 字符，保留 -seg/-pose 等后缀
    let re = regex::Regex::new(r"(yolo(?:e-)?[v]?\d+)(?:[nslmx])").unwrap();
    re.replace(stem, "${1}").to_string()
}

/// 按原始 stem（可含 scale）在内嵌注册表里找文件。
fn resolve_embedded_named(stem: &str) -> Option<String> {
    for d in ["", "v8/", "11/", "12/", "26/", "v10/", "v9/", "v6/", "v5/", "v3/", "rt-detr/"] {
        let c = if d.is_empty() { format!("models/{stem}.yaml") } else { format!("models/{d}{stem}.yaml") };
        if embedded::contains(&c) {
            return Some(c);
        }
    }
    None
}

/// 在内嵌注册表里按裸名（不含 scale）找文件。
fn resolve_embedded(bare: &str) -> Option<String> {
    // 嵌入键带 models/ 前缀；按目录优先级匹配
    const DIRS: &[&str] = &["", "v8/", "11/", "12/", "26/", "v10/", "v9/", "v6/", "v5/", "v3/", "rt-detr/"];
    for d in DIRS {
        let c = if d.is_empty() { format!("models/{bare}.yaml") } else { format!("models/{d}{bare}.yaml") };
        if embedded::contains(&c) {
            return Some(c);
        }
    }
    None
}

// ---------------- 数据集 YAML ----------------

/// ultralytics 数据集 YAML（path/train/val/test/names）。
#[derive(Debug, Clone)]
pub struct DatasetYaml {
    pub path: String,
    pub train: Vec<String>,
    pub val: Vec<String>,
    pub test: Vec<String>,
    /// 索引 → 类名。
    pub names: Vec<String>,
    /// 是否关键点任务（kpt_shape 存在）。
    pub kpt_shape: Option<(i64, i64)>,
    pub channel_names: Option<Vec<String>>,
}

fn resolve_split(base: &Path, items: &[String]) -> Vec<String> {
    items
        .iter()
        .map(|s| {
            let p = Path::new(s);
            if p.is_absolute() {
                s.clone()
            } else {
                base.join(p).to_string_lossy().into_owned()
            }
        })
        .collect()
}

/// 解析数据集 YAML。split 值可以是目录、txt 文件列表或 zip 引用（这里仅记路径，
/// 展开逻辑在 ul-data 中完成）。
pub fn parse_dataset_yaml(path: &Path) -> Result<DatasetYaml, crate::UlError> {
    let text = std::fs::read_to_string(path)?;
    let v: Value = serde_yaml::from_str(&text)?;
    let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    // path 字段：绝对则用原值，否则相对 yaml 所在目录
    let raw_path = v.get("path").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let root = if Path::new(&raw_path).is_absolute() {
        std::path::PathBuf::from(&raw_path)
    } else {
        base.join(&raw_path)
    };
    let get_split = |key: &str| -> Vec<String> {
        match v.get(key) {
            Some(Value::String(s)) => vec![root.join(s).to_string_lossy().into_owned()],
            Some(Value::Sequence(seq)) => seq
                .iter()
                .filter_map(|x| x.as_str())
                .map(|s| root.join(s).to_string_lossy().into_owned())
                .collect(),
            _ => vec![],
        }
    };
    let names: Vec<String> = match v.get("names") {
        Some(Value::Sequence(seq)) => seq.iter().map(|x| x.as_str().unwrap_or("?").to_string()).collect(),
        Some(Value::Mapping(m)) => {
            let mut pairs: Vec<(i64, String)> = m
                .iter()
                .filter_map(|(k, val)| Some((k.as_i64()?, val.as_str()?.to_string())))
                .collect();
            pairs.sort();
            pairs.into_iter().map(|(_, n)| n).collect()
        }
        _ => vec![],
    };
    Ok(DatasetYaml {
        path: root.to_string_lossy().into_owned(),
        train: resolve_split(&root, &get_split("train")),
        val: resolve_split(&root, &get_split("val")),
        test: resolve_split(&root, &get_split("test")),
        names,
        kpt_shape: v.get("kpt_shape").and_then(|x| x.as_sequence()).and_then(|s| {
            Some((s.first()?.as_i64()?, s.get(1)?.as_i64()?))
        }),
        channel_names: v
            .get("channels")
            .and_then(|x| x.as_sequence())
            .map(|s| s.iter().map(|x| x.as_str().unwrap_or("?").to_string()).collect()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const YOLO11: &str = include_str!("../../../ultralytics/cfg/models/11/yolo11.yaml");

    #[test]
    fn parse_yolo11() {
        let y = parse_model_yaml_text(YOLO11, "n", "11/yolo11.yaml").unwrap();
        assert_eq!(y.nc, 80);
        assert_eq!(y.scale, "n");
        assert!((y.depth_multiple - 0.50).abs() < 1e-9);
        assert!((y.width_multiple - 0.25).abs() < 1e-9);
        assert_eq!(y.max_channels, 1024.0);
        assert_eq!(y.reg_max, 16);
        assert!(!y.end2end);
        // yolo11 backbone 第一层 Conv [64,3,2]
        assert_eq!(y.backbone[0].module, "Conv");
        assert_eq!(y.backbone[0].args[0].as_i64(), Some(64));
    }

    #[test]
    fn guess_scale() {
        assert_eq!(guess_model_scale("yolo11n"), "n");
        assert_eq!(guess_model_scale("yolov8x-pose"), "x");
        assert_eq!(guess_model_scale("yolo26"), "");
    }

    #[test]
    fn strip() {
        assert_eq!(strip_scale("yolo11n", "n"), "yolo11");
        assert_eq!(strip_scale("yolov8n-pose", "n"), "yolov8-pose");
        assert_eq!(strip_scale("yolo26n-seg", "n"), "yolo26-seg");
    }
}
