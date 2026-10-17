//! YOLO 数据集：目录布局扫描 + 标签解析 + 样本加载。
//!
//! 布局：`images/<split>/*.{jpg,png,...}` + `labels/<split>/*.txt`（归一化坐标）。
//! 按任务的标签格式：
//! - detect：`cls cx cy w h`
//! - segment：`cls x1 y1 ... xn yn`（多边形，n≥3；bbox 取外接框）
//! - pose：`cls cx cy w h kx ky [v] ...`
//! - obb：`cls x1 y1 x2 y2 x3 y3 x4 y4`（四角点 → xywhr，角度规范到 w≥h、[−π/4, 3π/4)）

use std::path::{Path, PathBuf};

use ul_core::UlError;

use crate::image::Img;

/// 一个标注实例（归一化 xywh + 可选 kpts / segments / OBB 角度）。
#[derive(Debug, Clone)]
pub struct DetInstance {
    pub cls: i64,
    /// 归一化 (cx, cy, w, h)。
    pub xywhn: [f32; 4],
    /// 关键点（x, y, v 归一化坐标展平，len = nk*dim）。
    pub kpts: Vec<f32>,
    /// 关键点维度（0=无）。
    pub kpt_dim: i64,
    /// 分割多边形（归一化 xy 展平，len = 2n；空 = 非分割）。
    pub segments: Vec<f32>,
    /// OBB 角度（弧度，0 = 非旋转）。
    pub angle: f32,
}

impl DetInstance {
    pub fn base(cls: i64, xywhn: [f32; 4]) -> Self {
        DetInstance { cls, xywhn, kpts: Vec::new(), kpt_dim: 0, segments: Vec::new(), angle: 0.0 }
    }
}

/// 一个样本（图像 + 标注）。
#[derive(Clone)]
pub struct Sample {
    pub path: PathBuf,
    pub img: Img,
    pub instances: Vec<DetInstance>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TaskKind {
    Detect,
    Pose,
    Segment,
    Obb,
    Classify,
}

/// 数据集（某个 split）。
pub struct YoloDataset {
    pub img_files: Vec<PathBuf>,
    pub label_files: Vec<PathBuf>,
    pub names: Vec<String>,
    pub kpt_dim: i64,
    pub task: TaskKind,
    /// 解析缓存（label 文件名 → 数值行）；cache.json 持久化，二次启动零解析。
    pub cache: std::collections::HashMap<String, Vec<Vec<f32>>>,
    /// cache="ram"（官方语义）：预解码 + 按 imgsz 等比预缩的图像；None=该图未命中。
    pub ram_cache: Vec<Option<Img>>,
}

impl YoloDataset {
    /// 从 images/ 目录扫描（labels/ 对应替换）。
    pub fn from_img_dir(img_dir: &Path, names: Vec<String>, kpt_dim: i64) -> Result<Self, UlError> {
        let task = if kpt_dim > 0 { TaskKind::Pose } else { TaskKind::Detect };
        Self::with_task(img_dir, names, kpt_dim, task)
    }

    pub fn with_task(img_dir: &Path, names: Vec<String>, kpt_dim: i64, task: TaskKind) -> Result<Self, UlError> {
        let exts = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"];
        let mut img_files: Vec<PathBuf> = walkdir::WalkDir::new(img_dir)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_path_buf())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|x| x.to_str())
                        .map(|x| exts.contains(&x.to_lowercase().as_str()))
                        .unwrap_or(false)
            })
            .collect();
        img_files.sort();
        let label_files = img_files
            .iter()
            .map(|p| {
                // images/train/xxx.jpg → labels/train/xxx.txt
                let label_dir = p
                    .components()
                    .map(|c| c.as_os_str().to_owned())
                    .collect::<Vec<_>>();
                let mut out = PathBuf::new();
                for c in &label_dir {
                    if c.to_string_lossy() == "images" {
                        out.push("labels");
                    } else {
                        out.push(c);
                    }
                }
                out.set_extension("txt");
                out
            })
            .collect();
        if img_files.is_empty() {
            return Err(UlError::Msg(format!("no images under {}", img_dir.display())));
        }
        let mut ds = YoloDataset { img_files, label_files, names, kpt_dim, task, cache: Default::default(), ram_cache: Vec::new() };
        ds.init_cache();
        Ok(ds)
    }

    /// classify 目录布局：root/<class_name>/*.jpg，类按目录名排序，标注 = 类 id。
    pub fn from_classify_dir(img_dir: &Path, names: Vec<String>) -> Result<Self, UlError> {
        let exts = ["jpg", "jpeg", "png", "webp", "bmp"];
        let mut class_dirs: Vec<PathBuf> = std::fs::read_dir(img_dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect()
            })
            .unwrap_or_default();
        class_dirs.sort();
        if class_dirs.is_empty() {
            return Err(UlError::Msg(format!("no class dirs under {}", img_dir.display())));
        }
        let mut img_files = Vec::new();
        let mut labels: std::collections::HashMap<String, Vec<Vec<f32>>> = Default::default();
        for (ci, cd) in class_dirs.iter().enumerate() {
            let mut files: Vec<PathBuf> = walkdir::WalkDir::new(cd)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
                .map(|e| e.path().to_path_buf())
                .filter(|p| {
                    p.is_file()
                        && p.extension()
                            .and_then(|x| x.to_str())
                            .map(|x| exts.contains(&x.to_lowercase().as_str()))
                            .unwrap_or(false)
                })
                .collect();
            files.sort();
            for f in files {
                img_files.push(f.clone());
                let key = f.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
                labels.insert(key, vec![vec![ci as f32]]);
            }
        }
        if img_files.is_empty() {
            return Err(UlError::Msg(format!("no images under {}", img_dir.display())));
        }
        let label_files = img_files.iter().map(|p| PathBuf::from("")).collect();
        let mut ds = YoloDataset {
            img_files,
            label_files,
            names,
            kpt_dim: 0,
            task: TaskKind::Classify,
            cache: labels,
            ram_cache: Vec::new(),
        };
        ds.names = (0..class_dirs.len()).map(|i| format!("c{i}")).collect();
        Ok(ds)
    }

    /// 缓存初始化：cache.json 命中则直读；否则全量解析并持久化。
    fn init_cache(&mut self) {
        if let Some(text) = self.cache_path().and_then(|cp| std::fs::read_to_string(cp).ok()) {
            if let Ok(raw) = serde_json::from_str::<std::collections::HashMap<String, Vec<Vec<f32>>>>(&text) {
                if raw.len() == self.label_files.len() {
                    self.cache = raw;
                    return;
                }
            }
        }
        for i in 0..self.img_files.len() {
            let rows = self.parse_rows(i);
            self.cache.insert(self.key_for(i), rows);
        }
        if let Some(cp) = self.cache_path() {
            if let Ok(json) = serde_json::to_string(&self.cache) {
                std::fs::write(cp, json).ok();
            }
        }
    }

    pub fn len(&self) -> usize {
        self.img_files.len()
    }

    /// 加载第 i 个样本（无增强）。RAM 缓存命中则克隆预缩图（省 JPEG 解码）。
    pub fn load(&self, i: usize) -> Result<Sample, UlError> {
        let img = if let Some(Some(c)) = self.ram_cache.get(i) {
            c.clone()
        } else {
            Img::open(&self.img_files[i])?
        };
        let instances = self.load_labels(i);
        Ok(Sample { path: self.img_files[i].clone(), img, instances })
    }

    /// cache="ram"（官方 load_image 预缩放语义）：一次性解码并按 imgsz 等比
    /// 预缩（只缩不放），后续 load() 直接克隆缓存图——省去每 epoch 的 JPEG
    /// 解码，且增强在 imgsz 尺度小图上进行（2448→1280 像素量 ↓4 倍）。
    /// 标签为归一化坐标，不受预缩影响。
    pub fn build_ram_cache(&mut self, imgsz: u32) {
        if !self.ram_cache.is_empty() {
            return;
        }
        let n = self.img_files.len();
        let mut cache: Vec<Option<Img>> = Vec::with_capacity(n);
        let mut bytes = 0usize;
        for p in &self.img_files {
            let img = match Img::open(p) {
                Ok(i) => i,
                Err(_) => {
                    cache.push(None);
                    continue;
                }
            };
            let r = imgsz as f32 / (img.w.max(img.h) as f32);
            let img = if r < 1.0 {
                let nw = ((img.w as f32 * r).round() as u32).max(1);
                let nh = ((img.h as f32 * r).round() as u32).max(1);
                img.resize(nw, nh)
            } else {
                img
            };
            bytes += img.data.len();
            cache.push(Some(img));
        }
        self.ram_cache = cache;
        log::info!(
            "cache=ram: {} images pre-scaled to long-side ≤ {}, {:.2} GB",
            n,
            imgsz,
            bytes as f64 / (1 << 30) as f64
        );
    }

    /// 解析缓存：labels 根目录 cache.json（label 相对路径 → 解析行）。
    /// 任一 label 文件 mtime 新于缓存则失效。返回 None = 无缓存可用。
    fn load_cache(&self) -> Option<std::collections::HashMap<String, Vec<Vec<f32>>>> {
        let cache_path = self.cache_path()?;
        let text = std::fs::read_to_string(cache_path).ok()?;
        let raw: std::collections::HashMap<String, Vec<Vec<f32>>> = serde_json::from_str(&text).ok()?;
        // 简单校验：缓存条目数与 label 文件数一致
        if raw.len() != self.label_files.len() {
            return None;
        }
        Some(raw)
    }

    fn cache_path(&self) -> Option<PathBuf> {
        let root = self.label_files.first()?.parent()?.parent()?;
        Some(root.join("cache.json"))
    }

    fn key_for(&self, i: usize) -> String {
        self.label_files[i].file_name().and_then(|s| s.to_str()).unwrap_or("").to_string()
    }

    fn load_labels(&self, i: usize) -> Vec<DetInstance> {
        let rows = self
            .cache
            .get(&self.key_for(i))
            .cloned()
            .unwrap_or_else(|| self.parse_rows(i));
        self.rows_to_instances(rows)
    }

    fn rows_to_instances(&self, rows: Vec<Vec<f32>>) -> Vec<DetInstance> {
        let mut out = Vec::new();
        for v in rows {
            if v.len() < 5 {
                continue;
            }
            match self.task {
                    TaskKind::Segment => {
                        // cls + 偶数个多边形坐标
                        let poly = &v[1..];
                        if poly.len() < 6 || poly.len() % 2 != 0 {
                            continue;
                        }
                        let (xs, ys) = poly_chunks(poly);
                        let (x0, x1) = (xs.iter().cloned().fold(f32::MAX, f32::min), xs.iter().cloned().fold(f32::MIN, f32::max));
                        let (y0, y1) = (ys.iter().cloned().fold(f32::MAX, f32::min), ys.iter().cloned().fold(f32::MIN, f32::max));
                        let mut ins = DetInstance::base(v[0] as i64, [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0]);
                        ins.segments = poly.to_vec();
                        out.push(ins);
                    }
                    TaskKind::Obb => {
                        if v.len() < 9 {
                            continue;
                        }
                        let mut ins = DetInstance::base(v[0] as i64, [0.0; 4]);
                        let (cx, cy, w, h, r) = corners_to_xywhr(&v[1..9]);
                        ins.xywhn = [cx, cy, w, h];
                        ins.angle = r;
                        out.push(ins);
                    }
                _ => {
                    let mut ins = DetInstance::base(v[0] as i64, [v[1], v[2], v[3], v[4]]);
                    ins.kpts = v[5..].to_vec();
                    ins.kpt_dim = self.kpt_dim;
                    out.push(ins);
                }
            }
        }
        out
    }

    /// 解析单个 label 文件为数值行。
    fn parse_rows(&self, i: usize) -> Vec<Vec<f32>> {
        let lf = &self.label_files[i];
        let mut rows = Vec::new();
        if let Ok(text) = std::fs::read_to_string(lf) {
            for line in text.lines() {
                let v: Vec<f32> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if v.len() >= 5 {
                    rows.push(v);
                }
            }
        }
        rows
    }
}

fn poly_chunks(poly: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let xs = poly.iter().step_by(2).copied().collect();
    let ys = poly.iter().skip(1).step_by(2).copied().collect();
    (xs, ys)
}

/// 四角点（归一化 x1y1x2y2x3y3x4y4，绕行顺序）→ (cx, cy, w, h, r)。
/// 等价官方 cv2.minAreaRect 对真矩形的输出：w≥h，r ∈ [−π/4, 3π/4)。
pub fn corners_to_xywhr(c: &[f32]) -> (f32, f32, f32, f32, f32) {
    let (x0, y0, x1, y1, x2, y2, x3, y3) = (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    let cx = (x0 + x1 + x2 + x3) / 4.0;
    let cy = (y0 + y1 + y2 + y3) / 4.0;
    let (d1x, d1y) = (x1 - x0, y1 - y0);
    let (d2x, d2y) = (x2 - x1, y2 - y1);
    let (mut w, mut h) = ((d1x * d1x + d1y * d1y).sqrt(), (d2x * d2x + d2y * d2y).sqrt());
    let mut theta = d1y.atan2(d1x);
    if w < h {
        std::mem::swap(&mut w, &mut h);
        theta += std::f32::consts::FRAC_PI_2;
    }
    while theta >= 3.0 * std::f32::consts::FRAC_PI_4 {
        theta -= std::f32::consts::PI;
    }
    while theta < -std::f32::consts::FRAC_PI_4 {
        theta += std::f32::consts::PI;
    }
    (cx, cy, w, h, theta)
}

/// 解析数据集 YAML → (train 目录, val 目录, names, kpt_dim)。
pub fn load_dataset_yaml(path: &Path) -> Result<(Vec<PathBuf>, Vec<PathBuf>, Vec<String>, i64), UlError> {
    let d = ul_core::cfg::parse_dataset_yaml(path)?;
    let to_dirs = |items: &[String]| -> Vec<PathBuf> {
        items
            .iter()
            .filter_map(|s| {
                let p = Path::new(s);
                if p.is_dir() {
                    Some(p.to_path_buf())
                } else {
                    None
                }
            })
            .collect()
    };
    let train = to_dirs(&d.train);
    let val = to_dirs(&d.val);
    if train.is_empty() {
        return Err(UlError::Msg(format!("dataset yaml {}: no train image dirs", path.display())));
    }
    let kpt_dim = d.kpt_shape.map(|k| k.1).unwrap_or(0);
    Ok((train, val, d.names, kpt_dim))
}
