//! YOLO 数据集：目录布局扫描 + 标签解析 + 样本加载。
//!
//! 布局：`images/<split>/*.{jpg,png,...}` + `labels/<split>/*.txt`
//! （txt 行：`cls cx cy w h [kpt_x kpt_y kpt_v ...]`，归一化坐标）。

use std::path::{Path, PathBuf};

use ul_core::UlError;

use crate::image::Img;

/// 一个标注实例（归一化 xywh + 可选 kpts / OBB 角度）。
#[derive(Debug, Clone)]
pub struct DetInstance {
    pub cls: i64,
    /// 归一化 (cx, cy, w, h)。
    pub xywhn: [f32; 4],
    /// 关键点（x, y, v 归一化坐标展平，len = nk*3 或 nk*2）。
    pub kpts: Vec<f32>,
    /// 关键点维度（0=无）。
    pub kpt_dim: i64,
    /// OBB 角度（弧度，0 = 非旋转）。
    pub angle: f32,
}

/// 一个样本（图像 + 标注）。
#[derive(Clone)]
pub struct Sample {
    pub path: PathBuf,
    pub img: Img,
    pub instances: Vec<DetInstance>,
}

/// 数据集（某个 split）。
pub struct YoloDataset {
    pub img_files: Vec<PathBuf>,
    pub label_files: Vec<PathBuf>,
    pub names: Vec<String>,
    pub kpt_dim: i64,
    pub task: TaskKind,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TaskKind {
    Detect,
    Pose,
}

impl YoloDataset {
    /// 从 images/ 目录扫描（labels/ 对应替换）。
    pub fn from_img_dir(img_dir: &Path, names: Vec<String>, kpt_dim: i64) -> Result<Self, UlError> {
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
        Ok(YoloDataset { img_files, label_files, names, kpt_dim, task: if kpt_dim > 0 { TaskKind::Pose } else { TaskKind::Detect } })
    }

    pub fn len(&self) -> usize {
        self.img_files.len()
    }

    /// 加载第 i 个样本（无增强）。
    pub fn load(&self, i: usize) -> Result<Sample, UlError> {
        let img = Img::open(&self.img_files[i])?;
        let instances = self.load_labels(i, img.w as f32, img.h as f32);
        Ok(Sample { path: self.img_files[i].clone(), img, instances })
    }

    fn load_labels(&self, i: usize, w: f32, h: f32) -> Vec<DetInstance> {
        let lf = &self.label_files[i];
        let mut out = Vec::new();
        if let Ok(text) = std::fs::read_to_string(lf) {
            for line in text.lines() {
                let v: Vec<f32> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if v.len() < 5 {
                    continue;
                }
                out.push(DetInstance {
                    cls: v[0] as i64,
                    xywhn: [v[1], v[2], v[3], v[4]],
                    kpts: v[5..].to_vec(),
                    kpt_dim: self.kpt_dim,
                    angle: 0.0,
                });
            }
        }
        let _ = (w, h);
        out
    }
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
                } else if p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("txt") {
                    // txt 文件列表：逐行是图片路径
                    None // 简化：v1 只支持目录
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
