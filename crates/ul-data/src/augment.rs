//! 增强管线：mosaic4 / mixup / 随机透视（translate/scale/shear/degrees）/ HSV / 翻转 /
//! letterbox（验证与预测），与 ultralytics v8_transforms 语义一致。

use ul_core::bbox::LetterboxParams;

use crate::dataset::{DetInstance, Sample};
use crate::image::Img;

/// 增强超参（对应 default.yaml 的 Hyperparameters 段）。
#[derive(Debug, Clone)]
pub struct AugmentCfg {
    pub hsv_h: f64,
    pub hsv_s: f64,
    pub hsv_v: f64,
    pub degrees: f64,
    pub translate: f64,
    pub scale: f64,
    pub shear: f64,
    pub perspective: f64,
    pub flipud: f64,
    pub fliplr: f64,
    pub bgr: f64,
    pub mosaic: f64,
    pub mixup: f64,
    pub close_mosaic: usize,
}

impl Default for AugmentCfg {
    fn default() -> Self {
        AugmentCfg {
            hsv_h: 0.015,
            hsv_s: 0.7,
            hsv_v: 0.4,
            degrees: 0.0,
            translate: 0.1,
            scale: 0.5,
            shear: 0.0,
            perspective: 0.0,
            flipud: 0.0,
            fliplr: 0.5,
            bgr: 0.0,
            mosaic: 1.0,
            mixup: 0.0,
            close_mosaic: 10,
        }
    }
}

/// 随机源（可复现：seed = hash(epoch, index)）。
pub struct Rng(rng::Rng);

mod rng {
    /// xorshift64*（确定性、跨平台一致）。
    pub struct Rng(pub u64);
    impl Rng {
        pub fn new(seed: u64) -> Self {
            Rng(seed.max(1))
        }
        pub fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        pub fn f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.f32()
        }
        pub fn bool_p(&mut self, p: f64) -> bool {
            self.f32() < p as f32
        }
    }
}

fn seed_for(epoch: usize, index: usize) -> u64 {
    (epoch as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (index as u64).wrapping_mul(0xBF58476D1CE4E5B9)
}

/// 训练样本批处理上下文（mosaic 需要随机邻居）。
pub struct Augmentor {
    pub cfg: AugmentCfg,
    pub imgsz: u32,
    pub epoch: usize,
    pub dataset_len: usize,
    /// 邻居取样回调（mosaic 用）。
    pub neighbor: Box<dyn Fn(usize, &mut dyn FnMut(usize)) + Sync>,
}

impl Augmentor {
    /// 对第 `index` 个样本做完整训练增强（含 letterbox 到 imgsz）。
    pub fn train_augment(&self, index: usize, load: &dyn Fn(usize) -> Option<Sample>) -> Sample {
        let mut rng = Rng::rng_new(seed_for(self.epoch, index));
        let mosaic_on = self.cfg.mosaic > 0.0
            && (self.epoch + 1 < self.epoch + 1 || true) // close_mosaic 由 trainer 调整 cfg.mosaic
            && rng.bool_p(self.cfg.mosaic);
        let mut sample = if mosaic_on {
            self.mosaic4(index, load, &mut rng)
        } else {
            load(index).expect("sample load")
        };
        // 随机透视/仿射
        random_perspective(
            &mut sample,
            self.imgsz,
            self.cfg.translate,
            self.cfg.scale,
            self.cfg.shear,
            self.cfg.degrees,
            self.cfg.perspective,
            &mut rng,
        );
        // mixup
        if self.cfg.mixup > 0.0 && rng.bool_p(self.cfg.mixup) {
            let j = rng.next_u64() as usize % self.dataset_len;
            if let Some(other) = load(j) {
                mixup(&mut sample, other, &mut rng);
            }
        }
        // HSV
        if self.cfg.hsv_h > 0.0 || self.cfg.hsv_s > 0.0 || self.cfg.hsv_v > 0.0 {
            augment_hsv(
                &mut sample.img,
                self.cfg.hsv_h,
                self.cfg.hsv_s,
                self.cfg.hsv_v,
                &mut rng,
            );
        }
        // 翻转
        if self.cfg.flipud > 0.0 && rng.bool_p(self.cfg.flipud) {
            flip_ud(&mut sample);
        }
        if self.cfg.fliplr > 0.0 && rng.bool_p(self.cfg.fliplr) {
            flip_lr(&mut sample);
        }
        // BGR 通道交换
        if self.cfg.bgr > 0.0 && rng.bool_p(self.cfg.bgr) {
            for px in sample.img.data.chunks_exact_mut(3) {
                px.swap(0, 2);
            }
        }
        sample
    }

    /// mosaic4：四图拼 S×S。
    fn mosaic4(&self, index: usize, load: &dyn Fn(usize) -> Option<Sample>, rng: &mut Rng) -> Sample {
        let s = self.imgsz as i32;
        let s2 = s / 2;
        // 随机拼接中心
        let (yc, xc) = (
            rng.range((s2 * 3 / 4) as f32, (s2 * 5 / 4) as f32) as i32,
            rng.range((s2 * 3 / 4) as f32, (s2 * 5 / 4) as f32) as i32,
        );
        let mut canvas = Img::new((2 * s) as u32, (2 * s) as u32);
        let mut instances: Vec<DetInstance> = Vec::new();
        // 4 个位置：左上/右上/左下/右下，第 0 位是 index 本体
        let mut picked = [index; 4];
        for place in 1..4 {
            let mut j = index;
            (self.neighbor)(index, &mut |k| j = k);
            if j == index {
                j = (rng.next_u64() as usize) % self.dataset_len;
            }
            picked[place] = j;
        }
        let places: [(i32, i32, i32, i32); 4] = [
            (0, 0, xc, yc),                       // 左上：画到 (0,0)，占 (xc, yc)
            (xc, 0, 2 * s, yc),                   // 右上
            (0, yc, xc, 2 * s),                   // 左下
            (xc, yc, 2 * s, 2 * s),               // 右下
        ];
        for (place, j) in picked.iter().enumerate() {
            let Some(mut smp) = load(*j) else { continue };
            let (x1, y1, x2, y2) = places[place];
            let tw = x2 - x1;
            let th = y2 - y1;
            // 等比缩放填满该象限
            let gain = (tw as f32 / smp.img.w as f32).min(th as f32 / smp.img.h as f32);
            let (nw, nh) = ((smp.img.w as f32 * gain) as i32, (smp.img.h as f32 * gain) as i32);
            let resized = smp.img.resize(nw.max(1) as u32, nh.max(1) as u32);
            let (ox, oy) = match place {
                0 => (x2 - nw, y2 - nh), // 右下对齐拼接边
                1 => (x1, y1),
                2 => (x2 - nw, y1),
                _ => (x1, y1),
            };
            for yy in 0..nh {
                let ty = oy + yy;
                if ty < 0 || ty >= 2 * s {
                    continue;
                }
                for xx in 0..nw {
                    let tx = ox + xx;
                    if tx < 0 || tx >= 2 * s {
                        continue;
                    }
                    let p = resized.pixel(xx as u32, yy as u32);
                    canvas.set_pixel(tx as u32, ty as u32, p);
                }
            }
            // 标注映射：canvas 2S 空间
            for inst in &smp.instances {
                let mut ins = inst.clone();
                ins.xywhn[0] = (ins.xywhn[0] * nw as f32 + ox as f32) / (2 * s) as f32;
                ins.xywhn[1] = (ins.xywhn[1] * nh as f32 + oy as f32) / (2 * s) as f32;
                ins.xywhn[2] *= nw as f32 / (2 * s) as f32;
                ins.xywhn[3] *= nh as f32 / (2 * s) as f32;
                for k in ins.kpts.chunks_exact_mut(3) {
                    k[0] = (k[0] * nw as f32 + ox as f32) / (2 * s) as f32;
                    k[1] = (k[1] * nh as f32 + oy as f32) / (2 * s) as f32;
                }
                for pt in ins.segments.chunks_exact_mut(2) {
                    pt[0] = (pt[0] * nw as f32 + ox as f32) / (2 * s) as f32;
                    pt[1] = (pt[1] * nh as f32 + oy as f32) / (2 * s) as f32;
                }
                instances.push(ins);
            }
        }
        // 裁剪中央 [.., xc..xc+s, yc..yc+s]
        let mut crop = Img::new(s as u32, s as u32);
        for yy in 0..s {
            for xx in 0..s {
                let p = canvas.pixel((xc + xx) as u32, (yc + yy) as u32);
                crop.set_pixel(xx as u32, yy as u32, p);
            }
        }
        for ins in &mut instances {
            ins.xywhn[0] = (ins.xywhn[0] * 2.0 * s as f32 - xc as f32) / s as f32;
            ins.xywhn[1] = (ins.xywhn[1] * 2.0 * s as f32 - yc as f32) / s as f32;
            ins.xywhn[2] *= 2.0;
            ins.xywhn[3] *= 2.0;
            for k in ins.kpts.chunks_exact_mut(3) {
                k[0] = (k[0] * 2.0 * s as f32 - xc as f32) / s as f32;
                k[1] = (k[1] * 2.0 * s as f32 - yc as f32) / s as f32;
            }
            for pt in ins.segments.chunks_exact_mut(2) {
                pt[0] = (pt[0] * 2.0 * s as f32 - xc as f32) / s as f32;
                pt[1] = (pt[1] * 2.0 * s as f32 - yc as f32) / s as f32;
            }
        }
        // 裁剪后剔除出界的框
        instances.retain(|ins| {
            let (cx, cy, w, h) = (ins.xywhn[0], ins.xywhn[1], ins.xywhn[2], ins.xywhn[3]);
            cx > -0.05 && cx < 1.05 && cy > -0.05 && cy < 1.05 && w > 0.001 && h > 0.001
        });
        Sample { path: std::path::PathBuf::from(format!("mosaic4:{}", picked[0])), img: crop, instances }
    }
}

impl Rng {
    pub fn rng_new(seed: u64) -> Self {
        Rng(rng::Rng::new(seed))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    pub fn f32(&mut self) -> f32 {
        self.0.f32()
    }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        self.0.range(lo, hi)
    }
    pub fn bool_p(&mut self, p: f64) -> bool {
        self.0.bool_p(p)
    }
}

/// 独立随机源（trainer 邻居取样用）。
pub fn rng_for(seed: u64) -> Rng {
    Rng(rng::Rng::new(seed))
}

/// 随机透视/仿射（ultralytics RandomPerspective）。
pub fn random_perspective(
    sample: &mut Sample,
    imgsz: u32,
    translate: f64,
    scale: f64,
    shear: f64,
    degrees: f64,
    perspective: f64,
    rng: &mut Rng,
) {
    let s = imgsz as f32;
    let (h, w) = (sample.img.h as f32, sample.img.w as f32);
    // 当前图已经是 imgsz 尺寸（mosaic/直载后由 caller 保证；否则先 letterbox）
    if (h - s).abs() > 0.5 || (w - s).abs() > 0.5 {
        let lb = letterbox(&sample.img, imgsz as u32, [114, 114, 114]);
        let gain_w = s / w;
        let gain_h = s / h;
        let params = LetterboxParams::new(w, h, s, s);
        let _ = (gain_w, gain_h);
        for ins in &mut sample.instances {
            ins.xywhn = [
                (ins.xywhn[0] * w * params.gain_w + params.pad_w) / s,
                (ins.xywhn[1] * h * params.gain_h + params.pad_h) / s,
                ins.xywhn[2] * w * params.gain_w / s,
                ins.xywhn[3] * h * params.gain_h / s,
            ];
        }
        sample.img = lb;
    }

    // 变换参数
    let (cx, cy) = (s / 2.0, s / 2.0);
    let dx = rng.range(-translate as f32, translate as f32) * s;
    let dy = rng.range(-translate as f32, translate as f32) * s;
    let sc = 1.0 - rng.range(0.0, scale as f32) + rng.range(0.0, scale as f32);
    let sh = (rng.range(-shear as f32, shear as f32) as f64).to_radians().tan() as f32;
    let ang = (rng.range(-degrees as f32, degrees as f32) as f64).to_radians();
    let (cos_a, sin_a) = (ang.cos() as f32, ang.sin() as f32);
    let px = if perspective > 0.0 { rng.range(-perspective as f32, perspective as f32) } else { 0.0 };
    let py = if perspective > 0.0 { rng.range(-perspective as f32, perspective as f32) } else { 0.0 };

    // 正向矩阵 M = T(center)·R·S·Shear·T(-center)+平移，透视项 [px, py]
    // 输出坐标 = M · [x, y, 1]^T（齐次，3x3）
    // [a11 a12 a13; a21 a22 a23; p1 p2 1]
    let a11 = cos_a * sc;
    let a12 = -sin_a * sc + sh * cos_a * sc;
    let a13 = cx + dx - cx * a11 - cy * a12;
    let a21 = sin_a * sc;
    let a22 = cos_a * sc + sh * sin_a * sc;
    let a23 = cy + dy - cx * a21 - cy * a22;

    // 逆映射：解 2x2 线性 + 透视（对每像素）——直接数值解（perspective 常为 0，退化快路径）
    let warp = |img: &Img, out: &mut Img| {
        for yo in 0..out.h {
            for xo in 0..out.w {
                let (xo_f, yo_f) = (xo as f32 + 0.5, yo as f32 + 0.5);
                // 逆：先用透视逆（p 项小，一阶近似：den = 1 + px*x' + py*y'，x'=x-dx 等）
                let mut xi = a11 * (xo_f - a13) + a12 * (yo_f - a23);
                let mut yi = a21 * (xo_f - a13) + a22 * (yo_f - a23);
                // a 逆 = adj/det
                let det = a11 * a22 - a12 * a21;
                if det.abs() < 1e-9 {
                    continue;
                }
                let inv11 = a22 / det;
                let inv12 = -a12 / det;
                let inv21 = -a21 / det;
                let inv22 = a11 / det;
                let (rx, ry) = (xo_f - a13, yo_f - a23);
                xi = inv11 * rx + inv12 * ry;
                yi = inv21 * rx + inv22 * ry;
                // 透视一阶逆
                let den = 1.0 + px * xi + py * yi;
                let den = if den.abs() < 1e-6 { 1.0 } else { den };
                xi /= den;
                yi /= den;
                if xi >= 0.0 && yi >= 0.0 && xi < img.w as f32 - 1.0 && yi < img.h as f32 - 1.0 {
                    // 双线性
                    let (x0, y0) = (xi as u32, yi as u32);
                    let (fx, fy) = (xi - x0 as f32, yi - y0 as f32);
                    for ch in 0..3 {
                        let p00 = img.pixel(x0, y0)[ch] as f32;
                        let p10 = img.pixel(x0 + 1, y0)[ch] as f32;
                        let p01 = img.pixel(x0, y0 + 1)[ch] as f32;
                        let p11 = img.pixel(x0 + 1, y0 + 1)[ch] as f32;
                        let v = p00 * (1.0 - fx) * (1.0 - fy)
                            + p10 * fx * (1.0 - fy)
                            + p01 * (1.0 - fx) * fy
                            + p11 * fx * fy;
                        out.data[((yo as usize * out.w as usize + xo as usize) * 3 + ch)] = v.clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    };
    let mut out = Img::new(imgsz, imgsz);
    warp(&sample.img, &mut out);
    sample.img = out;

    // 标注点变换（归一化 → 像素 → 矩阵 → 归一化）
    let to_px = |ins: &DetInstance| {
        let pts = [
            [ins.xywhn[0], ins.xywhn[1]],
            [ins.xywhn[0] - ins.xywhn[2] / 2.0, ins.xywhn[1] - ins.xywhn[3] / 2.0],
            [ins.xywhn[0] + ins.xywhn[2] / 2.0, ins.xywhn[1] + ins.xywhn[3] / 2.0],
        ];
        pts
    };
    let apply = |p: [f32; 2]| -> [f32; 2] {
        let (x, y) = (p[0] * s, p[1] * s);
        let den = 1.0 + px * x + py * y;
        let den = if den.abs() < 1e-6 { 1.0 } else { den };
        let xr = a11 * x + a12 * y + a13;
        let yr = a21 * x + a22 * y + a23;
        [xr / den / s, yr / den / s]
    };
    for ins in &mut sample.instances {
        // 多边形逐点变换（分割），随后由外接框重建 bbox（官方：bbox 由 segments 派生）
        if !ins.segments.is_empty() {
            for pt in ins.segments.chunks_exact_mut(2) {
                let p = apply([pt[0], pt[1]]);
                pt[0] = p[0];
                pt[1] = p[1];
            }
            let pts = polygon_parts(&ins.segments);
            let (x0, x1) = (pts.iter().map(|p| p[0]).fold(f32::MAX, f32::min), pts.iter().map(|p| p[0]).fold(f32::MIN, f32::max));
            let (y0, y1) = (pts.iter().map(|p| p[1]).fold(f32::MAX, f32::min), pts.iter().map(|p| p[1]).fold(f32::MIN, f32::max));
            ins.xywhn = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0];
        } else {
            let pts = to_px(ins);
            let c = apply(pts[0]);
            let p1 = apply(pts[1]);
            let p2 = apply(pts[2]);
            ins.xywhn = [c[0], c[1], (p2[0] - p1[0]).abs(), (p2[1] - p1[1]).abs()];
        }
        for k in ins.kpts.chunks_exact_mut(3) {
            let kp = apply([k[0], k[1]]);
            k[0] = kp[0];
            k[1] = kp[1];
            if k[2] != 0.0 && (kp[0] < 0.0 || kp[0] > 1.0 || kp[1] < 0.0 || kp[1] > 1.0) {
                k[2] = 0.0;
            }
        }
    }
    sample.instances.retain(|ins| {
        ins.xywhn[0] > -0.05
            && ins.xywhn[0] < 1.05
            && ins.xywhn[1] > -0.05
            && ins.xywhn[1] < 1.05
            && ins.xywhn[2] > 0.001
            && ins.xywhn[3] > 0.001
    });
}

/// mixup：线性混合两图与标注框并集。
pub fn mixup(a: &mut Sample, b: Sample, rng: &mut Rng) {
    let lam = 0.5f32.max(rng.range(0.1, 0.9).min(0.5));
    if a.img.w != b.img.w || a.img.h != b.img.h {
        let b2 = b.img.resize(a.img.w, a.img.h);
        for i in 0..a.img.data.len() {
            let va = a.img.data[i] as f32 * lam + b2.data[i] as f32 * (1.0 - lam);
            a.img.data[i] = va.clamp(0.0, 255.0) as u8;
        }
    } else {
        for i in 0..a.img.data.len() {
            let va = a.img.data[i] as f32 * lam + b.img.data[i] as f32 * (1.0 - lam);
            a.img.data[i] = va.clamp(0.0, 255.0) as u8;
        }
    }
    a.instances.extend(b.instances);
}

/// HSV 抖动（等价 ultralytics augment_hsv 的增益随机化）。
pub fn augment_hsv(img: &mut Img, hg: f64, sg: f64, vg: f64, rng: &mut Rng) {
    let r = rng.range(-hg as f32, hg as f32) * 179.0;
    let s = 1.0 + rng.range(-sg as f32, sg as f32);
    let v = 1.0 + rng.range(-vg as f32, vg as f32);
    // 快速路径：只做 V/S 缩放 + 简化 H 平移（RGB 空间近似会偏色，这里做真 HSV 转换）
    for px in img.data.chunks_exact_mut(3) {
        let (mut hh, mut ss, mut vv) = rgb_to_hsv(px[0], px[1], px[2]);
        hh = (hh + r).rem_euclid(179.0);
        ss = (ss * s).clamp(0.0, 255.0);
        vv = (vv * v).clamp(0.0, 255.0);
        let (r8, g8, b8) = hsv_to_rgb(hh, ss, vv);
        px[0] = r8;
        px[1] = g8;
        px[2] = b8;
    }
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (rf, gf, bf) = (r as f32, g as f32, b as f32);
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == rf {
        60.0 * ((gf - bf) / d) / 2.0 + 30.0
    } else if max == gf {
        60.0 * ((bf - rf) / d) / 2.0 + 60.0
    } else {
        60.0 * ((rf - gf) / d) / 2.0 + 90.0
    };
    (h, if max == 0.0 { 0.0 } else { d / max * 255.0 }, max)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    // h ∈ [0, 179]（OpenCV 风格 0..180），s/v ∈ [0, 255]
    let h6 = h / 30.0; // 0..6
    let i = h6.floor() as i32;
    let f = h6 - i as f32;
    let sf = s / 255.0;
    let vf = v / 255.0;
    let p = vf * (1.0 - sf);
    let q = vf * (1.0 - sf * f);
    let t = vf * (1.0 - sf * (1.0 - f));
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (vf, t, p),
        1 => (q, vf, p),
        2 => (p, vf, t),
        3 => (p, q, vf),
        4 => (t, p, vf),
        _ => (vf, p, q),
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// 竖直翻转。
pub fn flip_ud(sample: &mut Sample) {
    let (w, h) = (sample.img.w, sample.img.h);
    let mut flipped = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = sample.img.pixel(x, y);
            flipped.set_pixel(x, h - 1 - y, p);
        }
    }
    sample.img = flipped;
    for ins in &mut sample.instances {
        ins.xywhn[1] = 1.0 - ins.xywhn[1];
        for pt in ins.segments.chunks_exact_mut(2) {
            pt[1] = 1.0 - pt[1];
        }
        for k in ins.kpts.chunks_exact_mut(3) {
            k[1] = 1.0 - k[1];
        }
    }
}

/// 水平翻转（含 COCO17 关键点左右交换表）。
pub fn flip_lr(sample: &mut Sample) {
    let (w, h) = (sample.img.w, sample.img.h);
    let mut flipped = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = sample.img.pixel(x, y);
            flipped.set_pixel(w - 1 - x, y, p);
        }
    }
    sample.img = flipped;
    for ins in &mut sample.instances {
        ins.xywhn[0] = 1.0 - ins.xywhn[0];
        for pt in ins.segments.chunks_exact_mut(2) {
            pt[0] = 1.0 - pt[0];
        }
        if ins.kpt_dim >= 2 {
            const SWAP: [usize; 17] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15];
            let nd = ins.kpt_dim as usize;
            let kpts = ins.kpts.clone();
            let nk = kpts.len() / nd;
            for i in 0..nk {
                let j = SWAP.get(i).copied().unwrap_or(i);
                let src = &kpts[j * nd..(j + 1) * nd];
                ins.kpts[i * nd..(i + 1) * nd].copy_from_slice(src);
                ins.kpts[i * nd] = 1.0 - ins.kpts[i * nd];
            }
        }
    }
}

/// letterbox（114 灰边，center=True），返回重排后的图。
pub fn letterbox(img: &Img, new_size: u32, color: [u8; 3]) -> Img {
    let params = LetterboxParams::new(img.w as f32, img.h as f32, new_size as f32, new_size as f32);
    let nw = (img.w as f32 * params.gain_w).round() as u32;
    let nh = (img.h as f32 * params.gain_h).round() as u32;
    let resized = img.resize(nw.max(1), nh.max(1));
    let mut out = Img::new(new_size, new_size);
    if color != [0, 0, 0] {
        for px in out.data.chunks_exact_mut(3) {
            px[0] = color[0];
            px[1] = color[1];
            px[2] = color[2];
        }
    }
    let ox = ((new_size - nw) / 2) as i32;
    let oy = ((new_size - nh) / 2) as i32;
    for y in 0..nh as i32 {
        for x in 0..nw as i32 {
            let p = resized.pixel(x as u32, y as u32);
            out.set_pixel((x + ox) as u32, (y + oy) as u32, p);
        }
    }
    out
}

/// 多边形展平数组 → 顶点对。
pub fn polygon_parts(seg: &[f32]) -> Vec<[f32; 2]> {
    seg.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}

/// 把样本的多边形按实例顺序光栅化为 (imgsz×imgsz) 掩码，像素值 = 实例序号+1（0=背景）。
/// 语义对齐官方 overlap=True 的合并掩码（后画覆盖先画）。
pub fn rasterize_mask(sample: &Sample, imgsz: u32) -> Vec<u8> {
    let mut out = vec![0u8; (imgsz * imgsz) as usize];
    for (idx, ins) in sample.instances.iter().enumerate() {
        if ins.segments.len() < 6 {
            continue;
        }
        let pts: Vec<[f32; 2]> = ins
            .segments
            .chunks_exact(2)
            .map(|c| [c[0] * imgsz as f32, c[1] * imgsz as f32])
            .collect();
        let n = pts.len();
        // 扫描线：每行求交点并填充跨段
        for yy in 0..imgsz as i32 {
            let y = yy as f32 + 0.5;
            let mut xs: Vec<f32> = Vec::new();
            for i in 0..n {
                let a = pts[i];
                let b = pts[(i + 1) % n];
                if (a[1] <= y && b[1] > y) || (b[1] <= y && a[1] > y) {
                    let t = (y - a[1]) / (b[1] - a[1]);
                    xs.push(a[0] + t * (b[0] - a[0]));
                }
            }
            xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
            for pair in xs.chunks(2) {
                if pair.len() < 2 {
                    continue;
                }
                let x0 = pair[0].max(0.0) as i32;
                let x1 = (pair[1].ceil() as i32).min(imgsz as i32);
                for xx in x0..x1 {
                    if xx >= 0 && (xx as u32) < imgsz {
                        out[(yy as u32 * imgsz + xx as u32) as usize] = (idx + 1) as u8;
                    }
                }
            }
        }
    }
    out
}
