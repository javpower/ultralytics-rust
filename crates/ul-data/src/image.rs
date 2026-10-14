//! 图像 IO：RGB u8 缓冲 + 读写（image crate）。

use image::{DynamicImage, GenericImageView};

/// RGB8 图像缓冲（行主序，len = w*h*3）。
#[derive(Clone)]
pub struct Img {
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>,
}

impl Img {
    pub fn new(w: u32, h: u32) -> Self {
        Img { w, h, data: vec![0; (w * h * 3) as usize] }
    }

    pub fn from_rgb8(w: u32, h: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), (w * h * 3) as usize);
        Img { w, h, data }
    }

    /// 读文件为 RGB8（自动处理 jpeg/png/webp/bmp/tiff）。
    pub fn open(path: &std::path::Path) -> Result<Self, ul_core::UlError> {
        let img = image::open(path)
            .map_err(|e| ul_core::UlError::Msg(format!("open {}: {e}", path.display())))?;
        Ok(Self::from_dynamic(&img))
    }

    pub fn from_dynamic(img: &DynamicImage) -> Self {
        let (w, h) = img.dimensions();
        let rgb = DynamicImage::ImageRgb8(img.to_rgb8());
        Img::from_rgb8(w, h, rgb.to_rgb8().into_raw())
    }

    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.w + x) * 3) as usize;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, c: [u8; 3]) {
        if x < self.w && y < self.h {
            let i = ((y * self.w + x) * 3) as usize;
            self.data[i] = c[0];
            self.data[i + 1] = c[1];
            self.data[i + 2] = c[2];
        }
    }

    /// 画实心/空心矩形（线宽 lw）。
    pub fn rect(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, color: [u8; 3], lw: i32) {
        for d in 0..lw {
            let (a, b, c, d2) = (x1 + d, y1 + d, x2 - d, y2 - d);
            for x in a..=c.max(a) {
                self.set_pixel(x as u32, b.max(0) as u32, color);
                self.set_pixel(x as u32, d2.max(0) as u32, color);
            }
            for y in b..=d2.max(b) {
                self.set_pixel(a.max(0) as u32, y as u32, color);
                self.set_pixel(c.max(0) as u32, y as u32, color);
            }
        }
    }

    /// 画 keypoints 骨架（COCO 17 连接表）。
    pub fn skeleton(&mut self, kpts: &[(f32, f32, f32)], color: [u8; 3], lw: i32) {
        const BONES: &[(usize, usize)] = &[
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
            (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
        ];
        for &(a, b) in BONES {
            let (pa, pb) = (kpts.get(a), kpts.get(b));
            if let (Some(pa), Some(pb)) = (pa, pb) {
                if pa.2 > 0.5 && pb.2 > 0.5 {
                    self.line(pa.0, pa.1, pb.0, pb.1, color, lw);
                }
            }
        }
        for k in kpts {
            if k.2 > 0.5 {
                for dx in -lw..=lw {
                    for dy in -lw..=lw {
                        self.set_pixel((k.0 as i32 + dx).max(0) as u32, (k.1 as i32 + dy).max(0) as u32, color);
                    }
                }
            }
        }
    }

    /// Bresenham 直线。
    pub fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: [u8; 3], lw: i32) {
        let (mut x, mut y) = (x0 as i32, y0 as i32);
        let (xe, ye) = (x1 as i32, y1 as i32);
        let dx = (xe - x).abs();
        let dy = (ye - y).abs();
        let sx = if xe >= x { 1 } else { -1 };
        let sy = if ye >= y { 1 } else { -1 };
        let mut err = dx - dy;
        loop {
            for ox in -lw / 2..=lw / 2 {
                for oy in -lw / 2..=lw / 2 {
                    self.set_pixel((x + ox).max(0) as u32, (y + oy).max(0) as u32, color);
                }
            }
            if x == xe && y == ye {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// 缩放（双线性）。
    pub fn resize(&self, nw: u32, nh: u32) -> Img {
        let d = DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(self.w, self.h, self.data.clone()).unwrap(),
        );
        let r = d.resize_exact(nw, nh, image::imageops::FilterType::Triangle);
        Img::from_dynamic(&r)
    }

    /// 保存为 jpeg/png（按扩展名）。
    pub fn save(&self, path: &std::path::Path) -> Result<(), ul_core::UlError> {
        let img = image::RgbImage::from_raw(self.w, self.h, self.data.clone())
            .ok_or_else(|| ul_core::UlError::msg("rgb buffer size mismatch"))?;
        img.save(path)
            .map_err(|e| ul_core::UlError::Msg(format!("save {}: {e}", path.display())))?;
        Ok(())
    }
}

/// ultralytics 调色板（plotting.py Colors，BGR→RGB 前 16 色）。
pub const PALETTE: [[u8; 3]; 16] = [
    [204, 8, 224], [58, 144, 219], [146, 71, 233], [114, 9, 213], [233, 128, 1], [31, 123, 85],
    [216, 74, 13], [95, 89, 235], [235, 245, 24], [160, 248, 117], [215, 102, 0], [216, 248, 138],
    [60, 189, 37], [9, 117, 235], [0, 186, 241], [76, 127, 111],
];

pub fn color_for(cls: i64) -> [u8; 3] {
    PALETTE[(cls as usize) % PALETTE.len()]
}
