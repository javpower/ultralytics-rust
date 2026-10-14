#[test]
fn raw_output_parity_320() {
    let (sd, _) = ul_nn::pt::load_pt(std::path::Path::new("E:/ultralytics-rust/weights/yolov8n.pt")).unwrap();
    let m = ul_nn::YoloModel::build("yolov8n", None, tch::Device::Cpu, false).unwrap();
    ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    // 与 py 脚本一致的输入：纯 resize（非 letterbox）
    let img = image::open("E:/dev/aegis-vision/data/coco8/images/val/000000000061.jpg").unwrap().to_rgb8();
    let r = image::imageops::resize(&img, 320, 320, image::imageops::FilterType::Triangle);
    let mut data = Vec::with_capacity(320 * 320 * 3);
    for c in 0..3 {
        for y in 0..320 {
            for x in 0..320 {
                data.push(r.get_pixel(x, y)[c] as f32 / 255.0);
            }
        }
    }
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let x = tch::Tensor::from_data_size(&bytes, &[1, 3, 320, 320], tch::Kind::Float);
    let out = match m.forward(&x, false).unwrap() {
        ul_nn::model::ModelOutput::Detect(ul_nn::head::HeadOutput::Inference(t)) => t,
        _ => panic!(),
    };
    let flat: Vec<f32> = Vec::try_from(&out.transpose(1, 2).contiguous().reshape([-1])).unwrap();
    // torch.save 的裸张量用我们的 pickle 加载器读不了（jit 格式）——改由 py 导出 json
    let js: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("E:/ultralytics-rust/weights/yolov8n_raw.json").unwrap(),
    )
    .unwrap();
    let ef: Vec<f32> = js["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    // py 输出 channel-major (84, 2100)：转成 anchor-major 再比
    let mut transposed: Vec<f32> = vec![0.0; 2100 * 84];
    for c in 0..84 {
        for a in 0..2100 {
            transposed[a * 84 + c] = ef[c * 2100 + a];
        }
    }
    let mut max_diff = 0f32;
    let mut worst = 0usize;
    for (i, (a, b)) in flat.iter().zip(transposed.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
            worst = i;
        }
    }
    println!("raw parity: max_diff={max_diff} at flat idx {worst} (anchor {}, chan {})", worst / 84, worst % 84);
    let mut box_bad = 0usize;
    let mut cls_bad = 0usize;
    let mut box_max = 0f32;
    let mut cls_max = 0f32;
    for a in 0..2100 {
        for c in 0..84 {
            let d = (flat[a * 84 + c] - transposed[a * 84 + c]).abs();
            if c < 4 {
                box_max = box_max.max(d);
                if d > 1e-2 {
                    box_bad += 1;
                }
            } else {
                cls_max = cls_max.max(d);
                if d > 1e-2 {
                    cls_bad += 1;
                }
            }
        }
    }
    println!("box: bad={box_bad} max={box_max:.4} | cls: bad={cls_bad} max={cls_max:.4}");
    println!("ours cls argmax first anchors:");
    for a in 0..8 {
        let mut best = (0f32, 0usize);
        for c in 0..80 {
            let v = flat[a * 84 + 4 + c];
            if v > best.0 {
                best = (v, c);
            }
        }
        println!("  anchor {a}: cls {} score {:.3}", best.1, best.0);
    }
    println!("py   cls argmax first anchors:");
    for a in 0..8 {
        let mut best = (0f32, 0usize);
        for c in 0..80 {
            let v = transposed[a * 84 + 4 + c];
            if v > best.0 {
                best = (v, c);
            }
        }
        println!("  anchor {a}: cls {} score {:.3}", best.1, best.0);
    }
    assert!(max_diff < 1e-3, "raw parity failed max_diff={max_diff}");
}

#[test]
fn raw_zeros640_parity() {
    let js: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("E:/ultralytics-rust/weights/yolov8n_zeros640.json").unwrap(),
    )
    .unwrap();
    let ef: Vec<f32> = js["values"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let (sd, _) = ul_nn::pt::load_pt(std::path::Path::new("E:/ultralytics-rust/weights/yolov8n.pt")).unwrap();
    let m = ul_nn::YoloModel::build("yolov8n", None, tch::Device::Cpu, false).unwrap();
    ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    let x = tch::Tensor::zeros([1, 3, 640, 640], (tch::Kind::Float, tch::Device::Cpu));
    let out = match m.forward(&x, false).unwrap() {
        ul_nn::model::ModelOutput::Detect(ul_nn::head::HeadOutput::Inference(t)) => t,
        _ => panic!(),
    };
    let flat: Vec<f32> = Vec::try_from(&out.reshape([-1])).unwrap();
    let mut max_diff = 0f32;
    for (a, b) in flat.iter().zip(ef.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    println!("zeros640 yolov8n parity: max_diff={max_diff:.3e} over {} values", ef.len());
    assert!(max_diff < 1e-3, "failed {max_diff}");
}

#[test]
fn raw_letterbox320_parity() {
    let js: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("E:/ultralytics-rust/weights/lb320.json").unwrap(),
    )
    .unwrap();
    let input: Vec<f32> = js["input"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let ef: Vec<f32> = js["values"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let (sd, _) = ul_nn::pt::load_pt(std::path::Path::new("E:/ultralytics-rust/weights/yolov8n.pt")).unwrap();
    let m = ul_nn::YoloModel::build("yolov8n", None, tch::Device::Cpu, false).unwrap();
    ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    let bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
    let x = tch::Tensor::from_data_size(&bytes, &[1, 3, 320, 320], tch::Kind::Float);
    let out = match m.forward(&x, false).unwrap() {
        ul_nn::model::ModelOutput::Detect(ul_nn::head::HeadOutput::Inference(t)) => t,
        _ => panic!(),
    };
    let ours: Vec<f32> = Vec::try_from(&out.reshape([-1])).unwrap();
    let mut max_diff = 0f32;
    let mut worst = 0usize;
    for (i, (a, b)) in ours.iter().zip(ef.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
            worst = i;
        }
    }
    println!("lb320 parity: max_diff={max_diff:.4} at chan {} anchor {}", worst / 2100, worst % 2100);
    // 分通道统计
    let mut per_chan = [0f32; 84];
    for c in 0..84 {
        for a in 0..2100 {
            per_chan[c] = per_chan[c].max((ours[c * 2100 + a] - ef[c * 2100 + a]).abs());
        }
    }
    println!("box chan max: {:?}", &per_chan[..4].iter().map(|f| f.to_string()).collect::<Vec<_>>());
    let cls_max = per_chan[4..].iter().cloned().fold(0f32, f32::max);
    println!("cls chan max overall: {cls_max:.4}");
    // 采样对比 w 通道值
    for a in [0usize, 800, 1600, 2000, 2092] {
        println!(
            "anchor {a}: ours xywh=({:.1},{:.1},{:.1},{:.1}) py xywh=({:.1},{:.1},{:.1},{:.1})",
            ours[a], ours[2100 + a], ours[2 * 2100 + a], ours[3 * 2100 + a],
            ef[a], ef[2100 + a], ef[2 * 2100 + a], ef[3 * 2100 + a]
        );
    }
    // 分层：P3 [0,1600) P4 [1600,2000) P5 [2000,2100)
    for (name, lo, hi) in [("P3", 0, 1600), ("P4", 1600, 2000), ("P5", 2000, 2100)] {
        let mut mx = 0f32;
        for c in 0..4 {
            for a in lo..hi {
                mx = mx.max((ours[c * 2100 + a] - ef[c * 2100 + a]).abs());
            }
        }
        println!("  {name} box max diff: {mx:.4}");
    }
    // 采样 anchor 100 与 2092 的原始 16-bin 分布（chan 2 = w 的 r 段）
    for a in [100usize, 2092] {
        let off = 2 * 16 * 2100 + a; // w 是第 3 组 16 bins? 实际布局 [l16,t16,r16,b16]
        let _ = off;
        print!("anchor {a} bins (chan-group layout): ");
        for g in 0..4 {
            let mut row = Vec::new();
            for k in 0..16 {
                row.push(format!("{:.2}", ours[(g * 16 + k) * 2100 + a]));
            }
            print!("[{}]", row.join(","));
        }
        println!();
    }
    assert!(max_diff < 1e-2, "lb320 parity failed {max_diff}");
}
