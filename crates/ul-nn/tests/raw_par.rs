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
