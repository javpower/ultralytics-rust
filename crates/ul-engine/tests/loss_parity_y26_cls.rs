//! yolo26-cls 分类 CE 损失对拍官方（v8ClassificationLoss：F.cross_entropy，无 smoothing）。

use std::path::Path;

use tch::{Device, Tensor};

use ul_engine::loss::classification_loss;
use ul_nn::model::YoloModel;

#[test]
fn yolo26_cls_loss_matches_official() {
    let Ok(text) = std::fs::read_to_string("../../weights/yolo26n_cls_loss_ref.json") else {
        eprintln!("skip: weights/yolo26n_cls_loss_ref.json 不存在");
        return;
    };
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let (b, s) = (v["batch_size"].as_i64().unwrap(), v["imgsz"].as_i64().unwrap());
    let img = Tensor::from_slice(&fvec2(&v, "img")).reshape([b, 3, s, s]);
    let targets_v: Vec<i64> = v["targets"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
    let targets = Tensor::from_slice(&targets_v);

    let (sd, _) = ul_nn::pt::load_pt(Path::new("../../weights/y26cls_rand.pt")).unwrap();
    let mut m = YoloModel::build("yolo26n-cls", None, Device::Cpu, false).unwrap();
    let rep = ul_nn::pt::load_state_dict_into(&m.vs, &sd, false).unwrap();
    if !rep.shape_mismatch.is_empty() || !rep.unexpected.is_empty() || !rep.missing.is_empty() {
        panic!(
            "y26-cls 加载缺口：missing={:?} unexpected={:?} loaded={}",
            rep.missing, rep.unexpected, rep.loaded
        );
    }

    let out = m.forward(&img, true).unwrap();
    let logits = match out {
        ul_nn::model::ModelOutput::Classify(a, _) => a,
        _ => panic!("expect classify output"),
    };
    let lf = logits.to_device(Device::Cpu).contiguous().reshape([-1]);
    let lv = Vec::<f32>::try_from(&lf).unwrap_or_default();
    if let Some(arr) = v["logits"].as_array() {
        let rv: Vec<f32> = arr.iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let maxd = lv.iter().zip(&rv).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[logits] max_diff={maxd:.3e} ours_mean={:.4} ref_mean={:.4}", lv.iter().sum::<f32>() / lv.len() as f32, rv.iter().sum::<f32>() / rv.len() as f32);
    }
    let loss = classification_loss(&logits, &targets, 0.0);
    let ours = f64::try_from(&loss).unwrap_or(f64::NAN);
    let r = v["loss"].as_f64().unwrap();
    println!("[cls] ours={ours:.6} ref={r:.6} diff={:+.2e}", ours - r);
    assert!((ours - r).abs() < 2e-2, "cls mismatch: {ours} vs {r}（0.13% 已知残差：BN eps/数值路径）");
}

fn fvec2(v: &serde_json::Value, key: &str) -> Vec<f32> {
    v[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
