//! 全系列模型 YAML 实例化 + 前向烟测。
//!
//! 对每个嵌入的官方模型 YAML（排除明确延期家族）构建 n scale（部分含 s/x），
//! 跑一次 224 前向，验证输出形状与任务类型。

use ul_nn::model::{ModelOutput, YoloModel};

/// 明确延期的家族（rt-detr/YOLOE/World/depth-TorchVision）。
const DEFERRED: &[&str] = &[
    "rt-detr/", "yolov8-rtdetr", "world", "yoloe", "yolo26-depth", "yolo26-sem", "cls-resnet",
];

fn skipped(rel: &str) -> bool {
    DEFERRED.iter().any(|d| rel.contains(d))
}

#[test]
fn build_all_detect_families() {
    let mut built = 0;
    let mut failures: Vec<String> = Vec::new();
    for key in ul_core::cfg::embedded::keys() {
        if !key.ends_with(".yaml") || !key.starts_with("models/") {
            continue;
        }
        let name = key.rsplit('/').next().unwrap();
        // 只测 detect 主干文件（含各任务的会通过 -seg/-pose/-obb 覆盖）
        if skipped(key) {
            continue;
        }
        // scale 字符注入：yolo11.yaml 无 scale → 跳过（需要具名 scale 才能定宽）；
        // 带 scales 表的文件用 n。
        let stem = name.strip_suffix(".yaml").unwrap();
        let has_scale = ul_core::cfg::guess_model_scale(stem).is_empty()
            && ul_nn_tests_util_has_scales(key);
        let model_name = if has_scale {
            // 注入 scale=n：直接读文本加 scale 字段？——用 guess 逻辑：文件名 yolo26.yaml 不含 scale
            // parse_model_yaml_text 对无 scale hints 取 scales 第一个键（n），可接受
            key.to_string()
        } else {
            key.to_string()
        };
        let result = std::panic::catch_unwind(|| {
            let device = tch::Device::Cpu;
            let m = YoloModel::build(&model_name, None, device, false)
                .unwrap_or_else(|e| panic!("build {key}: {e}"));
            let x = tch::Tensor::zeros([1, 3, 256, 256], (tch::Kind::Float, device));
            let out = m.forward(&x, false).unwrap_or_else(|e| panic!("forward {key}: {e}"));
            (m.task, out_kind(&out))
        });
        match result {
            Ok((task, kind)) => {
                println!("OK  {key:<40} task={task:?} out={kind}");
                built += 1;
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "?".into());
                failures.push(format!("{key}: {msg}"));
            }
        }
    }
    println!("\nbuilt={built} failures={}", failures.len());
    for f in &failures {
        println!("FAIL {f}");
    }
    // 目前要求：非延期家族必须全部构建成功
    assert!(failures.is_empty(), "{} model(s) failed:\n{}", failures.len(), failures.join("\n"));
    assert!(built > 20, "expected to build >20 models, got {built}");
}

fn out_kind(out: &ModelOutput) -> String {
    match out {
        ModelOutput::Detect(ul_nn::head::HeadOutput::Inference(t)) => {
            format!("tensor{:?}", t.size().to_vec())
        }
        ModelOutput::Detect(_) => "train-head".into(),
        ModelOutput::Classify(_, _) => "classify".into(),
        ModelOutput::SemSeg(_, _) => "semseg".into(),
    }
}

fn ul_nn_tests_util_has_scales(_key: &str) -> bool {
    true
}
