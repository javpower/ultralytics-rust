//! `yolo` 启动器（跨平台：Windows / Linux / macOS）：本体仅数百 KB、零第三方依赖。
//!
//! 职责：
//! 1. 开发模式：设置了 LIBTORCH → 直接拉起引擎（PATH 追加 $LIBTORCH/lib）
//! 2. 绿色模式：检测 runtime/libtorch；缺失时探测显卡 → 自动下载对应
//!    libtorch 运行时（有 N 卡下 CUDA，无卡下 CPU；mac 为 CPU/MPS）并解压
//! 3. 以追加过动态库搜索路径的环境启动引擎并透传退出码
use std::path::{Path, PathBuf};
use std::process::Command;

const CORE: &str = "yolo_core";
const VER: &str = "2.11.0";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    // 1) 开发模式：LIBTORCH 已设置 → 信任既有环境
    let dev_lib = std::env::var("LIBTORCH").ok().map(PathBuf::from);
    // 2) 绿色模式：runtime/libtorch 下的主动态库就绪
    let rt_lib = exe_dir.join("runtime").join("libtorch").join("lib");
    let rt_ready = rt_lib.join(core_dylib()).exists();

    let lib_dir: PathBuf = if let Some(dev) = dev_lib {
        dev.join("lib")
    } else if rt_ready {
        rt_lib.clone()
    } else {
        bootstrap(&exe_dir, &rt_lib);
        if !rt_lib.join(core_dylib()).exists() {
            eprintln!("runtime install failed; see README");
            std::process::exit(1);
        }
        rt_lib.clone()
    };

    let core = exe_dir.join(core_exe());
    if !core.exists() {
        eprintln!("missing {} (distribute alongside yolo)", core.display());
        std::process::exit(1);
    }

    // 动态库搜索：为子进程前置运行时 lib 目录（按平台变量）
    let cur = std::env::var(lib_path_var()).unwrap_or_default();
    let path_new = format!("{};{}", lib_dir.display(), cur);
    let status = Command::new(&core)
        .args(&args)
        .env(lib_path_var(), &path_new)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("launch {} failed: {e}", core.display());
            std::process::exit(1);
        });
    std::process::exit(status.code().unwrap_or(1));
}

fn core_exe() -> String {
    format!("yolo_core{}", std::env::consts::EXE_SUFFIX)
}
fn core_dylib() -> &'static str {
    if cfg!(windows) {
        "torch.dll"
    } else if cfg!(target_os = "macos") {
        "libtorch.dylib"
    } else {
        "libtorch.so"
    }
}
fn lib_path_var() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// 首次自举：探测显卡 → 下载对应 libtorch → 解压到 runtime/libtorch。
fn bootstrap(exe_dir: &Path, rt_lib: &PathBuf) {
    let has_gpu = cfg!(not(target_os = "macos")) && gpu_present();
    let (variant, fname) = runtime_variant(has_gpu);
    let url = format!("https://download.pytorch.org/libtorch/{variant}/{fname}");
    println!("first run: libtorch runtime not found, installing ({variant}) …");
    println!("  {url}");
    let _ = std::fs::create_dir_all(exe_dir.join("runtime"));
    let zip = exe_dir.join("runtime").join(format!("libtorch-{variant}.zip"));
    println!("  downloading (CUDA ~2.8GB / CPU ~300MB) …");
    if !fetch(&url, &zip) {
        eprintln!("download failed; check network or install manually (see README)");
        std::process::exit(1);
    }
    println!("  extracting …");
    let tmp = exe_dir.join("runtime").join("_extract");
    let _ = std::fs::create_dir_all(&tmp);
    extract_zip(&zip, &tmp);
    let src = tmp.join("libtorch");
    if !src.join("lib").join(core_dylib()).exists() {
        eprintln!("extract failed; please place libtorch/ under runtime/ manually");
        std::process::exit(1);
    }
    let dst = exe_dir.join("runtime").join("libtorch");
    let _ = std::fs::remove_dir_all(&dst);
    if std::fs::rename(&src, &dst).is_err() {
        let _ = std::fs::remove_dir_all(&dst);
        if !copy_dir(&src, &dst) {
            eprintln!("install failed; move libtorch manually");
            std::process::exit(1);
        }
    }
    let _ = std::fs::remove_file(&zip);
    let _ = std::fs::remove_dir_all(&tmp);
    println!("  runtime installed → {}", dst.display());
}

/// libtorch 运行时下载坐标（按平台与 GPU）。
fn runtime_variant(has_gpu: bool) -> (&'static str, String) {
    let plus = "%2B";
    match (std::env::consts::OS, std::env::consts::ARCH, has_gpu) {
        ("windows", _, true) => ("cu128", format!("libtorch-win-shared-with-deps-{VER}{plus}cu128.zip")),
        ("windows", _, false) => ("cpu", format!("libtorch-win-shared-with-deps-{VER}{plus}cpu.zip")),
        ("linux", _, true) => ("cu128", format!("libtorch-cxx11-abi-shared-with-deps-{VER}{plus}cu128.zip")),
        ("linux", _, false) => ("cpu", format!("libtorch-cpu-{VER}{plus}cpu.zip")),
        ("macos", "aarch64", _) => ("cpu", format!("libtorch-macos-arm64-{VER}.zip")),
        ("macos", _, _) => ("cpu", format!("libtorch-macos-x86_64-{VER}.zip")),
        (os, _, _) => ("cpu", format!("libtorch-{os}-{VER}.zip")),
    }
}

fn gpu_present() -> bool {
    Command::new("nvidia-smi")
        .arg("-L")
        .nopopup()
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn fetch(url: &str, dst: &Path) -> bool {
    let ok = Command::new("curl")
        .args(["-sL", "--max-time", "3600", "-o"])
        .arg(dst)
        .arg(url)
        .nopopup()
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok && dst.exists() {
        return true;
    }
    // 回退仅 Windows（PowerShell）；unix curl 系统自带
    if cfg!(windows) {
        let ps = format!(
            "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -Uri '{u}' -OutFile '{d}'",
            u = url,
            d = dst.display()
        );
        return Command::new("powershell")
            .args(["-NoProfile", "-Command", &ps])
            .nopopup()
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
    }
    false
}

fn copy_dir(src: &Path, dst: &Path) -> bool {
    let _ = std::fs::create_dir_all(dst);
    let mut ok = true;
    if let Ok(rd) = std::fs::read_dir(src) {
        for e in rd.flatten() {
            let to = dst.join(e.file_name());
            if e.path().is_dir() {
                ok &= copy_dir(&e.path(), &to);
            } else {
                ok &= std::fs::copy(e.path(), &to).map(|_| true).unwrap_or(false);
            }
        }
    }
    ok
}

/// zip 解压（官方包顶层目录为 libtorch/）；防御 zip-slip。
fn extract_zip(zip: &Path, dst: &Path) {
    use std::io::Read;
    let Ok(file) = std::fs::File::open(zip) else { return };
    let Ok(mut arch) = zip::ZipArchive::new(file) else { return };
    for i in 0..arch.len() {
        let mut f = match arch.by_index(i) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let name = f.name().to_string();
        if !name.starts_with("libtorch/") || name.ends_with('/') {
            continue;
        }
        let out = dst.join(&name);
        if let Some(pp) = out.parent() {
            let _ = std::fs::create_dir_all(pp);
        }
        let mut buf = Vec::with_capacity(f.size() as usize);
        if f.read_to_end(&mut buf).is_err() {
            continue;
        }
        let _ = std::fs::write(&out, &buf);
    }
}

// 辅助命令不闪控制台窗口（仅 Windows 有此概念）
trait NoPopup {
    fn nopopup(&mut self) -> &mut Self;
}
impl NoPopup for Command {
    #[cfg(windows)]
    fn nopopup(&mut self) -> &mut Self {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        self.creation_flags(CREATE_NO_WINDOW)
    }
    #[cfg(not(windows))]
    fn nopopup(&mut self) -> &mut Self {
        self
    }
}
