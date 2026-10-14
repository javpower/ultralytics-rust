//! 最小 pickle 虚拟机：解析 PyTorch `.pt`（zip）内嵌的 `data.pkl`。
//!
//! 支持 torch.save 协议 2-4 所需的 opcode 子集，产出 `PValue` 对象树；
//! `BINPERSID`（persistent_load）解析为 Storage 描述符，张量数据在 zip 的
//! `data/<key>` 成员中按 dtype/offset/stride 重建。

use std::collections::HashMap;

use ul_core::UlError;

/// pickle 对象树。
#[derive(Debug, Clone)]
pub enum PValue {
    None,
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<PValue>),
    Tuple(Vec<PValue>),
    /// 插入有序字典（键值对序列）。
    Dict(Vec<(PValue, PValue)>),
    /// GLOBAL/STACK_GLOBAL 的 (module, name)。
    Global(String, String),
    /// REDUCE：函数 + 参数。
    Reduce(Box<PValue>, Vec<PValue>),
    /// persistent_load 结果：('storage', 存储类型名, key, location, numel)。
    Storage { type_name: String, key: String, numel: i64 },
    /// 张量元数据（_rebuild_tensor_v2/v3 已在 VM 内归约）。
    Tensor { storage_key: String, offset: i64, size: Vec<i64>, stride: Vec<i64>, dtype: Option<String> },
    /// NEWOBJ+BUILD 捕获的任意对象：类名 + 属性 dict。
    Object { class: String, attrs: Vec<(PValue, PValue)> },
}

impl PValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            PValue::Int(i) => Some(*i),
            PValue::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PValue::Float(f) => Some(*f),
            PValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
    pub fn as_list(&self) -> Option<&[PValue]> {
        match self {
            PValue::List(l) | PValue::Tuple(l) => Some(l),
            _ => None,
        }
    }
    /// dict 查键（str 键）。
    pub fn get(&self, key: &str) -> Option<&PValue> {
        match self {
            PValue::Dict(pairs) | PValue::Object { attrs: pairs, .. } => pairs.iter().find_map(|(k, v)| {
                k.as_str().and_then(|ks| if ks == key { Some(v) } else { None })
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
enum StackItem {
    Val(PValue),
    /// MARK 标记前的栈深度。
    Mark(usize),
}

/// 反序列化 data.pkl。
pub fn loads(data: &[u8]) -> Result<PValue, UlError> {
    let mut vm = Vm { data, pos: 0, stack: Vec::new(), memo: HashMap::new() };
    vm.run()
}

struct Vm<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<StackItem>,
    memo: HashMap<i64, PValue>,
}

macro_rules! need {
    ($cond:expr, $msg:expr) => {
        if !($cond) {
            return Err(UlError::Msg($msg.to_string()));
        }
    };
}

impl<'a> Vm<'a> {
    fn u8(&mut self) -> Result<u8, UlError> {
        need!(self.pos < self.data.len(), "pickle: unexpected EOF");
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }
    fn le_i32(&mut self) -> Result<i32, UlError> {
        need!(self.pos + 4 <= self.data.len(), "pickle: EOF i32");
        let v = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn le_i64(&mut self) -> Result<i64, UlError> {
        need!(self.pos + 8 <= self.data.len(), "pickle: EOF i64");
        let v = i64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn le_u32(&mut self) -> Result<u32, UlError> {
        Ok(self.le_i32()? as u32)
    }
    fn le_f64(&mut self) -> Result<f64, UlError> {
        need!(self.pos + 8 <= self.data.len(), "pickle: EOF f64");
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], UlError> {
        need!(self.pos + n <= self.data.len(), "pickle: EOF bytes");
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// MARK：返回 mark 位置并弹出该深度以上的所有值。
    fn pop_mark(&mut self) -> Result<Vec<PValue>, UlError> {
        // 从栈顶向下找最近的 Mark
        let mark_pos = self
            .stack
            .iter()
            .rposition(|s| matches!(s, StackItem::Mark(_)))
            .ok_or_else(|| UlError::Msg("pickle: MARK without mark".into()))?;
        let items: Vec<PValue> = self.stack[mark_pos + 1..]
            .iter()
            .map(|s| match s {
                StackItem::Val(v) => v.clone(),
                StackItem::Mark(_) => PValue::None,
            })
            .collect();
        self.stack.truncate(mark_pos);
        Ok(items)
    }

    fn push(&mut self, v: PValue) {
        self.stack.push(StackItem::Val(v));
    }

    fn pop(&mut self) -> Result<PValue, UlError> {
        match self.stack.pop() {
            Some(StackItem::Val(v)) => Ok(v),
            Some(StackItem::Mark(_)) => Err(UlError::Msg("pickle: pop on MARK".into())),
            None => Err(UlError::Msg("pickle: stack underflow".into())),
        }
    }

    fn memo_put(&mut self, idx: i64) {
        if let Some(StackItem::Val(v)) = self.stack.last() {
            let v = v.clone();
            self.memo.insert(idx, v);
        }
    }

    fn run(&mut self) -> Result<PValue, UlError> {
        let mut trace: Vec<(usize, u8)> = Vec::new();
        loop {
            let op_pos = self.pos;
            let op = self.u8()?;
            if std::env::var("UL_PICKLE_TRACE").is_ok() {
                trace.push((op_pos, op));
                if trace.len() > 24 {
                    trace.remove(0);
                }
            }
            match op {
                b'.' => break, // STOP
                0x80 => {
                    self.u8()?; // PROTO
                }
                0x95 => {
                    self.le_i64()?; // FRAME
                }
                b'(' => self.stack.push(StackItem::Mark(0)), // MARK
                b'N' => self.push(PValue::None),
                0x29 => self.push(PValue::Tuple(vec![])), // EMPTY_TUPLE
                b'T' => {
                    // BINSTRING（带长度）
                    let n = self.le_i32()? as usize;
                    let s = self.take(n)?;
                    self.push(PValue::Str(String::from_utf8_lossy(s).into_owned()));
                }
                b'U' => {
                    // SHORT_BINSTRING / SHORT_BINUNICODE
                    let n = self.u8()? as usize;
                    let s = self.take(n)?;
                    self.push(PValue::Str(String::from_utf8_lossy(s).into_owned()));
                }
                b'X' => {
                    // BINUNICODE
                    let n = self.le_u32()? as usize;
                    let s = self.take(n)?;
                    self.push(PValue::Str(String::from_utf8_lossy(s).into_owned()));
                }
                0x8c => {
                    // SHORT_BINUNICODE
                    let n = self.u8()? as usize;
                    let s = self.take(n)?;
                    self.push(PValue::Str(String::from_utf8_lossy(s).into_owned()));
                }
                0x8d => {
                    // BINUNICODE8
                    let n = self.le_i64()? as usize;
                    let s = self.take(n)?;
                    self.push(PValue::Str(String::from_utf8_lossy(s).into_owned()));
                }
                b'C' => {
                    // BINBYTES
                    let n = self.le_i32()? as usize;
                    let s = self.take(n)?.to_vec();
                    self.push(PValue::Bytes(s));
                }
                0x8e => {
                    // BINBYTES8
                    let n = self.le_i64()? as usize;
                    let s = self.take(n)?.to_vec();
                    self.push(PValue::Bytes(s));
                }
                b'K' => {
                    let v = self.u8()? as i64;
                    self.push(PValue::Int(v));
                }
                b'M' => {
                    let v = u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64;
                    self.push(PValue::Int(v));
                }
                b'J' => {
                    let v = self.le_i32()? as i64;
                    self.push(PValue::Int(v));
                }
                b'L' => {
                    // LONG：十进制字符串 + 'L'
                    let mut digits = Vec::new();
                    loop {
                        let b = self.u8()?;
                        if b == b'L' {
                            break;
                        }
                        digits.push(b);
                    }
                    let s = String::from_utf8_lossy(&digits).into_owned();
                    let v: i64 = s.trim().parse().unwrap_or(0);
                    self.push(PValue::Int(v));
                }
                0x8a => {
                    // LONG1
                    let n = self.u8()? as usize;
                    let bytes = self.take(n)?;
                    let mut v: i64 = 0;
                    for (i, b) in bytes.iter().enumerate() {
                        v |= (*b as i64) << (8 * i);
                    }
                    // 符号扩展
                    if n > 0 && bytes[n - 1] & 0x80 != 0 {
                        v |= -1i64 << (8 * n);
                    }
                    self.push(PValue::Int(v));
                }
                b'G' => {
                    let v = self.le_f64()?;
                    self.push(PValue::Float(v));
                }
                0x88 => self.push(PValue::Bool(true)),
                0x89 => self.push(PValue::Bool(false)),
                b']' => self.push(PValue::List(vec![])),
                b'}' => self.push(PValue::Dict(vec![])),
                0x85 => {
                    // TUPLE1
                    let v = self.pop()?;
                    self.push(PValue::Tuple(vec![v]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(PValue::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(PValue::Tuple(vec![a, b, c]));
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.push(PValue::Tuple(items));
                }
                b'e' => {
                    // APPENDS：把 mark 后的所有元素加入 list
                    let items = self.pop_mark()?;
                    match self.stack.last_mut() {
                        Some(StackItem::Val(PValue::List(l))) => l.extend(items),
                        _ => return Err(UlError::Msg("pickle: APPENDS without list".into())),
                    }
                }
                b'a' => {
                    let v = self.pop()?;
                    match self.stack.last_mut() {
                        Some(StackItem::Val(PValue::List(l))) => l.push(v),
                        _ => return Err(UlError::Msg("pickle: APPEND without list".into())),
                    }
                }
                b'u' => {
                    // SETITEMS（proto 0 dict）
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(UlError::Msg(format!("pickle: SETITEMS odd at {}", self.pos)));
                    }
                    if !matches!(self.stack.last(), Some(StackItem::Val(PValue::Dict(_)))) {
                        return Err(UlError::Msg(format!(
                            "pickle: SETITEMS without dict at {} top={:?}",
                            self.pos,
                            self.stack.last().map(|s| match s {
                                StackItem::Val(v) => format!("{v:?}").chars().take(60).collect::<String>(),
                                StackItem::Mark(_) => "MARK".into(),
                            })
                        )));
                    }
                    match self.stack.last_mut() {
                        Some(StackItem::Val(PValue::Dict(d))) => {
                            for pair in items.chunks(2) {
                                d.push((pair[0].clone(), pair[1].clone()));
                            }
                        }
                        _ => return Err(UlError::Msg("pickle: SETITEMS without dict".into())),
                    }
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.stack.last_mut() {
                        Some(StackItem::Val(PValue::Dict(d))) => d.push((k, v)),
                        _ => return Err(UlError::Msg("pickle: SETITEM without dict".into())),
                    }
                }
                b'c' => {
                    // GLOBAL（两行 \n 结尾）
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    self.push(PValue::Global(module, name));
                }
                0x93 => {
                    // STACK_GLOBAL：栈顶 name，次栈顶 module
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let m = module.as_str().unwrap_or("").to_string();
                    let n = name.as_str().unwrap_or("").to_string();
                    self.push(PValue::Global(m, n));
                }
                b'R' => {
                    // REDUCE：栈顶 args tuple，次栈顶 func
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let arg_list = args.as_list().map(|l| l.to_vec()).unwrap_or_default();
                    let reduced = self.reduce(&func, arg_list)?;
                    self.push(reduced);
                }
                0x81 => {
                    // NEWOBJ：cls + args —— 捕获为 Object（类名）
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let name = match &cls {
                        PValue::Global(_, n) => n.clone(),
                        _ => "?".into(),
                    };
                    let arg_list = args.as_list().map(|l| l.to_vec()).unwrap_or_default();
                    self.push(PValue::Object { class: name, attrs: vec![(PValue::Str("__newargs__".into()), PValue::Tuple(arg_list))] });
                }
                0x92 => {
                    // NEWOBJ_EX
                    let kw = self.pop()?;
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let name = match &cls {
                        PValue::Global(_, n) => n.clone(),
                        _ => "?".into(),
                    };
                    let _ = kw;
                    let arg_list = args.as_list().map(|l| l.to_vec()).unwrap_or_default();
                    self.push(PValue::Object { class: name, attrs: vec![(PValue::Str("__newargs__".into()), PValue::Tuple(arg_list))] });
                }
                b'b' => {
                    // BUILD：state 更新对象
                    let state = self.pop()?;
                    let obj = self.pop()?;
                    self.build(obj, state)?;
                }
                b'Q' => {
                    // BINPERSID
                    let pid = self.pop()?;
                    let stor = self.persistent_load(&pid)?;
                    self.push(stor);
                }
                b'q' => {
                    let idx = self.u8()? as i64;
                    self.memo_put(idx);
                }
                b'r' => {
                    let idx = self.le_i32()? as i64;
                    self.memo_put(idx);
                }
                0x94 => {
                    // MEMOIZE：自动递增 id
                    let idx = self.memo.len() as i64;
                    self.memo_put(idx);
                }
                b'h' => {
                    // BINGET：1 字节 memo 索引（经验证与 torch.save 流一致）
                    let idx = self.u8()? as i64;
                    let v = self.memo.get(&idx).cloned().ok_or_else(|| {
                        UlError::Msg(format!("pickle: BINGET {} not in memo (len={})", idx, self.memo.len()))
                    })?;
                    self.push(v);
                }
                b'j' => {
                    // LONG_BINGET：4 字节 memo 索引
                    let idx = self.le_i32()? as i64;
                    let v = self.memo.get(&idx).cloned().ok_or_else(|| {
                        UlError::Msg(format!("pickle: LONG_BINGET {} not in memo (len={}) pos={}", idx, self.memo.len(), op_pos))
                    })?;
                    self.push(v);
                }
                0x90 => {
                    // FROZENSET → 当 list
                    let items = self.pop_mark()?;
                    self.push(PValue::List(items));
                }
                0x8f => self.push(PValue::List(vec![])), // EMPTY_SET
                other => {
                    return Err(UlError::Msg(format!(
                        "pickle: unsupported opcode 0x{other:02x} at {} trace={:x?}",
                        self.pos - 1,
                        &trace[..trace.len().min(24)]
                    )));
                }
            }
        }
        match self.pop() {
            Ok(v) => Ok(v),
            Err(_) => Err(UlError::Msg("pickle: empty result".into())),
        }
    }

    fn read_line(&mut self) -> Result<String, UlError> {
        let mut out = Vec::new();
        loop {
            let b = self.u8()?;
            if b == b'\n' {
                break;
            }
            out.push(b);
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// REDUCE 归约：识别 torch 张量重建函数。
    fn reduce(&mut self, func: &PValue, args: Vec<PValue>) -> Result<PValue, UlError> {
        let fname = match func {
            PValue::Global(_, n) => n.clone(),
            _ => String::new(),
        };
        match fname.as_str() {
            "_rebuild_tensor_v2" | "_rebuild_tensor" => {
                // (storage, offset, size, stride, requires_grad, backward_hooks)
                let storage = args.first().cloned().unwrap_or(PValue::None);
                let offset = args.get(1).and_then(|a| a.as_i64()).unwrap_or(0);
                let size: Vec<i64> = args.get(2).and_then(|a| a.as_list()).map(|l| l.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
                let stride: Vec<i64> = args.get(3).and_then(|a| a.as_list()).map(|l| l.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
                let (key, type_name) = match &storage {
                    PValue::Storage { key, type_name, .. } => (key.clone(), type_name.clone()),
                    _ => (String::new(), String::new()),
                };
                Ok(PValue::Tensor { storage_key: key, offset, size, stride, dtype: Some(type_name) })
            }
            "_rebuild_tensor_v3" => {
                let storage = args.first().cloned().unwrap_or(PValue::None);
                let offset = args.get(1).and_then(|a| a.as_i64()).unwrap_or(0);
                let size: Vec<i64> = args.get(2).and_then(|a| a.as_list()).map(|l| l.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
                let stride: Vec<i64> = args.get(3).and_then(|a| a.as_list()).map(|l| l.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
                let dtype = args.get(6).and_then(|a| a.as_str()).map(|s| s.to_string());
                let key = match &storage {
                    PValue::Storage { key, .. } => key.clone(),
                    _ => String::new(),
                };
                Ok(PValue::Tensor { storage_key: key, offset, size, stride, dtype })
            }
            "_rebuild_parameter" | "_rebuild_parameter_with_state" | "_rebuild_nn_parameter" => {
                // (tensor_data, requires_grad, backward_hooks[, state])
                Ok(args.first().cloned().unwrap_or(PValue::None))
            }
            "_rebuild_nongrad" | "tensor" | "_rebuild_tensor_legacy" => Ok(args.first().cloned().unwrap_or(PValue::None)),
            "_rebuild_xla_tensor" => Ok(args.first().cloned().unwrap_or(PValue::None)),
            "OrderedDict" => Ok(PValue::Dict(vec![])), // collections.OrderedDict()
            "dict" => Ok(PValue::Dict(vec![])),
            "list" => Ok(PValue::List(vec![])),
            "set" | "frozenset" => Ok(PValue::List(vec![])),
            "_reconstruct" => {
                // (cls, state, ...)：取 state（第二参）尽量保留
                Ok(args.get(1).cloned().unwrap_or(PValue::None))
            }
            _ => {
                // 未识别函数：保守返回 args（占位）
                Ok(PValue::Tuple(args))
            }
        }
    }

    /// BUILD：把 state 合入对象属性。
    fn build(&mut self, obj: PValue, state: PValue) -> Result<(), UlError> {
        let mut obj = obj;
        match (&mut obj, state) {
            (PValue::Object { class, attrs }, PValue::Dict(pairs)) => {
                for (k, v) in pairs {
                    attrs.push((k, v));
                }
                let _ = class;
            }
            (PValue::Object { attrs, .. }, PValue::Tuple(t)) => {
                // __getstate__ 返回 tuple（如 (dict, None)）
                for item in t {
                    if let PValue::Dict(pairs) = item {
                        for (k, v) in pairs {
                            attrs.push((k, v));
                        }
                    }
                }
                let _ = attrs;
            }
            (PValue::Dict(d), PValue::Dict(pairs)) => d.extend(pairs),
            _ => {}
        }
        self.push(obj);
        Ok(())
    }

    /// persistent_load：pid = ('storage', type_global, key, location, numel)。
    fn persistent_load(&mut self, pid: &PValue) -> Result<PValue, UlError> {
        match pid {
            PValue::Tuple(items) if !items.is_empty() => {
                let kind = items[0].as_str().unwrap_or("");
                if kind == "storage" {
                    let type_name = match &items[1] {
                        PValue::Global(_, n) => n.clone(),
                        PValue::Str(s) => s.clone(),
                        _ => String::new(),
                    };
                    let key = items[2].as_str().unwrap_or("").to_string();
                    let numel = items.get(4).and_then(|x| x.as_i64()).unwrap_or(0);
                    Ok(PValue::Storage { type_name, key, numel })
                } else {
                    Err(UlError::Msg(format!("pickle: unknown persistent kind '{kind}'")))
                }
            }
            _ => Err(UlError::Msg("pickle: bad BINPERSID payload".into())),
        }
    }
}
