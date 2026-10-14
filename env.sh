#!/usr/bin/env bash
# 构建环境：指向本机已就位的 libtorch-cu128-2.11（与 tch 0.24 硬配对）。
# 用法：source env.sh && cargo build --release
export LIBTORCH="E:\\libs\\libtorch-cu128-2.11"
export PATH="/e/libs/libtorch-cu128-2.11/lib:$PATH"
