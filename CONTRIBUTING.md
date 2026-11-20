# Contributing to AMDA

Thank you for your interest in contributing to **AMDA**.

## Clean-Room Policy

To protect users and contributors, AMDA is developed under clean-room engineering practices (without claiming formal third-party legal certification). Contributors **must never** copy, decompile, or derive code from proprietary NVIDIA CUDA, cuBLAS, cuBLASLt, NVML, or closed-source vendor headers, binaries, or internal documentation.

All interfaces, lowering passes, kernel descriptors, and runtime shims must be original implementations derived strictly from public specifications (PTX ISA documentation, ELF64/AMDHSA specifications, Linux `/dev/kfd` UAPI headers), publicly documented open-source compiler ABI conventions (such as LLVM/Clang), and AMDA's own clean-room C headers in `crates/*/include/`.

## Licensing & Contributor Rights (CLA Notice)

AMDA's open-core workspace crates are licensed under the **Apache License, Version 2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

By submitting a pull request, you certify that:
1. Your contribution is original work (or properly attributed under compatible open-source terms in compliance with our clean-room policy) that you have the legal right to submit under the Apache License, Version 2.0.
2. You understand and acknowledge that commercial or enterprise licensing of AMDA may require signing a Contributor License Agreement (CLA) prior to merging external contributions.

## Pre-PR Verification

Before opening a pull request, please ensure the full workspace builds and passes with zero warnings:

```bash
RUSTFLAGS="-D warnings" cargo test --workspace
RUSTFLAGS="-D warnings" cargo check --workspace --target x86_64-unknown-linux-gnu
cargo clippy --workspace --all-targets -- -D warnings
```
