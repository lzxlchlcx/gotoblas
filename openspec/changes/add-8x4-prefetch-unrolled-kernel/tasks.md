## 1. 准备与基线

- [ ] 1.1 记录当前 `8x6` baseline 配置、源码状态和已知 benchmark 数据
- [ ] 1.2 确认 `src/config/haswell.h` 中 `MR/NR` 切换方式，并保留可快速回退到 `8x6` 的路径
- [ ] 1.3 生成当前 `8x6` x86_64 AVX2 汇编，记录主循环寄存器分配、`alpha` broadcast 位置和是否存在 YMM spill

## 2. 实现 8x4 fast path

- [ ] 2.1 在 `src/kernel/avx2/dgemm_kernel.c` 新增 `#elif MR == 8 && NR == 4` fast path 分支
- [ ] 2.2 实现 `8x4` accumulator 布局：4 列、每列 rows 0-3/4-7 两个 YMM，共 8 个 accumulator
- [ ] 2.3 保持 kernel 接口与 `C += alpha * A * B` 语义不变，完整 tile 之外继续使用现有 fallback
- [ ] 2.4 将 `alpha` broadcast 放在 `p` loop 之后，避免 `valpha` 成为主循环 live value
- [ ] 2.5 修改 `src/config/haswell.h` 以便实验性启用 `MR=8, NR=4`

## 3. 2x p-loop unroll

- [ ] 3.1 为 `8x4` kernel 实现 2x `p` loop unrolled 主体
- [ ] 3.2 为 `k` 为奇数的情况实现 1x tail 处理路径
- [ ] 3.3 控制 unroll 版本中的 B broadcast 生命周期，避免一次保留过多 B 临时变量
- [ ] 3.4 控制 unroll 版本中的 A preload 生命周期，避免 next-A 预取导致 accumulator spill

## 4. 预取流水实验

- [ ] 4.1 实验 next-B broadcast：在不引入 YMM spill 的前提下提前生成下一列或下一迭代 B broadcast
- [ ] 4.2 实验 next-A preload：在当前 FMA 序列执行期间提前加载下一组 A vector
- [ ] 4.3 分别保留可比较的实现状态或记录 diff：基础 `8x4`、`8x4 + 2x unroll`、`8x4 + next-B`、`8x4 + next-A`
- [ ] 4.4 若 next-B 或 next-A 触发主循环 YMM spill，记录原因并回退该实验子方案

## 5. 汇编检查

- [ ] 5.1 生成 `8x4` x86_64 AVX2 汇编：`clang -target x86_64-apple-darwin -O2 -Wall -Isrc -D__AVX2__ -mavx2 -mfma -S src/kernel/avx2/dgemm_kernel.c`
- [ ] 5.2 检查 `p` loop 内是否存在针对 `%rsp` 或 `%rbp` 的 YMM `vmovapd/vmovupd` spill/reload
- [ ] 5.3 检查 `alpha` 的 `vbroadcastsd` 是否位于 `p` loop 之后
- [ ] 5.4 检查 FMA、B broadcast、A load 的排列，记录 next-B/next-A 是否真的提前

## 6. 正确性与性能验证

- [ ] 6.1 运行 `make test`，确保通用正确性测试通过
- [ ] 6.2 在 x86_64 AVX2 环境运行 DGEMM 正确性测试，覆盖 `k` 偶数和奇数、完整 tile 和边界 tile
- [ ] 6.3 在 i9-13900K 或等价 x86_64 AVX2 环境运行 `8x6` baseline benchmark：1T/4T，n=512/1024/2048
- [ ] 6.4 在相同环境运行 `8x4` 实验版本 benchmark：1T/4T，n=512/1024/2048
- [ ] 6.5 对比 GFLOPS、kernel 占比和整体耗时，判断 `8x4` 是否抵消了 kernel 调用次数增加和 A 复用下降

## 7. 文档与结论

- [ ] 7.1 更新 `Docs/实验日志.md`，记录实现方案、汇编检查结果、benchmark 数据和结论
- [ ] 7.2 如有必要，更新 `Docs/report/avx2-dgemm-assembly-kernel-notes.md` 中关于 `8x4` 寄存器预算与流水化实验的分析
- [ ] 7.3 根据 benchmark 结果决定默认配置保持 `8x6` 还是切换到 `8x4`
- [ ] 7.4 若 `8x4` 未采用默认配置，保留实验结论并确保代码可回退或不影响默认路径
