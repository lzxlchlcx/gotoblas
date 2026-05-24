## 1. 驱动循环重排

- [ ] 1.1 调整 `src/driver/gemm_driver.c` 中双精度 driver 的循环顺序，使 `col1` 不再位于 `row1` 外层
- [ ] 1.2 调整 `src/driver/gemm_driver.c` 中单精度 driver 的循环顺序，保持与双精度实现一致
- [ ] 1.3 在 `k0` 内先打包全部 B 列块，再遍历 A 微块并复用 packed A 调用 kernel

## 2. 正确性验证

- [ ] 2.1 编译项目并确认 driver 重排后可正常通过构建
- [ ] 2.2 运行现有 GEMM 正确性测试，确认 NN / NT / TN / TT 与边界用例行为不变
- [ ] 2.3 检查 `C` 子块指针偏移与 packed 缓冲区复用逻辑，确认没有越界或错位写入

## 3. 性能验证

- [ ] 3.1 运行 benchmark，记录重排前后 `pack_A` 占比变化
- [ ] 3.2 运行大矩阵场景 benchmark，确认整体 GFLOPS 有可测提升
- [ ] 3.3 如有需要，补充注释说明新的循环顺序和 packed A 复用意图
