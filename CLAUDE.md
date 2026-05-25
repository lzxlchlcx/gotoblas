1. 如果 遇见 `fatal error: cblas.h: No such file or directory` ，尝试 `LD_LIBRARY_PATH=/home/lzx/miniconda3/envs/main/lib:$LD_LIBRARY_PATH ./test/bench_compare`
2. 该项目的目标是优化 `Myblas` 直到超过 `Openblas` 
3. 文档放入 `Docs` 目录
4. `Docs/实验日志.md` 主要记录对代码的修改和效果