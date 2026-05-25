# TensorCustomPlugins

TensorRT 自定义插件仓库。逐步优化算子实现，记录优化过程。

## 结构

```
src/
├── layernorm/            # LayerNorm 插件
│   ├── layernorm_plugin.h
│   ├── layernorm_plugin.cpp   # TRT 接口实现 (IPluginV2DynamicExt)
│   ├── layernorm_kernel.cu    # CUDA kernel 基线实现
│   ├── layernorm_kernels.h    # kernel 声明
│   └── registry.cpp           # 插件注册
tests/
├── test_correctness.cpp       # 精度测试 (vs CPU)
├── bench_layernorm.cpp        # 性能基准测试
docs/
└── optimization_notes.md      # 优化记录
```

## 编译

```bash
bash scripts/build.sh
```

手动编译:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 运行测试

```bash
# 正确性测试
./build/tests/test_correctness

# 性能测试
./build/tests/bench_layernorm
```

## 使用插件

```bash
trtexec --plugins=build/src/layernorm/liblayernorm_plugin.so \
        --onnx=model.onnx --saveEngine=model.engine
```
