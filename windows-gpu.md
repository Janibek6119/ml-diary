# Meh, for most of processes here CPU must be anough anyway

https://www.tensorflow.org/install/source#gpu

| Version | Python version | Compiler | Build tools | cuDNN | CUDA |
|---------|----------------|----------|-------------|-------|------|
| tensorflow-2.13.0 | 3.8-3.11 | Clang 16.0.0 | Bazel 5.3.0 | 8.6 | 11.8 |
| tensorflow-2.12.0 | 3.8-3.11 | GCC 9.3.1 | Bazel 5.3.0 | 8.6 | 11.8 |
| tensorflow-2.11.0 | 3.7-3.10 | GCC 9.3.1 | Bazel 5.3.0 | 8.1 | 11.2 |
| tensorflow-2.10.0 | 3.7-3.10 | GCC 9.3.1 | Bazel 5.1.1 | 8.1 | 11.2 |


So, Option 1 is just `tensorflow-cpu==2.10` & `tensorflow-directml-plugin`
Option 2 is as in this video: https://www.youtube.com/watch?v=Zn6Lp0xaXj4:
VS 2019 (minimal setup)
`tensorflow=2.10`
https://developer.nvidia.com/cuda-11.2.2-download-archive
https://developer.nvidia.com/rdp/cudnn-archive - "Download cuDNN v8.1.1 (Feburary 26th, 2021), for CUDA 11.0,11.1 and 11.2"