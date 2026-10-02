# vLLM Code Wiki

> 本文档是 vLLM 项目源码的结构化技术 Wiki，面向需要理解、修改或扩展 vLLM 代码的开发者。
> 目标仓库版本：基于当前 `/workspace` 检出的 main 分支代码。

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [目录结构与模块职责](#3-目录结构与模块职责)
4. [核心模块详解与关键类/函数](#4-核心模块详解与关键类函数)
5. [依赖关系](#5-依赖关系)
6. [构建与运行方式](#6-构建与运行方式)
7. [扩展点与插件机制](#7-扩展点与插件机制)
8. [关键数据流](#8-关键数据流)

---

## 1. 项目概述

### 1.1 项目定位

**vLLM** 是一个高吞吐、低内存占用的大语言模型（LLM）推理与服务引擎，由 UC Berkeley Sky Computing Lab 发起，现已演化为社区驱动项目。

- 核心创新：**PagedAttention** —— 借鉴虚拟内存分页思想管理 Attention 的 KV Cache，实现近乎零内存碎片、高批处理吞吐。
- 关键能力：连续批处理（Continuous Batching）、张量/流水线/数据/专家并行、量化（GPTQ/AWQ/AutoRound/FP8 等）、投机解码、分块 Prefill、Prefix Caching、多 LoRA、多模态输入等。
- 硬件支持：NVIDIA GPU、AMD CPU/GPU、Intel CPU/GPU、PowerPC、Arm、TPU，以及 Intel Gaudi、IBM Spyre、华为昇腾等插件硬件。

### 1.2 顶层包导出

[vllm/\_\_init\_\_.py](file:///workspace/vllm/__init__.py) 通过懒加载（`__getattr__`）暴露核心公共 API：

| 导出符号 | 来源 | 用途 |
| --- | --- | --- |
| `LLM` | `vllm.entrypoints.llm` | 离线推理主入口 |
| `LLMEngine` / `AsyncLLMEngine` | `vllm.engine.*` | 同步/异步引擎 |
| `EngineArgs` / `AsyncEngineArgs` | `vllm.engine.arg_utils` | 引擎参数构造 |
| `SamplingParams` / `PoolingParams` | `vllm.sampling_params` / `vllm.pooling_params` | 采样/池化参数 |
| `ModelRegistry` | `vllm.model_executor.models` | 模型注册表 |
| `RequestOutput` 等输出类型 | `vllm.outputs` | 推理结果数据结构 |
| `PromptType` / `TextPrompt` / `TokensPrompt` | `vllm.inputs` | 输入类型 |

---

## 2. 整体架构

### 2.1 V1 多进程架构

vLLM V1 采用**多进程解耦**架构，将 HTTP 服务、调度/KV 管理、GPU 执行分离到不同进程，以最大化吞吐并隔离故障。

| 进程类型 | 数量 | 职责 |
| --- | --- | --- |
| **API Server 进程** | `A`（默认等于 DP） | 处理 HTTP 请求（OpenAI 兼容 API）、输入预处理（分词、多模态加载）、流式返回结果；通过 **ZMQ Socket** 与 Engine Core 通信 |
| **Engine Core 进程** | `DP`（默认 1） | 运行调度器（Scheduler）、管理 KV Cache、协调 GPU Worker 执行；运行 busy loop 持续调度 |
| **GPU Worker 进程** | `N = DP × PP × TP`（每 GPU 一个） | 加载模型权重、执行 forward pass、管理 GPU 显存 |
| **DP Coordinator 进程** | 仅当 `DP > 1` 时 1 个 | 跨 DP 秩的负载均衡与 MoE 同步前向协调 |

**进程总数公式**：`A + DP + N (+ 1 if DP > 1)`

例如：
- 单节点 4 GPU（`vllm serve -tp=4`）：1 API + 1 Core + 4 Worker = **6 进程**
- 8 GPU 数据并行（`vllm serve -tp=2 -dp=4`）：4 API + 4 Core + 8 Worker + 1 Coord = **17 进程**

### 2.2 核心类层次

```
LLMEngine (引擎)
  └─ EngineCore (内核调度循环)
       ├─ Scheduler (调度器)
       │    ├─ KVCacheManager (KV Cache 分页管理)
       │    └─ EncoderCacheManager (Encoder KV)
       ├─ Executor (执行器, 抽象)
       │    ├─ MultiprocExecutor (多进程)
       │    ├─ RayDistributedExecutor (Ray)
       │    └─ UniProcExecutor (单进程)
       │         └─ GPUWorker (GPU 工作进程)
       │              └─ GPUModelRunner (模型运行器)
       │                   └─ Model (torch.nn.Module)
       └─ StructuredOutputManager (结构化输出)
```

### 2.3 设计原则

1. **配置中心化**：所有配置封装在 `VllmConfig` 对象中，贯穿整个类层次，新增功能只需扩展配置类，无需修改构造函数签名。
2. **统一模型构造签名**：所有模型构造函数为 `__init__(self, *, vllm_config: VllmConfig, prefix: str = "")`（keyword-only），便于模型组合与统一加载。
3. **初始化时分片/量化**：张量并行的权重切分与量化在模型初始化期间完成，避免先加载全量权重再分片的内存爆炸。
4. **可插拔设计**：Attention Backend、Model Loader、Custom Op、Scheduler、IO Processor 等均支持注册/替换。

---

## 3. 目录结构与模块职责

```
/workspace
├── vllm/                        # 核心 Python 包
│   ├── config/                  # 配置对象体系（VllmConfig 及子配置）
│   ├── engine/                  # 旧版引擎（LLMEngine/AsyncLLMEngine）
│   ├── v1/                      # V1 引擎核心
│   │   ├── engine/              # EngineCore, LLMEngine, 输入/输出处理器
│   │   ├── core/                # Scheduler, KVCacheManager, BlockPool
│   │   ├── executor/            # 执行器（多进程/Ray/单进程）
│   │   ├── worker/              # GPU/CPU/TPU Worker 与 ModelRunner
│   │   ├── attention/           # Attention Backend 体系
│   │   ├── sample/              # 采样器（拒绝采样等）
│   │   ├── spec_decode/         # 投机解码（Eagle/Medusa/Ngram）
│   │   ├── metrics/             # 指标采集与 Prometheus
│   │   ├── kv_offload/          # KV Cache 卸载（CPU/Disk）
│   │   ├── structured_output/   # 结构化输出（Outlines/llguidance/xgrammar）
│   │   └── pool/                # 池化（Embedding/Score）
│   ├── model_executor/          # 模型执行层
│   │   ├── models/              # 300+ 模型实现 + 注册表
│   │   ├── layers/              # 神经网络层（attention/moe/linear/...）
│   │   ├── model_loader/        # 权重加载器（HF/ BitsAndBytes/ GGUF/...）
│   │   └── offloader/           # 权重卸载（Prefetch/UVA）
│   ├── distributed/             # 分布式通信
│   │   ├── parallel_state.py    # 并行组管理
│   │   ├── device_communicators/# NCCL/All-Reduce/P2P 通信
│   │   ├── kv_transfer/         # KV 传输（Disaggregated Prefill）
│   │   ├── ec_transfer/         # Encoder-Decoder 传输
│   │   └── weight_transfer/     # 权重传输
│   ├── entrypoints/             # 对外入口
│   │   ├── llm.py               # LLM 类（离线推理）
│   │   ├── openai/              # OpenAI 兼容 API Server
│   │   ├── cli/                 # vllm CLI
│   │   └── grpc_server.py       # gRPC 服务
│   ├── multimodal/              # 多模态（图像/音频/视频）
│   ├── lora/                    # LoRA 多适配器支持
│   ├── compilation/             # torch.compile / CUDA Graph
│   ├── platforms/               # 硬件平台抽象
│   ├── plugins/                 # 插件系统
│   ├── tokenizers/              # 分词器适配
│   ├── reasoning/               # 推理内容解析（Thinking 标签等）
│   ├── tool_parsers/            # 工具调用解析
│   ├── inputs/                  # 输入数据结构与预处理
│   ├── utils/                   # 通用工具
│   └── tracing/                 # OpenTelemetry 追踪
├── csrc/                        # C++/CUDA 内核源码
│   ├── attention/               # PagedAttention v1/v2 内核
│   ├── moe/                     # MoE 路由/GEMM 内核
│   ├── cutlass_extensions/      # CUTLASS 扩展
│   ├── cpu/                     # CPU 内核
│   └── quantization/            # 量化内核
├── tests/                       # 测试套件
├── benchmarks/                  # 基准测试脚本
├── docs/                        # 官方文档
├── requirements/                # 依赖声明
├── docker/                      # Dockerfile
├── CMakeLists.txt               # C++ 扩展构建
└── setup.py                     # Python 包构建
```

---

## 4. 核心模块详解与关键类/函数

### 4.1 配置体系 `vllm/config/`

配置体系是 vLLM 的骨架，所有模块共享一个 `VllmConfig`。

**`VllmConfig`** — [vllm/config/vllm.py](file:///workspace/vllm/config/vllm.py)

- 聚合所有子配置，作为引擎级全局状态在类层次间传递。
- 关键子配置：

| 子配置类 | 文件 | 职责 |
| --- | --- | --- |
| `ModelConfig` | `config/model.py` | 模型架构、dtype、max_model_len、HF 配置 |
| `CacheConfig` | `config/cache.py` | KV Cache：block_size、GPU/CPU 空间、量化 dtype、Prefix Caching |
| `ParallelConfig` | `config/parallel.py` | TP/PP/DP/EP/CP 并行度、执行后端（mp/ray/uni） |
| `SchedulerConfig` | `config/scheduler.py` | 调度策略、max_num_batched_tokens、chunked prefill、策略 |
| `AttentionConfig` | `config/attention.py` | Attention backend 选择、滑动窗口 |
| `LoRAConfig` | `config/lora.py` | 多 LoRA 配置 |
| `SpeculativeConfig` | `config/speculative.py` | 投机解码配置 |
| `CompilationConfig` | `config/compilation.py` | torch.compile / CUDA Graph / 优化级别 O0~O3 |
| `DeviceConfig` | `config/device.py` | 设备类型 |
| `LoadConfig` | `config/load.py` | 权重加载格式 |
| `ObservabilityConfig` | `config/observability.py` | 日志、指标、追踪 |
| `KVTransferConfig` | `config/kv_transfer.py` | KV 传输（分离式 Prefill） |
| `MultiModalConfig` | `config/multimodal.py` | 多模态限制 |

**优化级别 `OptimizationLevel`**：
- `O0`：无优化，立即启动
- `O1`：Dynamo+Inductor 编译 + Piecewise CUDA Graph
- `O2`：O1 + 完整 CUDA Graph
- `O3`：当前等同 O2

### 4.2 V1 引擎核心 `vllm/v1/engine/`

#### `EngineCore` — [vllm/v1/engine/core.py](file:///workspace/vllm/v1/engine/core.py)

引擎内核，运行核心调度循环。

**关键方法**：
- `__init__`：初始化 ModelExecutor、KV Cache、Scheduler、StructuredOutputManager、Batch Queue。
- `step()`：单步执行 —— 调度 → 执行模型 → 采样 → 更新调度器状态。
  ```python
  def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
      scheduler_output = self.scheduler.schedule()
      future = self.model_executor.execute_model(scheduler_output, non_block=True)
      grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
      model_output = future.result()
      if model_output is None:
          model_output = self.model_executor.sample_tokens(grammar_output)
      engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
      return engine_core_outputs, ...
  ```
- `step_with_batch_queue()`：流水线并行场景下的异步调度/执行队列，消除流水线气泡。

#### `LLMEngine` (V1) — [vllm/v1/engine/llm_engine.py](file:///workspace/vllm/v1/engine/llm_engine.py)

V1 版引擎对外接口，包装 `EngineCore`，管理输入/输出处理器、ZMQ 客户端、指标日志。

#### `InputProcessor` / `OutputProcessor` — [vllm/v1/engine/](file:///workspace/vllm/v1/engine/)

- `InputProcessor`：将原始请求转换为 `EngineCoreRequest`（分词、多模态预处理）。
- `OutputProcessor`：将 `EngineCoreOutputs` 转换为用户可读的 `RequestOutput`（detokenize、stop reason 判定）。

#### `Request` — [vllm/v1/request.py](file:///workspace/vllm/v1/request.py)

单个推理请求的运行时表示，包含 prompt_token_ids、sampling_params、arrival_time、状态（`RequestStatus`）、LoRA 引用等。

### 4.3 调度器 `vllm/v1/core/sched/`

#### `Scheduler` — [vllm/v1/core/sched/scheduler.py](file:///workspace/vllm/v1/core/sched/scheduler.py)

核心调度器，决定每个迭代处理哪些请求及每个请求处理多少 token。

**关键方法**：
- `schedule() -> SchedulerOutput`：每个迭代调用一次，产出调度决策（`{req_id: num_tokens}`）。支持新请求 Prefill、自回归解码、分块 Prefill、Prefix Caching、投机解码等。
- `update_from_output(scheduler_output, model_output)`：根据模型输出更新请求状态、检测 stop、释放 KV block。
- `get_grammar_bitmask(scheduler_output)`：获取结构化输出的 grammar 掩码。

#### `SchedulerInterface` — [vllm/v1/core/sched/interface.py](file:///workspace/vllm/v1/core/sched/interface.py)

调度器抽象基类，定义 `schedule`、`update_from_output`、`get_grammar_bitmask` 等接口，并提供 `PauseState`（UNPAUSED/PAUSED_NEW/PAUSED_ALL）暂停机制。

#### `SchedulerOutput` — [vllm/v1/core/sched/output.py](file:///workspace/vllm/v1/core/sched/output.py)

调度结果，包含：
- `scheduled_new_reqs` / `scheduled_cached_reqs` / `scheduled_running_reqs`
- `total_num_scheduled_tokens`
- `num_allocated_slots`（KV slot 数）
- `block_tables`（KV block 映射表）

#### 调度策略
通过 `SchedulerConfig.scheduling_policy` 配置，支持 FCFS、Priority 等策略，由 `create_request_queue` 工厂创建 `RequestQueue`。

### 4.4 KV Cache 管理 `vllm/v1/core/`

#### `KVCacheManager` — [vllm/v1/core/kv_cache_manager.py](file:///workspace/vllm/v1/core/kv_cache_manager.py)

PagedAttention 的核心实现，将 KV Cache 划分为固定大小的 block（页）进行管理。

**关键概念**：
- `KVCacheBlock`：KV Cache 的最小分配单元，含 `block_id`、引用计数、hash（用于 Prefix Caching）。
- `KVCacheBlocks`：分配结果，作为 Scheduler 与 KVCacheManager 之间的接口。
- `get_block_ids()`：将 block 列表转为 block_id 元组，供模型层使用。
- 支持 **Prefix Caching**：通过 block hash 命中已计算的前缀 KV，避免重复计算。

#### `BlockPool` — [vllm/v1/core/block_pool.py](file:///workspace/vllm/v1/core/block_pool.py)

管理所有 KV block 的池，处理分配、释放、复用。

#### `KVCacheConfig` / `KVCacheSpec` — [vllm/v1/kv_cache_interface.py](file:///workspace/vllm/v1/kv_cache_interface.py)

定义 KV Cache 的格式规范（block_size、page_size_bytes、max_memory_usage_bytes），支持多层多组 KV Cache（如 MLA、Encoder-Decoder）。

### 4.5 执行器 `vllm/v1/executor/`

#### `Executor` (抽象) — [vllm/v1/executor/abstract.py](file:///workspace/vllm/v1/executor/abstract.py)

执行器抽象基类，负责在单设备或多设备上执行模型。

**关键静态方法 `get_class(vllm_config)`**：根据 `ParallelConfig.distributed_executor_backend` 选择执行器：
- `"mp"` → `MultiprocExecutor`
- `"ray"` → `RayDistributedExecutor`
- `"uni"` → `UniProcExecutor`
- `"external_launcher"` → `ExecutorWithExternalLauncher`
- 字符串 → `resolve_obj_by_qualname` 动态导入

#### `MultiprocExecutor` — [vllm/v1/executor/multiproc_executor.py](file:///workspace/vllm/v1/executor/multiproc_executor.py)

多进程执行器，为每个 GPU spawn 一个 `GPUWorker` 进程，通过 multiprocessing 通信。支持 PP 的 batch queue。

**关键类/方法**：
- `FutureWrapper`：聚合多 worker 的 Future。
- `execute_model(scheduler_output, non_block)`：异步发起模型执行，返回 Future。
- `sample_tokens(grammar_output)`：异步采样。

#### `RayDistributedExecutor` / `UniProcExecutor`

- Ray 执行器使用 Ray 编排多节点 Worker。
- UniProc 执行器在单进程内运行，用于调试/单卡场景。

### 4.6 Worker 与 Model Runner `vllm/v1/worker/`

#### `GPUWorker` — [vllm/v1/worker/gpu_worker.py](file:///workspace/vllm/v1/worker/gpu_worker.py)

每个 GPU 一个 Worker 进程，负责：
- 初始化分布式环境（`init_distributed_environment`、`ensure_model_parallel_initialized`）
- 加载模型权重（通过 `get_model_loader`）
- 执行 forward pass
- 管理 CUDA Graph 捕获

**关键方法**：
- `init_device()` / `load_model()`：设备与模型初始化
- `execute_model(scheduler_output)`：准备输入张量、执行模型 forward
- `determine_num_available_blocks()`：探测可用 KV block 数

#### `GPUModelRunner` — [vllm/v1/worker/gpu_model_runner.py](file:///workspace/vllm/v1/worker/gpu_model_runner.py)

Worker 内的模型运行器，包含大量执行逻辑：
- 准备输入张量（input_ids、positions、attn_metadata）
- CUDA Graph 捕获与重放（`CUDAGraphWrapper`）
- LoRA 映射、多模态特征注入
- 中间张量通信（PP）

#### `CPUWorker` / `CPUModelRunner` — [vllm/v1/worker/cpu_worker.py](file:///workspace/vllm/v1/worker/cpu_worker.py)

CPU 平台的 Worker 与 ModelRunner。

### 4.7 Attention Backend 体系 `vllm/v1/attention/`

#### `AttentionBackend` (抽象) — [vllm/v1/attention/backend.py](file:///workspace/v1/attention/backend.py)

所有 Attention 后端的基类。

**关键属性/方法**：
- `get_name()`：后端名称
- `get_impl_cls()`：返回实现类 `AttentionImplBase`
- `get_builder_cls()`：返回 metadata builder
- `get_kv_cache_shape()`：定义 KV Cache 张量形状
- `supported_dtypes` / `supported_kv_cache_dtypes`：支持的 dtype
- `forward_includes_kv_cache_update`：forward 是否包含 KV 更新

**已注册的 Backend**（部分）：

| Backend | 文件 | 适用场景 |
| --- | --- | --- |
| `FlashAttentionBackend` | `backends/flash_attn.py` | NVIDIA GPU 默认 |
| `FlashInferBackend` | `backends/flashinfer.py` | FlashInfer 内核 |
| `TritonAttentionBackend` | `backends/triton_attn.py` | Triton 实现 |
| `FlexAttentionBackend` | `backends/flex_attention.py` | PyTorch FlexAttention |
| `CPUAttentionBackend` | `backends/cpu_attn.py` | CPU |
| `RocmAttentionBackend` | `backends/rocm_attn.py` | AMD ROCm |
| `Mamba1/2AttentionBackend` | `backends/mamba*_attn.py` | Mamba SSM |
| `GDNAttentionBackend` | `backends/gdn_attn.py` | GDN |
| `LinearAttentionBackend` | `backends/linear_attn.py` | 线性注意力 |
| `TreeAttentionBackend` | `backends/tree_attn.py` | 树注意力 |
| `FlashMLASparseBackend` 等 | `backends/mla/` | DeepSeek MLA |

`AttentionType` 枚举区分 `DECODER`、`ENCODER`、`ENCODER_ONLY`、`ENCODER_DECODER`。

### 4.8 模型执行层 `vllm/model_executor/`

#### 模型注册表 `ModelRegistry` — [vllm/model_executor/models/registry.py](file:///workspace/vllm/model_executor/models/registry.py)

管理 300+ 模型架构的注册与查找。
- `register_model(*architectures)`：装饰器注册模型类到 HF architecture name。
- `get_model_cls(model_config)`：根据 HF config 的 architectures 查找模型类。
- 支持通过 `interfaces` / `interfaces_base` 中的 Protocol 进行能力探测（`is_mixture_of_experts`、`supports_multimodal`、`supports_pp` 等）。

#### 模型实现 `vllm/model_executor/models/`

每个模型一个文件，如 `llama.py`、`qwen2_moe.py`、`deepseek_v2.py`、`llava.py` 等。统一构造签名：

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        ...
```

模型按能力分类（见 `interfaces.py`）：
- 文本生成模型、池化模型、MoE 模型、多模态模型、SSM/Mamba 模型、Encoder-Decoder 模型等。

#### 神经网络层 `vllm/model_executor/layers/`

| 子目录/模块 | 职责 |
| --- | --- |
| `attention/` | Attention 层（`Attention`、`MLAAttention`、`CrossAttention`、`EncoderOnlyAttention`） |
| `fused_moe/` | 融合 MoE 层（FusedMoE、router、专家 GEMM、DeepEP/Mori/NIXL prepare-finalize） |
| `linear.py` | 并行线性层（`ColumnParallelLinear`、`RowParallelLinear`、`MergedColumnParallelLinear`） |
| `layernorm.py` | LayerNorm/RMSNorm |
| `activation.py` | 激活函数 |
| `rotary_embedding.py` | RoPE（含 MRoPE、XDRoPE） |
| `vocab_parallel_embedding.py` | 词表并行嵌入 |
| `fla/` | Flash Linear Attention 内核 |
| `logits_processor.py` | Logits 处理 |

#### 权重加载器 `vllm/model_executor/model_loader/`

| Loader | 格式 |
| --- | --- |
| `DefaultModelLoader` | HF safetensors / pt / fastsafetensors / npcache |
| `BitsAndBytesModelLoader` | bitsandbytes 量化 |
| `DummyModelLoader` | 随机权重（调试用） |
| `GGUFModelLoader` | GGUF 格式 |
| `TensorizerLoader` | Tensorizer 流式加载 |
| `ShardedStateLoader` | 分片状态加载 |
| `RunaiModelStreamerLoader` | Runai 流式加载 |

**工厂函数**：
- `get_model_loader(load_config)`：根据 `LoadConfig.load_format` 返回对应 Loader。
- `get_model(*, vllm_config, ...)`：加载并返回模型实例。
- `register_model_loader(load_format)`：装饰器注册自定义 Loader。

#### 自定义算子 `CustomOp` — [vllm/model_executor/custom_op.py](file:///workspace/vllm/model_executor/custom_op.py)

vLLM 自定义 CUDA 算子的注册与分发体系。
- `op_registry`：内置算子注册表
- `op_registry_oot`：Out-of-Tree 算子注册表
- `CustomOp`：按平台分派 `forward_native`/`forward_cuda`/`forward_rocm`/`forward_cpu` 等
- `PluggableLayer`：支持 OOT 整层替换的 `nn.Module`

### 4.9 分布式 `vllm/distributed/`

#### `parallel_state.py` — [vllm/distributed/parallel_state.py](file:///workspace/vllm/distributed/parallel_state.py)

接管 PyTorch 分布式环境，管理所有并行组。

**工作流**：
1. `init_distributed_environment()` — 初始化分布式环境
2. `initialize_model_parallel()` / `ensure_model_parallel_initialized()` — 初始化模型并行组
3. 业务代码
4. `destroy_model_parallel()` / `destroy_distributed_environment()`

**并行组**：
- `get_tp_group()` — 张量并行组
- `get_pp_group()` — 流水线并行组
- `get_dp_group()` — 数据并行组
- `get_ep_group()` — 专家并行组
- `get_cp_group()` — 上下文并行组
- `get_dcp_group()` — 解耦上下文并行组

#### 设备通信器 `vllm/distributed/device_communicators/`

| 模块 | 职责 |
| --- | --- |
| `base_device_communicator.py` | 通信器抽象基类 |
| `cuda_communicator.py` | NCCL CUDA 通信 |
| `custom_all_reduce.py` | 自定义 all-reduce |
| `quick_all_reduce.py` | Quick all-reduce |
| `flashinfer_all_reduce.py` | FlashInfer all-reduce |
| `pynccl.py` | Python NCCL 绑定 |
| `shm_broadcast.py` | 共享内存广播 |
| `symm_mem.py` | 对称内存通信 |
| `all2all.py` | All-to-All（MoE EP） |

#### KV 传输 `vllm/distributed/kv_transfer/`

支持 **Disaggregated Prefill**（Prefill 与 Decode 分离部署）：
- `KVConnectorBase_V1`：KV 连接器抽象
- 支持 Mooncake、NIXL 等连接器
- `KVOutputAggregator`：聚合多 worker 的 KV 输出

### 4.10 入口层 `vllm/entrypoints/`

#### `LLM` 类 — [vllm/entrypoints/llm.py](file:///workspace/vllm/entrypoints/llm.py)

离线推理主接口。

**关键方法**：
- `generate(prompts, sampling_params, ...)`：批量生成
- `chat(messages, ...)`：对话生成
- `embed(prompts, ...)`：Embedding
- `score(prompts, ...)`：打分

#### OpenAI 兼容 API Server — [vllm/entrypoints/openai/api_server.py](file:///workspace/vllm/entrypoints/openai/api_server.py)

基于 FastAPI + uvicorn 实现的 OpenAI 兼容服务。

- `build_async_engine_client()`：构建 AsyncLLMEngine 客户端
- 端点：`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/models`、音频转录、推理内容等
- 支持流式（SSE）、工具调用、结构化输出

#### CLI — [vllm/entrypoints/cli/main.py](file:///workspace/vllm/entrypoints/cli/main.py)

`vllm` 命令行入口，子命令：
- `vllm serve <model>` — 启动 API 服务
- `vllm complete` / `vllm chat` — 命令行交互
- `vllm run-batch` — 批量离线推理
- `vllm collect-env` — 收集环境信息

### 4.11 采样与投机解码

#### 采样器 `vllm/v1/sample/`

- `sampler.py`：核心采样逻辑（top-k、top-p、temperature、repetition penalty 等）
- `rejection_sampler.py`：投机解码的拒绝采样

#### 投机解码 `vllm/v1/spec_decode/`

| 模块 | 草案来源 |
| --- | --- |
| `eagle.py` | EAGLE 草案模型 |
| `medusa.py` | Medusa 多头 |
| `mlp_speculator.py` | MLP Speculator |
| `ngram_proposer.py` / `ngram_proposer_gpu.py` | N-gram 提议 |
| `draft_model.py` | 通用草案模型 |
| `suffix_decoding.py` | 后缀解码 |

### 4.12 编译系统 `vllm/compilation/`

- `compiler_interface.py`：编译接口
- `cuda_graph.py`：CUDA Graph 管理（`CUDAGraphWrapper`）
- `decorators.py`：编译装饰器
- `passes/`：Inductor pass
- `piecewise_backend.py`：Piecewise CUDA Graph 后端

### 4.13 平台抽象 `vllm/platforms/`

`current_platform` 单例，统一硬件平台接口：
- `cuda.py` / `rocm.py` / `cpu.py` / `tpu.py` / `xpu.py`
- 提供设备能力探测、kernel 编译、随机数种子等抽象方法

### 4.14 多模态 `vllm/multimodal/`

- `MULTIMODAL_REGISTRY`：多模态注册表，将 HF processor 适配为 vLLM 输入
- `media/`：图像/音频/视频的加载与特征
- `processing/`：输入处理流水线
- `encoder_budget.py`：多模态编码器 token 预算控制
- `hasher.py`：多模态输入哈希（用于缓存）

### 4.15 LoRA `vllm/lora/`

- `model_manager.py`：多 LoRA 适配器管理
- `lora_model.py`：LoRA 模型包装
- `layers/`：LoRA 版线性层、MoE 层
- `punica_wrapper/`：Punica kernel 封装
- `request.py`：`LoRARequest`

### 4.16 工具 `vllm/utils/`

| 模块 | 职责 |
| --- | --- |
| `torch_utils.py` | 张量工具、随机种子 |
| `mem_utils.py` | 内存分析 |
| `network_utils.py` | ZMQ、IP、端口 |
| `import_utils.py` | 动态导入 |
| `hashing.py` | 哈希函数（Prefix Caching） |
| `gc_utils.py` | GC 控制 |
| `nccl.py` | NCCL 辅助 |
| `collection_utils.py` | 集合工具 |

### 4.17 C++/CUDA 内核 `csrc/`

| 目录 | 内容 |
| --- | --- |
| `attention/` | PagedAttention v1/v2、merge attention、vertical slash |
| `moe/` | MoE top-k、router GEMM、permute/unpermute、DSV3 |
| `cutlass_extensions/` | CUTLASS GEMM 扩展 |
| `cpu/` | CPU attention、MoE、LayerNorm（oneDNN/AMX/NEON） |
| `quantization/` | 量化激活内核 |
| `quickreduce/` | 快速 reduce |
| `rocm/` | ROCm 专用内核 |
| 根目录 | cache、layernorm、pos_encoding、sampler、custom_all_reduce 等 |

入口绑定：[csrc/torch_bindings.cpp](file:///workspace/csrc/torch_bindings.cpp)

---

## 5. 依赖关系

### 5.1 模块依赖拓扑（高层）

```
entrypoints/
  ├── engine/ (LLMEngine)
  │     └── v1/engine/ (EngineCore)
  │           ├── v1/core/sched/ (Scheduler)
  │           │     └── v1/core/ (KVCacheManager)
  │           ├── v1/executor/ (Executor)
  │           │     └── v1/worker/ (GPUWorker → GPUModelRunner)
  │           │           ├── model_executor/models/ (Model)
  │           │           │     └── model_executor/layers/
  │           │           ├── model_executor/model_loader/
  │           │           └── v1/attention/ (AttentionBackend)
  │           └── distributed/ (parallel_state, communicators)
  └── tokenizers/, multimodal/, lora/
```

### 5.2 关键依赖方向

- **配置驱动**：几乎所有模块依赖 `vllm/config/`，`VllmConfig` 自顶向下传递。
- **引擎 → 执行器 → Worker → Model**：严格的自上而下调用链。
- **Model → Layers → CustomOp → csrc**：模型层调用 vLLM 自定义层，最终调用 C++/CUDA 内核。
- **Distributed 贯穿**：`parallel_state`、`device_communicators` 被 Worker、ModelRunner、MoE 层等多处使用。

### 5.3 外部运行时依赖（`requirements/common.txt`）

| 类别 | 依赖 |
| --- | --- |
| 深度学习 | `torch==2.10.0`、`transformers>=4.56.0,<5`、`tokenizers>=0.21.1` |
| 服务 | `fastapi[standard]>=0.115.0`、`uvloop`、`aiohttp`、`openai`、`pyzmq`、`msgspec` |
| 数据/序列化 | `numpy`、`protobuf`、`cloudpickle`、`cbor2`、`gguf>=0.17.0` |
| 结构化输出 | `outlines_core==0.2.11`、`lm-format-enforcer==0.11.3`、`xgrammar==0.1.29`、`llguidance` |
| 多模态 | `pillow`、`opencv-python-headless`、`einops` |
| 指标/追踪 | `prometheus_client`、`prometheus-fastapi-instrumentator`、`opentelemetry-*` |
| 量化 | `compressed-tensors==0.13.0` |
| 其他 | `sentencepiece`、`tiktoken`、`mistral_common`、`mcp`、`blake3`、`py-cpuinfo` |

平台特定依赖见 `requirements/cuda.txt`、`requirements/cpu.txt`、`requirements/rocm.txt`、`requirements/tpu.txt`、`requirements/xpu.txt`。

### 5.4 构建依赖

`pyproject.toml` build-system：`cmake>=3.26.1`、`ninja`、`torch==2.10.0`、`setuptools`、`setuptools-scm`。

---

## 6. 构建与运行方式

### 6.1 环境要求

- Python 3.10 ~ 3.13
- Linux（含 WSL）或 macOS
- 可选：CUDA / ROCm / TPU / XPU 驱动与 SDK

### 6.2 安装

**pip 安装（预编译）**：
```bash
pip install vllm
```

**从源码安装**（见 `AGENTS.md`）：
```bash
# 1. 创建虚拟环境
uv venv --python 3.12
source .venv/bin/activate

# 2. 仅 Python 改动
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# 3. 含 C/C++ 改动（重新编译内核）
uv pip install -e .
```

**目标设备自动检测**（[setup.py](file:///workspace/setup.py)）：
- 根据 `torch.version.cuda` / `torch.version.hip` 自动选择 `cuda` / `rocm` / `cpu`
- 可通过 `VLLM_TARGET_DEVICE` 环境变量强制指定

### 6.3 构建系统

- **C++ 扩展**：使用 CMake + Ninja 构建，入口 [CMakeLists.txt](file:///workspace/CMakeLists.txt)
- **Python 打包**：setuptools + setuptools-scm（版本从 git tag 推导）
- **自定义构建类**：`cmake_build_ext`（`setup.py`），支持 sccache/ccache、`MAX_JOBS` 并行编译

### 6.4 运行

#### 离线推理（LLM 类）
```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(
    ["Hello, my name is"],
    SamplingParams(temperature=0.8, top_p=0.95),
)
```

#### 在线服务（OpenAI 兼容 API）
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
# 或
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct
```

#### 分布式推理
```bash
# 张量并行
vllm serve model --tensor-parallel-size 4

# 数据并行
vllm serve model --tensor-parallel-size 2 --data-parallel-size 4

# 流水线并行
vllm serve model --pipeline-parallel-size 2
```

#### 关键 CLI 参数

| 参数 | 说明 |
| --- | --- |
| `--model` | 模型名称或路径 |
| `--tensor-parallel-size` / `-tp` | 张量并行度 |
| `--pipeline-parallel-size` / `-pp` | 流水线并行度 |
| `--data-parallel-size` / `-dp` | 数据并行度 |
| `--dtype` | 模型 dtype（auto/float16/bfloat16/float32） |
| `--max-model-len` | 最大上下文长度 |
| `--gpu-memory-utilization` | GPU 显存利用率 |
| `--quantization` | 量化方法（awq/gptq/fp8/...） |
| `--enforce-eager` | 禁用 CUDA Graph |
| `--optimization-level` | O0/O1/O2/O3 |
| `--api-key` | API 密钥 |
| `--port` | 服务端口 |

### 6.5 运行测试

```bash
# 安装测试依赖
uv pip install -r requirements/test.txt

# 运行特定测试
pytest tests/path/to/test.py -v -s -k test_name

# 运行某目录全部测试
pytest tests/path/to/dir -v -s
```

### 6.6 Lint 与类型检查

```bash
pre-commit run --all-files          # 全部
pre-commit run ruff-check --all-files
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

---

## 7. 扩展点与插件机制

### 7.1 注册新模型

在 `vllm/model_executor/models/` 下新建文件，使用 `@ModelRegistry.register_model("MyModelArch")` 装饰器。模型构造函数须遵循 `__init__(self, *, vllm_config, prefix="")`。

### 7.2 注册自定义 Model Loader

```python
from vllm.model_executor.model_loader import register_model_loader

@register_model_loader("my_format")
class MyLoader(BaseModelLoader):
    def download_model(self): ...
    def load_weights(self, model, ...): ...
```

### 7.3 注册自定义 Attention Backend

继承 `AttentionBackend`，实现 `get_name`、`get_impl_cls`、`get_builder_cls`、`get_kv_cache_shape` 等方法。

### 7.4 自定义 CustomOp / PluggableLayer

- `CustomOp`：按平台分派 forward，通过 `op_registry` 注册。
- `PluggableLayer`：支持 OOT 整层替换，通过 `op_registry_oot` 注册。

### 7.5 插件系统 `vllm/plugins/`

- **IO Processor 插件**（`vllm/plugins/io_processors/`）：自定义输入/输出处理器。
- **LoRA Resolver 插件**（`vllm/plugins/lora_resolvers/`）：LoRA 权重解析（filesystem、HF Hub）。
- 通过 `entry-points`（`pyproject.toml` 中 `vllm.general_plugins`）注册通用插件。

### 7.6 自定义 Scheduler

继承 `SchedulerInterface`，实现 `schedule`、`update_from_output` 等方法，通过 `SchedulerConfig` 配置使用。

### 7.7 硬件平台扩展

在 `vllm/platforms/` 下新增平台实现（参考 `cuda.py`、`cpu.py`），实现 `Platform` 接口。

---

## 8. 关键数据流

### 8.1 请求处理主流程（在线服务）

```
Client HTTP Request
  │
  ▼
[API Server Process]
  ├─ FastAPI 路由接收
  ├─ InputProcessor 分词 / 多模态预处理
  ├─ 转换为 EngineCoreRequest
  └─ ZMQ 发送 ──────────────────────────┐
                                          │
[Engine Core Process]                    │
  ├─ ZMQ 接收请求                        │
  ├─ Scheduler.schedule()                │
  │   ├─ 分配/复用 KV blocks            │
  │   └─ 产出 SchedulerOutput           │
  ├─ Executor.execute_model()  ── ZMQ/RPC ──► [GPU Worker Process]
  │                                          ├─ GPUModelRunner 准备输入
  │                                          ├─ Model.forward()
  │                                          ├─ (CUDA Graph 重放)
  │                                          └─ 返回 hidden_states/logits
  ├─ sample_tokens()
  ├─ Scheduler.update_from_output()
  └─ EngineCoreOutputs ── ZMQ ──────────────┐
                                             │
[API Server Process]                         │
  ├─ OutputProcessor detokenize
  ├─ stop reason 判定
  └─ SSE 流式返回 Client
```

### 8.2 单步调度循环（EngineCore.step）

```
has_requests?
  │ no  → return {}, False
  │ yes
  ▼
scheduler.schedule()  → SchedulerOutput
  │
  ▼
model_executor.execute_model(non_block=True)  → Future
  │
  ▼
scheduler.get_grammar_bitmask()  → GrammarOutput | None
  │
  ▼
future.result()  → ModelRunnerOutput
  │ None? → sample_tokens(grammar_output)
  ▼
scheduler.update_from_output(scheduler_output, model_output)
  │
  ▼
return EngineCoreOutputs, model_executed
```

### 8.3 KV Cache 分配（PagedAttention）

1. 请求到达，计算 prompt token 的 block hash。
2. KVCacheManager 查找 hash 命中的已有 block（Prefix Caching）。
3. 未命中部分从 BlockPool 分配新 block。
4. block_table 记录请求 → block 的映射。
5. 模型 forward 时通过 block_table 访问分散的 KV block。
6. 请求结束后释放 block（引用计数归零时归还 BlockPool）。

---

## 附录 A：关键文件速查表

| 关注点 | 文件路径 |
| --- | --- |
| 引擎内核 | [vllm/v1/engine/core.py](file:///workspace/vllm/v1/engine/core.py) |
| V1 引擎接口 | [vllm/v1/engine/llm_engine.py](file:///workspace/vllm/v1/engine/llm_engine.py) |
| 调度器 | [vllm/v1/core/sched/scheduler.py](file:///workspace/vllm/v1/core/sched/scheduler.py) |
| KV Cache 管理 | [vllm/v1/core/kv_cache_manager.py](file:///workspace/vllm/v1/core/kv_cache_manager.py) |
| 多进程执行器 | [vllm/v1/executor/multiproc_executor.py](file:///workspace/vllm/v1/executor/multiproc_executor.py) |
| GPU Worker | [vllm/v1/worker/gpu_worker.py](file:///workspace/vllm/v1/worker/gpu_worker.py) |
| GPU Model Runner | [vllm/v1/worker/gpu_model_runner.py](file:///workspace/vllm/v1/worker/gpu_model_runner.py) |
| Attention 抽象 | [vllm/v1/attention/backend.py](file:///workspace/vllm/v1/attention/backend.py) |
| VllmConfig | [vllm/config/vllm.py](file:///workspace/vllm/config/vllm.py) |
| 模型注册表 | [vllm/model_executor/models/registry.py](file:///workspace/vllm/model_executor/models/registry.py) |
| 模型加载器 | [vllm/model_executor/model_loader/__init__.py](file:///workspace/vllm/model_executor/model_loader/__init__.py) |
| 分布式状态 | [vllm/distributed/parallel_state.py](file:///workspace/vllm/distributed/parallel_state.py) |
| LLM 类 | [vllm/entrypoints/llm.py](file:///workspace/vllm/entrypoints/llm.py) |
| OpenAI Server | [vllm/entrypoints/openai/api_server.py](file:///workspace/vllm/entrypoints/openai/api_server.py) |
| CLI 入口 | [vllm/entrypoints/cli/main.py](file:///workspace/vllm/entrypoints/cli/main.py) |
| CustomOp | [vllm/model_executor/custom_op.py](file:///workspace/vllm/model_executor/custom_op.py) |
| CUDA 绑定 | [csrc/torch_bindings.cpp](file:///workspace/csrc/torch_bindings.cpp) |
| 构建脚本 | [setup.py](file:///workspace/setup.py) |
| CMake 配置 | [CMakeLists.txt](file:///workspace/CMakeLists.txt) |

## 附录 B：环境变量要点

核心环境变量定义在 [vllm/envs.py](file:///workspace/vllm/envs.py)，常用：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `VLLM_TARGET_DEVICE` | auto | 构建目标设备 |
| `VLLM_HOST_IP` | "" | 分布式通信 IP |
| `VLLM_PORT` | None | 分布式通信端口 |
| `VLLM_WORKER_MULTIPROC_METHOD` | fork | Worker spawn 方式 |
| `VLLM_CPU_KVCACHE_SPACE` | 0 | CPU KV Cache 空间（GB） |
| `VLLM_LOGGING_LEVEL` | INFO | 日志级别 |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | 60 | 引擎迭代超时 |
| `VLLM_USE_PRECOMPILED` | - | 安装时使用预编译扩展 |
| `MAX_JOBS` | CPU 数 | 编译并行度 |

---

*本文档基于源码静态分析生成，供开发者快速建立全局心智模型。具体实现细节请以源码为准，关键逻辑可结合 `docs/design/` 下的设计文档深入阅读。*
