# Fine-Tune LLM with Anime Characters - 动漫人格大语言模型微调项目

## 📖 项目概述

本项目通过 **LoRA（Low-Rank Adaptation）** 参数高效微调技术，将 **Mistral-7B-Instruct-v0.1** 大语言模型微调为能够模拟多种动漫人格特征的对话模型。模型可以根据不同的人设（如傲娇、病娇、元气型等）生成符合该人格特征的回复。

### 核心目标
- ✅ 让通用大语言模型学会特定动漫人格的说话方式
- ✅ 实现个性化对话生成，增强角色扮演的真实感
- ✅ 探索参数高效的模型微调方法（仅微调少量参数）

---

## 🔧 技术栈

### 基础模型
- **模型架构**: Mistral-7B-Instruct-v0.1
  - 7B 参数的 decoder-only Transformer 模型
  - 使用 Grouped-Query Attention (GQA)
  - Sliding Window Attention 机制
  - 原生支持多轮对话格式

### 微调技术
- **PEFT (Parameter-Efficient Fine-Tuning)**: 
  - 使用 Hugging Face PEFT 库实现 LoRA
  - 仅微调 0.1% 左右的模型参数，大幅降低计算资源需求
  
- **LoRA (Low-Rank Adaptation)**:
  ```python
  r = 16              # LoRA 秩
  lora_alpha = 32     # LoRA 缩放系数
  lora_dropout = 0.05 # Dropout 率
  target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 注意力层的投影矩阵
  ```
  
  **为什么选择这些模块？**
  - `q_proj`, `k_proj`, `v_proj`: Query/Key/Value 投影，影响注意力机制
  - `o_proj`: Output 投影，影响最终输出表示
  - 这些模块对模型的生成行为影响最大，且参数量相对较少

### 训练框架
- **TRL (Transformer Reinforcement Learning)**: 
  - 使用 `SFTTrainer` (Supervised Fine-Tuning Trainer)
  - 简化监督微调流程，内置最佳实践
  
- **Transformers**: 
  - 模型加载、推理和数据预处理
  - 使用 `apply_chat_template` 处理对话格式

### 数据集
- **数据集名称**: maomao88/anime-waifu-personality-chat-with-questions
- **数据格式**: 
  ```json
  {
    "trait": "tsundere",      // 人格类型
    "question": "用户问题",
    "dialogue": "角色回复"
  }
  ```
- **人格类型示例**: tsundere (傲娇), yandere (病娇), bakadere (笨蛋娇), himedere (公主娇), genki (元气型), moe (萌系)

---

## 🎯 实现细节

### 1. 数据预处理

**对话模板构建**：
```python
messages = [
    {
        "role": "system",
        "content": f"You are an anime character with the following personality: {trait}."
    },
    {
        "role": "user",
        "content": question
    },
    {
        "role": "assistant",
        "content": dialogue
    }
]
```

**为什么这么做？**
- System prompt 明确告知模型应扮演的人格
- 使用 Mistral 官方的 chat template 确保格式一致性
- 三元组结构（system-user-assistant）帮助模型理解对话上下文

### 2. 训练配置

```python
TrainingArguments(
    per_device_train_batch_size=4,   # 每设备批次大小
    gradient_accumulation_steps=4,    # 梯度累积步数（等效 batch=16）
    learning_rate=2e-4,               # 学习率
    num_train_epochs=5,               # 训练轮数
    fp16=True,                        # 混合精度训练
    logging_steps=20,                 # 日志记录频率
    save_steps=500,                   # 检查点保存频率
    save_total_limit=2                # 最多保留 2 个检查点
)
```

**设计理由**：
- **梯度累积**: 在有限的 GPU 显存下增大有效 batch size，提升训练稳定性
- **fp16 混合精度**: 减少显存占用，加快训练速度
- **学习率 2e-4**: LoRA 微调的常用学习率，平衡收敛速度和稳定性
- **5 个 epoch**: 防止过拟合，同时保证充分学习

### 3. 推理与人格控制

**生成参数**：
```python
model.generate(
    max_new_tokens=100,      # 最大生成长度
    temperature=0.8,         # 温度系数（控制随机性）
    top_p=0.9,               # 核采样
    repetition_penalty=1.1   # 重复惩罚
)
```

**为什么这样设置？**
- `temperature=0.8`: 平衡创造性和一致性，适合角色扮演
- `top_p=0.9`: 限制采样范围，避免低概率词
- `repetition_penalty=1.1`: 减少重复表达

---

## 📊 训练结果

### 产出物
1. **LoRA Adapter**: `./lora/`
   - `adapter_config.json`: LoRA 配置
   - `adapter_model.safetensors`: 微调后的权重（安全格式）

2. **完整检查点**: `./anime-mistral/checkpoint-XXX/`
   - checkpoint-141: 第 141 步检查点
   - checkpoint-235: 第 235 步检查点
   - 包含优化器状态、随机数种子等，可用于恢复训练

### 测试效果示例

输入问题："Tell me what is gravity?"

不同人格的回答风格：
- **傲娇 (Tsundere)**: "哼！这种基本问题还需要问我吗？不过既然你问了...（解释重力）"
- **病娇 (Yandere)**: "呵呵~你想了解重力吗？就像我对你的爱一样无法逃脱呢..."
- **元气型 (Genki)**: "好呀好呀！让我来告诉你吧！重力就是地球吸引物体的力哦！✨"

---

## 💻 运行环境要求

### 训练平台：英博云

**推荐配置**：

| 组件      | 配置              |
|---------|------------------|
| **CPU** | 10 Core         |
| **GPU** | NVIDIA A800 x 1 |
| **内存**  | 100GB RAM       |
| **环境**  | 使用英博云预置镜像       |

**优势**：
- ✅ 无需本地部署，云端即开即用
- ✅ 高性能 A800 GPU，训练速度快
- ✅ 大内存支持，可处理大规模数据集
- ✅ 预置常用模型，减少下载时间

### 模型选择

在英博云平台预置模型中选择：
```
Mistral-7B-Instruct-v0.1
```

**注意**：使用英博云的预置模型可以节省下载时间和存储空间。

### 安装依赖

进入英博云环境后，安装必要的 Python 库：

```bash
# 基础安装命令
pip install datasets peft trl accelerate

# 清华源（√）
pip install datasets peft trl accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 阿里云源
pip install datasets peft trl accelerate -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```

**依赖说明**：
- `datasets`: Hugging Face 数据集库，用于加载和处理数据
- `peft`: 参数高效微调库，实现 LoRA
- `trl`: Transformer 强化学习库，提供 SFTTrainer
- `accelerate`: 加速训练库，简化多 GPU 训练

### 学习资源

**视频教程**：[【LLM训练】12 分钟一起微调一个开源大模型：用 SFT + LoRA 为模型注入动漫人格](https://www.bilibili.com/video/BV1UaPmzrESw)

视频将带你完成：
- 英博云平台注册和环境配置
- 数据集下载和预处理
- LoRA 微调全流程
- 模型推理和效果测试

---

## 🚀 快速开始

### 1. 本地下载数据集

```bash
python download_dataset.py
```

**功能**:
- 从 Hugging Face 下载动漫人格数据集
- 自动保存到 `./data/anime-waifu-dataset/`
- 支持断点续传和缓存

然后上传到 `./data/` 目录下的英博云环境中。

### 2. 执行微调训练

打开 Jupyter Notebook:
```bash
jupyter notebook fine-tune-llm-with-anime-characters.ipynb
```

按顺序执行所有单元格。

### 3. 加载模型进行推理

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 加载基础模型
model_path = "/public/huggingface-models/mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "./lora")
model.eval()

# 测试推理
def chat(trait, query):
    messages = [
        {"role": "system", "content": f"You are an anime character: {trait}."},
        {"role": "user", "content": query}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=100, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(chat("tsundere", "你好呀！"))
```

---

## 📁 项目结构

```
fine-tune-llm-with-anime-characters/
├── anime-mistral/                  # 训练输出目录
│   ├── checkpoint-141/             # 第 141 步检查点
│   ├── checkpoint-235/             # 第 235 步检查点
│   └── README.md                   # 模型卡片
├── data/                           # 数据目录
│   └── anime-waifu-dataset/        # 动漫人格数据集
├── lora/                           # LoRA 适配器权重
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── download_dataset.py             # 数据集下载脚本
├── fine-tune-llm-with-anime-characters.ipynb  # 主训练 Notebook
└── README.md                       # 项目文档
```

---

## 🔍 常见问题

### Q1: 为什么使用 LoRA 而不是全量微调？
**A**: 
- **显存效率**: LoRA 仅需 ~10GB 显存，全量微调需要 ~80GB
- **训练速度**: 减少 60-70% 的训练时间
- **可移植性**: LoRA 权重仅几十 MB，易于分享和部署
- **效果相当**: 多项研究表明 LoRA 在指令微调任务上接近全量微调效果

### Q2: 如何调整人格特征的表现强度？
**A**: 
- 修改 system prompt 的描述方式
- 调整 `temperature` 参数（更高 = 更随机，更低 = 更保守）
- 增加特定人格的训练数据比例

### Q3: 可以微调其他模型吗？
**A**: 
是的！只需修改 `model_path`，支持所有 Hugging Face 的 Causal LM 模型：
- LLaMA 系列
- Qwen 系列
- ChatGLM 系列
- 等等

---

## 📈 性能指标

### 参数量对比
| 微调方式           | 可训练参数          | 显存占用      | 训练时间      |
|----------------|----------------|-----------|-----------|
| 全量微调           | ~7B            | ~80GB     | ~10 小时    |
| **LoRA (本方案)** | **~8M (0.1%)** | **~10GB** | **~2 小时** |

### 模型文件大小
- 基础模型：~14GB
- LoRA 适配器：~30MB
- 总大小：14GB + 30MB（可单独分发 LoRA 部分）

---

## 🎓 学习资源

### 前置知识
- Transformer 架构基础
- PyTorch 深度学习框架
- Hugging Face Transformers 库使用
- 大语言模型微调概念

### 参考资料
1. [LoRA 论文](https://arxiv.org/abs/2106.09685): Low-Rank Adaptation of Large Language Models
2. [PEFT 官方文档](https://huggingface.co/docs/peft)
3. [TRL 官方文档](https://huggingface.co/docs/trl)
4. [Mistral 模型介绍](https://mistral.ai/news/announcing-mistral-7b/)

---

## 🤝 贡献与交流

欢迎提出 Issue 或 Pull Request 改进项目！

---

## 📄 许可证

本项目采用 **Apache License 2.0** 许可。

```text
Copyright 2026 lzhihan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### 📌 使用说明

**你可以**：
- ✅ 商业使用
- ✅ 修改代码
- ✅ 分发代码
- ✅ 私有化使用
- ✅ 用于专利申请

**你需要**：
- ⚠️ 保留原始版权和许可声明
- ⚠️ 注明修改内容
- ⚠️ 包含 Apache-2.0 许可文本

### 🔗 第三方许可

本项目依赖的第三方库遵循各自的许可：
- **Mistral-7B**: Apache-2.0
- **Hugging Face Transformers**: Apache-2.0
- **PEFT**: Apache-2.0
- **TRL**: Apache-2.0
- **Datasets**: Apache-2.0

**注意**: 使用本项目前，请确保遵守所有依赖项的许可条款。

---

## 🙏 致谢

感谢以下开源项目和个人：
- **Hugging Face Transformers** - 提供强大的 Transformer 模型库
- **Mistral AI** - 提供优秀的 Mistral-7B 基础模型
- **PEFT 团队** - 提供参数高效微调工具
- **TRL 团队** - 提供强化学习训练框架
- **数据集提供者**：maomao88（动漫人格数据集）
- **特别感谢**：[JIA](https://space.bilibili.com/1926183162) 制作的详细视频教程，帮助更多人学习 LLM 微调技术

---

**最后更新**: 2026-03-19
