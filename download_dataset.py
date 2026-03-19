"""
download_dataset.py - 从 Hugging Face 下载动漫人格数据集

数据集信息:
- 名称：maomao88/anime-waifu-personality-chat-with-questions
- 描述：动漫人格对话数据集，包含问答形式的训练数据
- 用途：用于微调模型，注入动漫人格特征
"""
from datasets import load_dataset
import os

# ==================== 配置 ====================
DATASET_NAME = "maomao88/anime-waifu-personality-chat-with-questions"
SAVE_PATH = "./anime-waifu-dataset"

print(f"开始下载数据集：{DATASET_NAME}")
print(f"保存路径：{SAVE_PATH}\n")

# 创建保存目录
os.makedirs(SAVE_PATH, exist_ok=True)

try:
    # 加载数据集
    # 参数说明:
    # - repo_id: HuggingFace 数据集仓库 ID
    # - split: 数据分割，None 表示加载所有分割 (train/validation/test)
    dataset = load_dataset(DATASET_NAME)

    print(f"✓ 数据集加载完成!")
    print(f"  数据集结构：{dataset}")

    # 打印数据集信息
    if isinstance(dataset, dict):
        for split_name, split_data in dataset.items():
            print(f"\n  {split_name} 集样本数：{len(split_data)}")
            if len(split_data) > 0:
                print(f"  示例数据：{split_data[0]}")
    else:
        print(f"  总样本数：{len(dataset)}")
        if len(dataset) > 0:
            print(f"  示例数据：{dataset[0]}")

    # 保存到本地磁盘
    print(f"\n正在保存到：{SAVE_PATH}...")
    dataset.save_to_disk(SAVE_PATH)

    print(f"\n✓ 数据集已保存到：{SAVE_PATH}")
    print("\n提示：使用以下代码加载本地数据集:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{SAVE_PATH}')")

except Exception as e:
    print(f"\n✗ 下载失败：{str(e)}")
    print("\n可能的原因:")
    print("  1. 网络连接问题，请检查网络或配置代理")
    print("  2. 数据集不存在或需要访问权限")
    print("  3. 磁盘空间不足")
