import fiftyone as fo
import fiftyone.zoo as foz
import os

# 1. 设置 FiftyOne 的数据目录 (确保你已经创建了这个目录，或者 FiftyOne 会创建它)
# 建议在运行脚本前设置环境变量，或者在这里设置（但要确保路径存在）
fiftyone_data_path = "D:/Coding/Wrote_Codes/PY/pythonProject/MachineLearning/ObjectDetection/datasets/coco" # 替换为你想要的路径
os.makedirs(fiftyone_data_path, exist_ok=True)
fo.config.dataset_zoo_dir = fiftyone_data_path
print(f"FiftyOne 数据目录已设置为: {fo.config.dataset_zoo_dir}")

# 下载COCO数据集
try:
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections","segmentations"],
        classes=["person"],
        max_samples=100,
        )
except Exception as e:
    print(f"下载数据集时出错: {e}")
    dataset = None


# 可视化数据集
session = fo.launch_app(dataset)