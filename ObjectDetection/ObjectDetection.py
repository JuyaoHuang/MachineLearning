# 已弃置

import torch
from torchvision.models import detection
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import fiftyone as fo
import fiftyone.zoo as foz
import json
from torch.utils.data import Dataset
import torchvision.transforms as T

# --- 全局配置 ---
# COCO数据集80个标签对照表
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'}

COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
    '#008080', '#000080', '#aa6e28', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#e6beff', '#808080']

# 为每一个标签对应一种颜色，方便显示
COLOR_MAP = {k: COLORS[i % len(COLORS)] for i, k in enumerate(COCO_CLASSES.keys())}

# 判断GPU设备是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- FiftyOne 数据加载函数 ---
def load_fiftyone_coco_dataset(split="validation", max_samples=None):
    """加载 FiftyOne 的 COCO-2017 数据集，只包含行人，并带边界框标注"""
    # 设置 FiftyOne 的数据目录 (确保这个目录是包含 coco-2017 的父目录)
    fiftyone_data_path = "D:/Coding/PyTorch_cache/FifthoneData/Datasets" # 请替换为你实际的路径
    os.makedirs(fiftyone_data_path, exist_ok=True) # 确保目录存在
    fo.config.dataset_zoo_dir = fiftyone_data_path
    print(f"FiftyOne 数据目录已设置为: {fo.config.dataset_zoo_dir}")

    print(f"正在加载 FiftyOne COCO-2017 数据集 (split='{split}', max_samples={max_samples})...")
    try:
        dataset = foz.load_zoo_dataset("coco-2017",
                                       split=split,
                                       label_types=["detections"], # 需要边界框标注来过滤
                                       classes=["person"],         # 只保留包含 'person' 的图片
                                       max_samples=max_samples)
        print(f"FiftyOne: 成功加载了 {len(dataset)} 张 '{split}' 集中的包含 'person' 的图片。")
        
        # 验证标注是否存在
        if len(dataset) > 0:
            sample = dataset.take(1)[0]
            if "detections" not in sample or not sample["detections"].detections:
                print("FiftyOne 警告: 加载的样本中未发现 'detections' 标注信息。")
                return None
            else:
                print("FiftyOne: 检测到有效的 'detections' 标注。")
        else:
            print("FiftyOne: 数据集为空，没有找到符合条件的图片。")
            return None
        
        return dataset
    except Exception as e:
        print(f"加载 FiftyOne COCO 数据集时发生错误: {e}")
        print("请检查 FIFTYONE_DATA_DIR 设置，网络连接，以及 COCO 数据集文件（包括图像和标注）的完整性。")
        return None

# --- PyTorch Dataset 类 ---
class PyTorchCocoPersonDataset(Dataset):
    def __init__(self, fo_dataset, transform=None, target_class_id=1):
        self.fo_dataset = fo_dataset
        self.transform = transform
        self.target_class_id = target_class_id
        
        self.filtered_samples = []
        if self.fo_dataset:
            print("正在预过滤 FiftyOne 数据集，确保包含行人 Ground Truth...")
            # 遍历 FiftyOne dataset 来获取样本，并进行过滤
            # 迭代 FiftyOne dataset 的方法是直接用 for loop
            for sample in self.fo_dataset:
                if "ground_truth" in sample and sample["ground_truth"].detections:
                    has_person = False
                    for det in sample["ground_truth"].detections:
                        if det.label == 'Person':
                            has_person = True
                            break
                    if has_person:
                        self.filtered_samples.append(sample) # 存储 FiftyOne 的 Sample 对象
            print(f"预过滤后，FiftyOne 数据集包含 {len(self.filtered_samples)} 个带有行人 Ground Truth 的样本。")
        else:
            print("FiftyOne 数据集未成功加载，无法进行预过滤。")

        if not self.filtered_samples:
            raise ValueError("经过过滤后，未找到任何包含行人标注的样本。请检查 FiftyOne 的加载参数和数据。")

    def __getitem__(self, index):
        # 直接通过索引访问存储的 FiftyOne Sample 对象
        sample = self.filtered_samples[index]
        img_path = sample["filepath"]
        true_detections = sample["ground_truth"] # 这是一个 FiftyOne Detections 对象

        # 加载图片
        img_pil = Image.open(img_path).convert("RGB")
        img_w, img_h = img_pil.size

        # 准备真实标注
        true_boxes = []
        true_labels = []
        
        if true_detections and true_detections.detections:
            for det in true_detections.detections:
                if det.label == 'Person': # 只处理 'Person' 类别
                    # FiftyOne 的 bounding_box 是 [x, y, w, h] 归一化比例值
                    # 转换为 [xmin, ymin, xmax, ymax] 绝对像素值
                    xmin = det.bounding_box[0] * img_w
                    ymin = det.bounding_box[1] * img_h
                    xmax = xmin + det.bounding_box[2] * img_w
                    ymax = ymin + det.bounding_box[3] * img_h
                    
                    true_boxes.append([xmin, ymin, xmax, ymax])
                    true_labels.append(self.target_class_id) 

        # 转换为 Tensor
        boxes_tensor = torch.as_tensor(true_boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(true_labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes_tensor
        target['labels'] = labels_tensor
        # FiftyOne 样本对象有一个 'id' 属性，可以作为 image_id
        target['image_id'] = torch.tensor([sample.id]) 
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        target['area'] = area
        target['iscrowd'] = torch.zeros((len(true_boxes),), dtype=torch.int64)

        # 应用图像转换
        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, target

    def __len__(self):
        return len(self.filtered_samples)

    def __len__(self):
        return len(self.filtered_samples)

# --- IoU 计算函数 ---
def caculate_ioU(box1, box2):
    """计算两个边界框的 IoU
    Args:
        box1 (list/np.array): [xmin, ymin, xmax, ymax]
        box2 (list/np.array): [xmin, ymin, xmax, ymax]
    Returns:
        float: IoU 值
    """
    xmin_intersection = max(box1[0], box2[0])
    ymin_intersection = max(box1[1], box2[1])
    xmax_intersection = min(box1[2], box2[2])
    ymax_intersection = min(box1[3], box2[3])

    intersection_width = max(0, xmax_intersection - xmin_intersection)
    intersection_height = max(0, ymax_intersection - ymin_intersection)
    intersection_area = intersection_width * intersection_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

# --- 评估函数 ---
def evaluate_predictions_with_pytorch_loader(
    pytorch_loader, model, device, iou_threshold=0.5, person_class_id=1, pred_score_threshold_match=0.1, pred_score_threshold_fp=0.8
):
    """
    使用 PyTorch DataLoader 评估模型的精确率和召回率。
    Args:
        pytorch_loader: PyTorch DataLoader 对象。
        model: 已加载并设置为 eval 模式的 PyTorch 模型。
        device: 模型所在的设备 (cuda or cpu)。
        iou_threshold: 计算 TP/FP/FN 时使用的 IoU 阈值。
        person_class_id: 识别为行人的类别 ID (通常是 1)。
        pred_score_threshold_match: 用于匹配真实框的预测框的最小置信度阈值。
        pred_score_threshold_fp: 用于判断假正例 (FP) 的预测框的置信度阈值。
    Returns:
        tuple: (tp, fp, fn, precision, recall)
    """
    model.eval()
    tp = 0 # 真正例 true positive
    fp = 0 # 假正例 false positive
    fn = 0 # 假负例 false negative

    print(f"开始评估，使用 IoU 阈值: {iou_threshold}")
    
    # 迭代 PyTorch DataLoader
    for i, (images_batch, targets_batch) in enumerate(pytorch_loader):
        # images_batch: List of Tensors
        # targets_batch: List of target Dictionaries

        # 将图像和真实标注移到设备
        images_on_device = [img.to(device) for img in images_batch]
        targets_on_device = []
        for target in targets_batch:
            target_processed = {}
            for k, v in target.items():
                target_processed[k] = v.to(device)
            targets_on_device.append(target_processed)

        # --- 模型预测 ---
        # PyTorch detection models 的 forward 方法在推理时，传入图像列表，返回预测结果列表
        predictions = model(images_on_device)

        # --- 匹配逻辑 ---
        for img_idx in range(len(images_batch)): # 遍历 batch 中的每张图片
            preds = predictions[img_idx] # 预测结果 for this image
            gts = targets_on_device[img_idx] # 真实的标注 for this image

            pred_boxes = preds['boxes'].cpu().numpy()
            pred_labels = preds['labels'].cpu().numpy()
            pred_scores = preds['scores'].cpu().numpy()

            true_boxes = gts['boxes'].cpu().numpy()
            true_labels = gts['labels'].cpu().numpy()

            # --- 匹配逻辑 ---
            image_tp = 0
            image_fp = 0
            image_fn = 0
            matched_preds = [False] * len(pred_boxes) # 标记预测框是否已被匹配

            # 1. 匹配真实框 (计算 TP 和 FN)
            for gt_idx, gt_box in enumerate(true_boxes):
                if true_labels[gt_idx] != person_class_id: # 只关注行人
                    continue

                best_iou = 0
                best_pred_idx = -1

                # 找到与当前真实框 IoU 最大的预测框 (且置信度高于一个低阈值，未被匹配)
                for pred_idx, pred_box in enumerate(pred_boxes):
                    if pred_labels[pred_idx] == person_class_id and not matched_preds[pred_idx] and pred_scores[pred_idx] > pred_score_threshold_match:
                        iou = caculate_ioU(gt_box, pred_box)
                        if iou > iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_pred_idx = pred_idx

                if best_pred_idx != -1: # 找到一个匹配的预测框
                    image_tp += 1
                    matched_preds[best_pred_idx] = True # 标记为已匹配
                else: # 没有找到匹配的预测框
                    image_fn += 1

            # 2. 计算 FP (未匹配到的预测框)
            for pred_idx, pred_box in enumerate(pred_boxes):
                # 如果这个预测框是行人，并且没有被匹配到，且置信度高于FP阈值
                if pred_labels[pred_idx] == person_class_id and not matched_preds[pred_idx] and pred_scores[pred_idx] > pred_score_threshold_fp:
                    image_fp += 1
            
            # 累加到总计数器
            tp += image_tp
            fp += image_fp
            fn += image_fn

        # 计算总体精确率和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return tp, fp, fn, precision, recall

# --- 主执行块 ---
# --- 主执行块 ---
if __name__ == '__main__':
    # 1. 设置 FiftyOne 数据目录
    fiftyone_data_path = "D:/Coding/PyTorch_cache/FifthoneData/Datasets" # 请确保此路径正确
    os.environ["FIFTYONE_DATA_DIR"] = fiftyone_data_path
    print(f"FiftyOne 数据目录已设置为: {os.environ['FIFTYONE_DATA_DIR']}")    

    # 2. 确保 COCO 数据已下载 (只包含行人)
    print("\n确保 COCO 数据已下载 (只包含行人)...")
    try:
        # 尝试加载一个小的样本来触发下载和检查
        # 确保 COCO-2017 数据集在 FIFTYONE_DATA_DIR/coco-2017/ 下可用
        # 如果 FIFTYONE_DATA_DIR 没有 coco-2017 文件夹，FiftyOne 会尝试下载
        # 我们使用 max_samples=1 只是为了触发下载流程并检查下载是否成功
        _ = foz.load_zoo_dataset("coco-2017", 
                                 split="validation", 
                                 label_types=["detections"], 
                                 classes=["person"], 
                                 max_samples=1)
        print("COCO 数据集下载（或验证存在）成功。")
    except Exception as e:
        print(f"COCO 数据集下载失败: {e}")
        print("请检查 FIFTYONE_DATA_DIR 设置，网络连接，以及 COCO 数据集文件（包括图像和标注）的完整性。")
        exit()

    # 3. 加载 FiftyOne COCO 数据集对象
    print("\n正在加载 FiftyOne COCO 数据集对象...")
    # 注意：这里的 max_samples 控制了 FiftyOne 加载到内存中的样本数量，
    # 而不是实际下载的数据量。下载是根据 label_types 和 classes 进行的。
    # 为了评估，我们通常需要较多的样本。
    # 如果你的数据集很大，可以先用较小的 max_samples 来调试代码。
    train_dataset_fo = load_fiftyone_coco_dataset(split="train", max_samples=200) # 加载200个训练样本
    val_dataset_fo = load_fiftyone_coco_dataset(split="validation", max_samples=50) # 加载50个验证样本

    if train_dataset_fo is None or val_dataset_fo is None:
        print("FiftyOne COCO 数据集加载失败，无法继续。")
        exit()

    # 4. 创建 PyTorch Dataset 和 DataLoader
    print("\n正在创建 PyTorch Dataset 和 DataLoader...")
    transform_for_pytorch = T.Compose([
        T.ToTensor(), # 将 PIL Image 转换为 Tensor (0-1 范围)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 均值和标准差
    ])

    COCO_ROOT_DIR = os.path.join(os.environ.get("FIFTYONE_DATA_DIR", "."), "coco-2017")
    if not os.path.exists(COCO_ROOT_DIR):
        print(f"COCO dataset not found at the expected path: {COCO_ROOT_DIR}. Ensure FiftyOne has downloaded it correctly.")
        exit()

    try:
        train_dataset_pytorch = PyTorchCocoPersonDataset(
            fo_dataset=train_dataset_fo,
            transform=transform_for_pytorch,
            target_class_id=1
        )
        val_dataset_pytorch = PyTorchCocoPersonDataset(
            fo_dataset=val_dataset_fo,
            transform=transform_for_pytorch,
            target_class_id=1
        )
    except FileNotFoundError as e:
        print(e)
        exit()
    except ValueError as e: # 捕获 PyTorchCocoPersonDataset 中可能抛出的 ValueError
        print(e)
        exit()
    except Exception as e:
        print(f"创建 PyTorch Dataset 时发生错误: {e}")
        exit()

    # collate_fn 是必需的，因为目标检测的 target 是一个列表，每个元素是字典
    def collate_fn_for_detection(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    batch_size = 4 # 根据你的 GPU 显存调整
    train_loader_pytorch = torch.utils.data.DataLoader(
        train_dataset_pytorch,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn_for_detection
    )

    val_loader_pytorch = torch.utils.data.DataLoader(
        val_dataset_pytorch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn_for_detection
    )
    print("PyTorch DataLoader 创建完成。")

    # 5. 加载预训练模型
    print("\n加载预训练的 Mask R-CNN 模型...")
    model = detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # --- 修改模型的分类头，使其只输出行人类别 (2个类别: 背景 + 行人) ---
    num_classes_for_person_detection = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes_for_person_detection)
    
    # 如果你已经微调了模型，应该在这里加载你的权重
    # model.load_state_dict(torch.load("path/to/your/finetuned_model.pth"))

    model.to(device)
    model.eval() # 设置为评估模式

    # 6. 使用 PyTorch DataLoader 进行评估
    print("\n开始评估模型性能 (精确率和召回率)...")
    
    # 调用评估函数
    tp, fp, fn, precision, recall = evaluate_predictions_with_pytorch_loader(
        val_loader_pytorch, # 使用验证集进行评估
        model,
        device,
        iou_threshold=0.5,
        person_class_id=1, # COCO 中 'person' 的类别 ID 是 1
        pred_score_threshold_match=0.1, # 用于匹配的低置信度阈值
        pred_score_threshold_fp=0.8    # 用于判断FP的置信度阈值
    )

    print(f"\n--- 评估结果 ---")
    print(f"IoU 阈值: {0.5}")
    print(f"总 TP: {tp}")
    print(f"总 FP: {fp}")
    print(f"总 FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")