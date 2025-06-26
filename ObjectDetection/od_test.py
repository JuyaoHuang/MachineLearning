import torch
from torchvision.models import detection
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fiftyone as fo
import fiftyone.zoo as foz
import os
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

# ----设置 Fiftyone 的数据目录----
fiftyone_data_path = "D:/Coding/Wrote_Codes/PY/pythonProject/MachineLearning/ObjectDetection/datasets/coco" # 替换为你想要的路径
os.makedirs(fiftyone_data_path, exist_ok=True)
fo.config.dataset_zoo_dir = fiftyone_data_path
print(f"FiftyOne 数据目录已设置为: {fo.config.dataset_zoo_dir}")

# ----解析 COCO 标注文件(json格式),获取 person 的真实边框标注----
import json
def parse_coco_annotations(annotation_file,img_filename):
    """
    解析 COCO 标注文件，找到指定图片的所有行人标注。
    Args:
        annotation_file (str): COCO 标注 JSON 文件路径。
        target_image_filename (str): 你要查找标注的图片文件名。
    Returns:
        图像边框列表 boxes
        标签列表 labels
        如果找不到图片或图片中没有行人，则返回 ([], [])。
    """
    try:
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到所需标注文件: {annotation_file}")
        return [], []

    # 创建一个 filename -> image_id 的映射，查找目标图片的 image_id
    target_image_id = None
    for img_info in coco_data['images']:
        # 比较文件名
        if os.path.basename(img_info['file_name']) == img_filename:
            target_image_id = img_info['id']
            break

    if target_image_id is None:
        print(f"错误: 在标注文件中未找到图片 '{img_filename}'。")
        return [], []

    print(f"找到图片 '{img_filename}' 的 image_id: {target_image_id}")

    # 提取该图片的标注信息
    target_annotations = []
    # COCO 的 categories 列表提供了 category_id 到 category_name 的映射
    # 我们需要知道 'person' 的 category_id，通常是 1
    person_id = None
    for category in coco_data['categories']:
        if category['name'] == 'person':
            person_id = category['id']
            break
    
    if person_id is None:
        print("错误: 在标注文件中未找到 'person' 类别。")
        return [], []
    
    print(f"'person' 的 category_id 是: {person_id}")

    for ann in coco_data['annotations']:
        if ann['image_id'] == target_image_id:

            if ann['category_id'] == person_id:
                # COCO 标注的边界框格式是 [xmin, ymin, width, height]
                # 我们需要将其转换为 [xmin, ymin, xmax, ymax]
                xmin, ymin, width, height = ann['bbox']
                xmax = xmin + width
                ymax = ymin + height
                # 注意：PIL 的坐标系是 (x, y) 左上角为 (0, 0)
                target_annotations.append([xmin, ymin, xmax, ymax])

    print(f"为图片 '{img_filename}' 找到 {len(target_annotations)} 个行人真实标注。")
    return target_annotations, [person_id] * len(target_annotations) # 返回标签列表，这里都假定为 person 的ID


# IoU 计算函数 
def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU (Intersection over Union)。
    边界框格式: [xmin, ymin, xmax, ymax]
    """
    # 确定重叠矩形的左上角和右下角坐标
    ixmin = max(box1[0], box2[0])
    iymin = max(box1[1], box2[1])
    ixmax = min(box1[2], box2[2])
    iymax = min(box1[3], box2[3])

    # 计算重叠区域的宽度和高度
    iw = max(0, ixmax - ixmin + 1)
    ih = max(0, iymax - iymin + 1)

    # 计算重叠面积
    intersection_area = iw * ih

    # 计算两个边界框的面积
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # 计算并集面积
    union_area = area1 + area2 - intersection_area

    # 计算 IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


# 目标检测函数+指标评估
def my_detection(img_path,model,truth_boxes,truth_labels,iou_threshold=0.5):
    """ 
    对单张图片进行预测, 并返回评估指标所需的信息
    对单张图片进行目标检测，并计算精确率和召回率（针对行人）。
    Args:
        img_path (str): 输入图片路径。
        model (torchvision.models.detection): 预训练的目标检测模型。
        truth_boxes (list): 真实边界框的列表，格式为 [[xmin, ymin, xmax, ymax], ...]。
        truth_labels (list): 真实类别标签的列表，格式为 [label1, label2, ...]
        iou_threshold (float): 用于匹配预测框和真实框的 IoU 阈值。
    Returns:
        tuple: (precision, recall, fig, ax)
               precision (float): 行人的精确率。
               recall (float): 行人的召回率。
               fig (PIL.Image.Image): 带有检测结果和标签的图片对象。
               ax (PIL.ImageDraw.Image): 用于绘制的对象。
    """
    # 加载预训练目标检测模型maskrcnn
    # model = detection.maskrcnn_resnet50_fpn(pretrained=True)
    # 使用fasterrcnn模型
    # model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # 改为使用 VGG16 骨干的 SSD300
    # model = detection.ssd300_vgg16(pretrained=True) 
    # 使用Retinanet模型
    # model = detection.retinanet_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    # 读取输入图像，并转化为tensor
    origin_img = Image.open(img_path, mode='r').convert('RGB')

    # SSD300模型需要输入尺寸为300x300的图像
    # img = TF.resize(origin_img, (300, 300))  # 调整图像大小为300x300

    img = TF.to_tensor(origin_img)
    img = img.to(device)

    # 将图像输入神经网络模型中，得到输出
    output = model(img.unsqueeze(0))
    labels = output[0]['labels'].cpu().detach().numpy()  # 预测每一个obj的标签
    scores = output[0]['scores'].cpu().detach().numpy()  # 预测每一个obj的得分
    bboxes = output[0]['boxes'].cpu().detach().numpy()  # 预测每一个obj的边框

# ----预测框和真实框匹配----
    # 只选取得分大于0.8的检测结果
    obj_index = np.argwhere(scores > 0.8).squeeze(axis=1).tolist()

    person_id = 1 # 行人 id

    # 过滤出预测为行人的框及其得分和位置
    pred_index = np.where(labels == person_id)[0] # 获取所有预测为行人的索引
    pred_boxes = bboxes[pred_index] 
    pred_scores = scores[pred_index]  

    person_truth_boxes = np.array(truth_boxes)  # 转换为numpy数组

    # 初始化三个统计量 tp fp fn
    tp = 0  # 真正例
    fp = 0  # 假正例
    fn = 0  # 假负例

    match_truth_index = set() # 用于记录已匹配的真实框索引
    match_pred_index = set()  # 用于记录已匹配的预测框索引

    # 遍历所有预测为行人的框 pred_boxes
    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        best_iou = 0
        best_truth_index = -1

        # 找到与当前预测框匹配值 ioU 最大的真实框
        for j in range(len(person_truth_boxes)):
            truth_box = person_truth_boxes[j]
            iou = calculate_iou(pred_box, truth_box)
            if iou > best_iou:
                best_iou = iou
                best_truth_index = j
        
        # 如果最大 iou 大于阈值，并且此真实框未被匹配过
        if best_iou >= iou_threshold and best_truth_index not in match_truth_index:
            tp += 1
            match_truth_index.add(best_truth_index)  # 标记此真实框已被匹配
            match_pred_index.add(i)  # 标记此预测框已被匹配

    # 假正例 是所有被预测为行人的框中，未被标记为 tp 的框的个数
    fp = len(pred_boxes) - len(match_pred_index)
    # 假负例 是所有真实行人框中，未被标记为 tp 的框的个数
    fn = len(person_truth_boxes) - len(match_truth_index)

    # 计算精确率和召回率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

# ---- 绘图部分 ----

    draw = ImageDraw.Draw(origin_img)
    font = ImageFont.truetype('arial.ttf', 15)  
    person_id = 1

    person_color = COLOR_MAP.get(person_id, 'red')  # 获取行人颜色，默认为红色

    # 绘制真实框 -- 绿色
    for i,truth_box in enumerate(person_truth_boxes):
        xmin,ymin,xmax,ymax = truth_box
        draw.rectangle([xmin,ymin,xmax,ymax],outline='green',width=2)
    
    # 绘制预测边界框 (红色 TP, 黄色 FP) 和文本
    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        pred_score = pred_scores[i]
        is_tp = i in match_pred_index  # 是否为真正例

        xmin, ymin, xmax, ymax = pred_box
        color = 'red' if is_tp else 'yellow'
        text_color = 'white' if is_tp else 'black'

        draw.rectangle([xmin,ymin,xmax,ymax], outline=color,width=2)

        display_text = f"Person score: {pred_score:.3f}"
        text_width = font.getlength(display_text)
        text_height = font.getbbox(display_text)[3]  
        text_loc = [xmin + 2., ymin - text_height - 2.]  # 放在框的上方一点
        # if text_loc[1] < 0: text_loc[1] = ymin + 2.  # 如果超出顶部，则放在框的下方
        textbox_loc = [xmin, text_loc[1], xmin + text_width + 4., text_loc[1] + text_height + 2.]

        img_width, img_height = origin_img.size
        textbox_loc[2] = min(textbox_loc[2], img_width)
        textbox_loc[3] = min(textbox_loc[3], img_height)
        draw.rectangle(xy=textbox_loc, fill=color)
        draw.text(xy=text_loc, text=display_text, fill=text_color, font=font)


    # ----该部分只画红框，暂时舍弃----
    #     
    # 使用ImageDraw将检测到的边框和类别打印在图片中，得到最终的输出
    # draw = ImageDraw.Draw(origin_img)
    # font = ImageFont.truetype('arial.ttf', 15)  # 加载字体文件

    # person_id_in_COCO = 1  # COCO数据集中person的标签ID
    # for i in obj_index:
    #     person_label = labels[i]  # 获取标签

    #     # --- 检查标签是否是 'person' ---
    #     if person_label == person_id_in_COCO: 
    #         box_loc = bboxes[i].tolist()
    #         label_text = COCO_CLASSES[person_label] 
    #         score = scores[i] 

    #         # 画框
    #         draw.rectangle(xy=box_loc, outline=COLOR_MAP[person_label], width=2) # 增加边框宽度使其更明显

    #         # --- 文本绘制部分 ---
    #         display_text = f"{label_text}: {score:.3f}"

    #         # 获取文本的尺寸，以便绘制背景框
    #         # 获取文本的实际宽度
    #         text_width = font.getlength(display_text)
    #         # 获取文本的实际高度 (通常是 font.getbbox(text)[3] )
    #         # 注意: font.getbbox() 返回的是一个包含 (left, top, right, bottom) 的元组，
    #         # 其中 left 和 top 通常是 0 或负值，right 和 bottom 是文本的宽度和高度的偏移。
    #         # 我们需要的是文本区域的实际高度，这通常是 text_bbox[3] - text_bbox[1]
    #         # 对于 arial.ttf， text_bbox[3] 应该能代表高度的近似值。
    #         text_height = font.getbbox(display_text)[3] # 假设 text_bbox[1] 是 0 或负值，bottom 代表了高度

    #         # 设置标签文本的左上角位置(left, top)
    #         # 通常放在框的左上角上方或内部
    #         text_loc = [box_loc[0] + 2., box_loc[1] - text_height - 2.] # 放在框的上方一点

    #         # 确保文本框不会超出图片顶部
    #         if text_loc[1] < 0:
    #             text_loc[1] = box_loc[1] + 2. # 如果超出顶部，则放在框的右侧
    #         # 设置显示标签的背景框 (left, top, right, bottom)
    #         textbox_loc = [
    #             text_loc[0], text_loc[1],
    #             text_loc[0] + text_width + 4., text_loc[1] + text_height + 2. # 加上一点padding
    #         ]

    #         # 确保textbox不会超出图片边界
    #         img_width, img_height = origin_img.size
    #         textbox_loc[2] = min(textbox_loc[2], img_width)
    #         textbox_loc[3] = min(textbox_loc[3], img_height)

    #         # 绘制标签背景框
    #         draw.rectangle(xy=textbox_loc, fill=COLOR_MAP[person_label])
    #         # 绘制标签文本
    #         draw.text(xy=text_loc, text=display_text, fill='white', font=font)                


    origin_img.show()
    origin_img.save(f"ObjectDetection/ouputs/COCO/{os.path.basename(img_path)}_detection.jpg")
    return precision, recall, origin_img, draw


if __name__ == '__main__':
    
    # 假设你已经有了 image_file_path，并且知道它在哪个 COCO 集合里
    # 并且有对应的 annotation_file 路径
    img_filepath = "ObjectDetection/datasets/coco/coco-2017/validation/data/000000016598.jpg"
    coco_val_annotation = "ObjectDetection/datasets/coco/coco-2017/raw/instances_val2017.json"


    img_name = os.path.basename(img_filepath)
    print(f"正在解析图片: {img_name}")

    # 解析真实标注
    img_boxes, img_labels = parse_coco_annotations(
        coco_val_annotation,
        img_name,
    )

    if img_boxes and img_labels:
        print(f"真实标注的边界框: {img_boxes}")
        print(f"真实标注的标签: {img_labels}")

        # 加载预训练目标检测模型maskrcnn
        model = detection.maskrcnn_resnet50_fpn(pretrained=True)
        # 使用fasterrcnn模型
        # model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # 改为使用 VGG16 骨干的 SSD300
        # model = detection.ssd300_vgg16(pretrained=True) 
        # 使用Retinanet模型
        # model = detection.retinanet_resnet50_fpn(pretrained=True)

        # 调用目标检测函数
        print("开始目标检测...")
        precision, recall, fig, ax=my_detection(
                                        img_filepath,
                                        model,
                                        img_boxes,
                                        img_labels,
                                        iou_threshold=0.5
                                        ) 
        
        
        print(f"评估结果：")
        print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}")

    else:
        print("没有找到行人标注或图片不存在。请检查文件名和路径。")    
        


    # model.to(device)
    # model.eval()



        # 绘制源代码
                # if person_label == "person":
        #     box_loc = bboxes[i].tolist()
        #     draw.rectangle(xy=box_loc, outline=COLOR_MAP[labels[i]])  # 画框

        #     # 获取标签文本的左上和右下边界(left, top, right, bottom)
        #     text_size = font.getbbox(COCO_CLASSES[labels[i]])
        #     # 设置标签文本的左上角位置(left, top)
        #     text_loc = [box_loc[0] + 2., box_loc[1]]
        #     # 设置显示标签的边框(left, top, right, bottom)
        #     textbox_loc = [
        #         box_loc[0], box_loc[1],
        #         box_loc[0] + text_size[2] + 4., box_loc[1] + text_size[3]
        #     ]
        #     # 绘制标签边框
        #     draw.rectangle(xy=textbox_loc, fill=COLOR_MAP[labels[i]])
        #     # 绘制标签文本
        #     draw.text(xy=text_loc, text=COCO_CLASSES[labels[i]], fill='white', font=font)  