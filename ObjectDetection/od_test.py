import torch
from torchvision.models import detection
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fiftyone as fo
import fiftyone.zoo as foz
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

# 引入 FiftyOne 库 加载标签数据
def load_fiftyone_dataset(split="train",max_samples=50):
    """加载 FiftyOne 的 OpenImages-v7数据集, 只包含行人和边框"""
    # 设置 FiftyOne 的数据目录
    fo.config.dataset_zoo_dir = "ObjectDetection/datasets/OpenImages"
    print(f"FiftyOne 的数据目录:{fo.config.dataset_zoo_dir}")

    print(f"正在加载 FiftyOne Open Images V7 数据集 (split='{split}', max_samples={max_samples})...")
    try:
        dataset=foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],  # 只加载检测标注
            classes=["person"],
            max_samples=max_samples,
        )
        print(f"加载了 {len(dataset)} 张包含行人的图片及其边界框标注。")
        # 验证数据集是否存在
        sample=list(dataset.take(1))[0]
        if "detections" not in sample:
            print("错误: 未加载到'detection'标注信息！")
        return dataset
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None

# 目标检测函数
def my_detection(model,img_path,true_detections=None,threshold=0.5,person_class_id=1):
    """ 
        对单张图片进行预测, 并返回评估指标所需的信息
        true_detections 真实的标注信息，用于计算评估指标 TF FP FN
        threshold 用于匹配预测框和真实框的交并比阈值
        person_class_id 行人类别的ID，默认为1
    """
    model.eval()
    # 读取输入图像，并转化为tensor
    origin_img = Image.open(img_path, mode='r').convert('RGB')

    # SSD300模型需要输入尺寸为300x300的图像
    # img = TF.resize(origin_img, (300, 300))  # 调整图像大小为300x300

    img = TF.to_tensor(origin_img)
    img = img.to(device)

    # 将图像输入神经网络模型中，得到输出
    # img.unsqueeze(0): 添加批次维度，确保输入是模型期望的 [batch_size, C, H, W] 格式，即使批次大小为 1。
    with torch.no_grad():
        output = model(img.unsqueeze(0))
    # output 是一个列表，其中 outputs[0] 包含了这张图片的预测结果。
    preds = output[0]  # 获取图片的预测结果

    pred_labels = output[0]['labels'].cpu().detach().numpy()  # 预测每一个obj的标签
    pred_scores = output[0]['scores'].cpu().detach().numpy()  # 预测每一个obj的得分
    pred_bboxes = output[0]['boxes'].cpu().detach().numpy()  # 预测每一个obj的边框



    # 只选取得分大于0.8的检测结果
    # obj_index = np.argwhere(scores > 0.8).squeeze(axis=1).tolist()
    # 使用ImageDraw将检测到的边框和类别打印在图片中，得到最终的输出
    # draw = ImageDraw.Draw(origin_img)
    # font = ImageFont.truetype('arial.ttf', 15)  # 加载字体文件
    # for i in obj_index:
    #     person_label = labels[i]  # 获取标签
        # # --- 检查标签是否是 'person' ---
        # if person_label == person_id_in_COCO: 
        #     box_loc = bboxes[i].tolist()
        #     label_text = COCO_CLASSES[person_label] 
        #     score = scores[i] 

        #     # 画框
        #     draw.rectangle(xy=box_loc, outline=COLOR_MAP[person_label], width=2) # 增加边框宽度使其更明显

        #     # --- 文本绘制部分 ---
        #     display_text = f"{label_text}: {score:.3f}"

        #     # 获取文本的尺寸，以便绘制背景框
        #     # 获取文本的实际宽度
        #     text_width = font.getlength(display_text)
        #     # 获取文本的实际高度 (通常是 font.getbbox(text)[3] )
        #     # 注意: font.getbbox() 返回的是一个包含 (left, top, right, bottom) 的元组，
        #     # 其中 left 和 top 通常是 0 或负值，right 和 bottom 是文本的宽度和高度的偏移。
        #     # 我们需要的是文本区域的实际高度，这通常是 text_bbox[3] - text_bbox[1]
        #     # 对于 arial.ttf， text_bbox[3] 应该能代表高度的近似值。
        #     text_height = font.getbbox(display_text)[3] # 假设 text_bbox[1] 是 0 或负值，bottom 代表了高度

        #     # 设置标签文本的左上角位置(left, top)
        #     # 通常放在框的左上角上方或内部
        #     text_loc = [box_loc[0] + 2., box_loc[1] - text_height - 2.] # 放在框的上方一点

        #     # 确保文本框不会超出图片顶部
        #     if text_loc[1] < 0:
        #         text_loc[1] = box_loc[1] + 2. # 如果超出顶部，则放在框的右侧
        #     # 设置显示标签的背景框 (left, top, right, bottom)
        #     textbox_loc = [
        #         text_loc[0], text_loc[1],
        #         text_loc[0] + text_width + 4., text_loc[1] + text_height + 2. # 加上一点padding
        #     ]

        #     # 确保textbox不会超出图片边界
        #     img_width, img_height = origin_img.size
        #     textbox_loc[2] = min(textbox_loc[2], img_width)
        #     textbox_loc[3] = min(textbox_loc[3], img_height)

        #     # 绘制标签背景框
        #     draw.rectangle(xy=textbox_loc, fill=COLOR_MAP[person_label])
        #     # 绘制标签文本
        #     draw.text(xy=text_loc, text=display_text, fill='white', font=font)                


    # origin_img.show()
    # origin_img.save("ObjectDetection/ouputs/OpenImages/mygo1_detected.jpg")

if __name__ == '__main__':
    my_detection("ObjectDetection/datasets/OpenImages/mygo1.jpg") 

    # 加载预训练目标检测模型maskrcnn
    model = detection.maskrcnn_resnet50_fpn(pretrained=True)
    # 使用fasterrcnn模型
    # model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # 改为使用 VGG16 骨干的 SSD300
    # model = detection.ssd300_vgg16(pretrained=True) 
    # 使用Retinanet模型
    # model = detection.retinanet_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()



        # 绘制源代码
        # if person_label == person_id_in_COCO:
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