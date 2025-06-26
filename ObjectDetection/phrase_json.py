import json
import os

# 你需要知道这个图片的 image_id。一种方法是预先创建一个 filename -> image_id 的映射，
# 或者直接在标注文件中查找。这里我们假设你已经知道了 test_image_filename 对应的 image_id。
# 如果你没有这个映射，可以先加载标注文件，然后遍历其 'images' 部分来查找。

# --- COCO JSON 解析函数 ---
def parse_coco_annotations(annotation_file,img_filename):
    """
    解析 COCO 标注文件，找到指定图片的所有行人标注。
    Args:
        annotation_file (str): COCO 标注 JSON 文件路径。
        target_image_filename (str): 你要查找标注的图片文件名。
    Returns:
        tuple: (list of boxes, list of labels) for 'person' detections in the target image.
               如果找不到图片或图片中没有行人，则返回 ([], [])。
    """
    try:
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到所需标注文件: {annotation_file}")
        return [], []

    # 查找目标图片的 image_id
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


# 假设你已经有了 image_file_path，并且知道它在哪个 COCO 集合里
# 并且有对应的 annotation_file 路径
img_filepath = "ObjectDetection/datasets/coco/coco-2017/validation/data/000000018380.jpg"
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
else:
    print("没有找到行人标注或图片不存在。请检查文件名和路径。")    
