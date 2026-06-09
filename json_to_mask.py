import json
import cv2
import numpy as np
from labelme import utils

"""
将polygon的json文件，转化为png格式mask，仅包含1种mask
"""

def json_to_mask(json_path, output_mask_path):
    """
    新版 LabelMe JSON 转 mask（兼容新版本 API）
    """
    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    h, w = data['imageHeight'], data['imageWidth']
    shapes = data.get('shapes', [])
    
    # 获取所有唯一的标签名称
    label_names = []
    label_to_value = {}
    
    for shape in shapes:
        label = shape.get('label', '__background__')
        if label not in label_names and label != '__background__':
            label_names.append(label)
    
    # 构建标签到数值的映射（背景为 0，前景从 1 开始）
    label_to_value['__background__'] = 0
    for idx, label in enumerate(label_names, start=1):
        label_to_value[label] = 255  # FIXME 尝试这样改，是否能够出现白色
    
    print(f"📋 标签映射: {label_to_value}")
    
    # 调用新版 shapes_to_label
    lbl, _ = utils.shapes_to_label(
        img_shape=(h, w),
        shapes=shapes,
        label_name_to_value=label_to_value
    )
    
    # 保存 mask（uint8 格式）
    cv2.imwrite(output_mask_path, lbl.astype(np.uint8))
    print(f"✅ Mask 已保存至: {output_mask_path}")
    print(f"📊 标签数值范围: {lbl.min()} ~ {lbl.max()}")
    
    return lbl

# ========== 使用示例 ==========
if __name__ == "__main__":
    folder_path=r'D:/ReferenceCode/patchcore-inspection-main/mvtec/vail_trail1/ground_truth/defect'
    json_file = folder_path+"/"+r"Apr15_9_148893-10-1_27522_a0_back.json"      # 修改为你的 JSON 路径
    output_file = folder_path+"/"+r"Apr15_9_148893-10-1_27522_a0_back_mask.png"   # 修改为输出路径
    
    json_to_mask(json_file, output_file)
