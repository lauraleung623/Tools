import subprocess
import sys
import json
from pathlib import Path
from PIL import Image
import os

def rename_files_remove_suffix_after_underscore(directory: str) -> None:
    """
    将文件名中最后一个 '_' 及其后面的部分删除，保留扩展名。
    示例：abc_123_45.txt -> abc_123.txt
    """
    directory=Path(directory)
    if not directory.exists() or not directory.is_dir():
        print(f"目录不存在或不是文件夹: {directory}")
        return

    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue

        stem = file_path.stem
        if "_" not in stem:
            continue

        last_underscore_idx = stem.rfind("_")
        new_stem = stem[:last_underscore_idx]
        if not new_stem:
            print(f"跳过（新文件名为空）: {file_path.name}")
            continue

        new_path = file_path.with_name(f"{new_stem}{file_path.suffix}")

        # 避免重名覆盖
        if new_path.exists() and new_path != file_path:
            idx = 1
            while True:
                candidate = file_path.with_name(f"{new_stem}_{idx}{file_path.suffix}")
                if not candidate.exists():
                    new_path = candidate
                    break
                idx += 1

        file_path.rename(new_path)
        print(f"{file_path.name} -> {new_path.name}")
    return

def run_lablelme2yolo(json_dir,val_size=1):
    """
    labelme2yolo is a tool to convert labelme annotations to YOLO format.
    """
    output_dir=Path(json_dir).parent / "yolo_labels"
    output_dir.mkdir(parents=True,exist_ok=True)
    cmd=["labelme2yolo",
    "--json_dir",json_dir,
    "--val_size",str(val_size),
    "--output_dir",str(output_dir)]
    subprocess.run(cmd)
    print(f"YOLO labels saved to: {output_dir}")
    return str(output_dir)

def run_yolo2labelme(yolo_label_dir, image_dir, class_names, output_dir=None):
    """
    将YOLO txt标签反向转换为Labelme json标签。

    参数：
    - yolo_label_dir: YOLO标签目录（*.txt）
    - image_dir: 对应图片目录
    - class_names: 类别名称列表，索引即YOLO的class_id
    - output_dir: 输出json目录，默认在yolo_label_dir同级创建labelme_json
    """
    yolo_label_dir = Path(yolo_label_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) if output_dir else yolo_label_dir.parent / "labelme_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not yolo_label_dir.exists() or not yolo_label_dir.is_dir():
        print(f"YOLO标签目录不存在或不是文件夹: {yolo_label_dir}")
        return None

    if not image_dir.exists() or not image_dir.is_dir():
        print(f"图片目录不存在或不是文件夹: {image_dir}")
        return None

    image_exts = [".bmp", ".jpg", ".jpeg", ".png", ".webp"]
    converted = 0

    for txt_path in yolo_label_dir.iterdir():
        if not txt_path.is_file() or txt_path.suffix.lower() != ".txt":
            continue

        stem = txt_path.stem
        image_path = None
        for ext in image_exts:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"跳过（未找到同名图片）: {txt_path.name}")
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        shapes = []
        lines = txt_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"跳过异常行（字段不足）: {txt_path.name} -> {line}")
                continue

            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= len(class_names):
                print(f"跳过异常类别（越界）: {txt_path.name} -> {line}")
                continue

            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            box_w = float(parts[3]) * img_w
            box_h = float(parts[4]) * img_h

            x1 = x_center - box_w / 2
            y1 = y_center - box_h / 2
            x2 = x_center + box_w / 2
            y2 = y_center + box_h / 2

            shape = {
                "label": class_names[class_id],
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                #"mask": None
            }
            shapes.append(shape)

        labelme_obj = {
            "version": "5.4.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.relpath(image_path, output_dir).replace("\\", "/"),
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w
        }

        output_json = output_dir / f"{stem}.json"
        output_json.write_text(
            json.dumps(labelme_obj, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        converted += 1

    print(f"Labelme labels saved to: {output_dir} (共转换 {converted} 个文件)")
    return str(output_dir)

def dataset_split(output_dir):
    """
    按比例拆分数据集训练集
    """
    return

def files_comparsion(image_dir,label_dir):
    """
    比较文件夹中的文件是否相同
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    if not image_dir.exists() or not image_dir.is_dir():
        print(f"image_dir不存在或不是文件夹: {image_dir}")
        return

    if not label_dir.exists() or not label_dir.is_dir():
        print(f"label_dir不存在或不是文件夹: {label_dir}")
        return

    bmp_files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == ".bmp"]
    label_stems = {p.stem for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

    missing_label_bmps = [bmp for bmp in bmp_files if bmp.stem not in label_stems]

    if not missing_label_bmps:
        print("检查完成：image_dir中的bmp文件均有同名txt标签。")
        return

    print("以下bmp文件在label_dir中不存在同名txt：")
    for bmp_path in missing_label_bmps:
        print(str(bmp_path))

    print(f"共 {len(missing_label_bmps)} 个bmp缺少同名txt标签。")
    return

if __name__ == "__main__":
    # 将labelme_json转换为yolo_labels
    json_dir=r'D:/dissolution/vials/vials1n2nsuspension/crops/labels/val_json/air'
    output_dir=run_lablelme2yolo(json_dir,val_size=1)
    #假设全部label都存储在val里
    target_dir=output_dir+"/labels/val"
    rename_files_remove_suffix_after_underscore(target_dir)


    # 将yolo_label转为labelme_json
    # yolo_label_dir=r'D:/dissolution/vials/vials1n2nsuspension/crops/labels/val/unclear'
    # image_dir=r'D:/dissolution/vials/vials1n2nsuspension/crops/images/val/unclear'
    # class_names=['interface','substance']
    # output_dir=run_yolo2labelme(yolo_label_dir,image_dir,class_names)

    # 对比的label和bmp
    #image_dir=r'D:\dissolution\vials\vials1n2nsuspension\crops\images\train'
    #label_dir=r'D:\dissolution\vials\vials1n2nsuspension\crops\labels\train'
    #files_comparsion(image_dir,label_dir)
