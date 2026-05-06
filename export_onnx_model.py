from ultralytics import YOLO
import torch 
import cv2
from pathlib import Path
import argparse
import zlib
# 使用3.11.9ML环境

def _clamp(v: float, lo: int, hi: int) -> int:
    return max(lo, min(int(round(v)), hi))
def _xyxy_from_values(values: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """
    将一行数字解析为裁剪框，支持以下格式:
    - YOLO 检测框: cls cx cy w h (归一化)
    - 矩形框: x1 y1 x2 y2 (可归一化/可绝对像素)
    - 多点坐标: cls x1 y1 x2 y2 ... 或 x1 y1 x2 y2 ... (取外接矩形)
    """
    if len(values) < 4:
        return None

    # 1) YOLO: cls cx cy w h
    if len(values) == 5 and all(0.0 <= v <= 1.0 for v in values[1:]):
        cx, cy, bw, bh = values[1], values[2], values[3], values[4]
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
    else:
        # 2) xyxy (4值) 或 3) 多点坐标 (>=6值)
        pts = values
        if len(values) >= 5 and (len(values) - 1) % 2 == 0:
            # 常见分割标签: cls + x1 y1 x2 y2 ...
            pts = values[1:]

        if len(pts) == 4:
            x1, y1, x2, y2 = pts
            if all(0.0 <= v <= 1.0 for v in pts):
                x1, x2 = x1 * img_w, x2 * img_w
                y1, y2 = y1 * img_h, y2 * img_h
        elif len(pts) >= 6 and len(pts) % 2 == 0:
            xs = pts[0::2]
            ys = pts[1::2]
            if all(0.0 <= v <= 1.0 for v in pts):
                xs = [x * img_w for x in xs]
                ys = [y * img_h for y in ys]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            return None

    left = _clamp(min(x1, x2), 0, img_w - 1)
    top = _clamp(min(y1, y2), 0, img_h - 1)
    right = _clamp(max(x1, x2), 0, img_w - 1)
    bottom = _clamp(max(y1, y2), 0, img_h - 1)

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
def _xyxy_from_values_wmargin(values: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """
    仅考虑yolo的情况，截取20%上下边缘
    """
    if len(values) == 5 and all(0.0 <= v <= 1.0 for v in values[1:]):
        cx, cy, bw, bh = values[1], values[2], values[3], values[4]
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - (bh / 2)*1.2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + (bh / 2)*1.2) * img_h
    else:
        print(f"不是yolo格式: {values}")
        return None
    
    left = _clamp(min(x1, x2), 0, img_w - 1)
    top = _clamp(min(y1, y2), 0, img_h - 1)
    right = _clamp(max(x1, x2), 0, img_w - 1)
    bottom = _clamp(max(y1, y2), 0, img_h - 1)

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom

def crop_from_labels(labels_dir: Path,images_dir: Path,
    output_dir: Path,image_suffix: str = ".bmp") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(labels_dir.glob("*.txt")) #MARK 生成器，包含所有*.tx的文件/目录
    if not txt_files:
        print(f"未找到标签文件: {labels_dir}")
        return

    total_saved = 0
    for txt_path in txt_files:
        img_path = images_dir / f"{txt_path.stem}{image_suffix}"
        if not img_path.exists():
            print(f"跳过（未找到同名图片）: {img_path.name}")
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"跳过（图片读取失败）: {img_path}")
            continue
        img_h, img_w = image.shape[:2]

        lines = txt_path.read_text(encoding="utf-8").splitlines()
        crop_idx = 1
        for line_no, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                values = [float(x) for x in line.split()]
            except ValueError:
                print(f"跳过（非法数字）: {txt_path.name}:{line_no}")
                continue

            box = _xyxy_from_values(values, img_w, img_h)
            if box is None:
                print(f"跳过（无法解析坐标）: {txt_path.name}:{line_no}")
                continue

            left, top, right, bottom = box
            cropped = image[top:bottom, left:right]
            if cropped.size == 0:
                print(f"跳过（裁剪为空）: {txt_path.name}:{line_no}")
                continue

            out_name = f"{txt_path.stem}_{crop_idx}.bmp"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), cropped)
            total_saved += 1
            crop_idx += 1

        print(f"处理完成: {txt_path.name} -> {crop_idx - 1} 个裁剪结果")

    print(f"\n总计保存: {total_saved} 张裁剪图")
def crop_from_labels_wmargin(labels_dir: Path,images_dir: Path,
    output_dir: Path,image_suffix: str = ".bmp") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(labels_dir.glob("*.txt")) #MARK 生成器，包含所有*.tx的文件/目录
    if not txt_files:
        print(f"未找到标签文件: {labels_dir}")
        return

    total_saved = 0
    for txt_path in txt_files:
        img_path = images_dir / f"{txt_path.stem}{image_suffix}"
        if not img_path.exists():
            print(f"跳过（未找到同名图片）: {img_path.name}")
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"跳过（图片读取失败）：{img_path}")
            continue
        img_h,img_w=image.shape[:2]

        lines = txt_path.read_text(encoding="utf-8").splitlines()
        crop_idx=1
        for line_no, line in enumerate(lines,start=1):
            line= line.strip()
            if not line:
                continue
            try:
                values = [float(x) for x in line.split()]
            except ValueError:
                print(f"跳过（非法数字：{txt_path.name}:{line_no}")
                continue
            
            box = _xyxy_from_values_wmargin(values,img_w,img_h)  #TODO 这里是否要改成，
            if box is None:
                print(f"跳过(无法解析坐标): {txt_path.name}:{line_no}")
                continue

            left,top,right,bottom=box
            cropped=image[top:bottom,left:right]
            if cropped.size == 0:
                print(f"跳过(裁剪为空): {txt_path.name}:{line_no}")
                continue
            
            out_name = f"{txt_path.stem}_{crop_idx}.bmp"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path),cropped)
            total_saved += 1
            crop_idx += 1

        print(f"处理完成: {txt_path.name} -> {crop_idx - 1} 个裁剪结果")

    print(f"\n总计保存: {total_saved} 张裁剪图")
    return

def main_crop_from_labels(local_overrides: dict | None = None) -> None:
    # 主流程默认变量: 不传命令行参数时，直接使用这些值运行
    defaults = {
        "labels": Path("./labels"),
        "images": Path("./images"),
        "output": Path("./outputs"),
        "ext": ".bmp",
    }

    if local_overrides:
        defaults.update(local_overrides)

    parser = argparse.ArgumentParser(description="根据同名 txt 标签裁剪 bmp 图片")
    parser.add_argument("--labels", type=Path, default=defaults["labels"], help="txt 标签文件目录")
    parser.add_argument("--images", type=Path, default=defaults["images"], help="bmp 图片目录")
    parser.add_argument("--output", type=Path, default=defaults["output"], help="裁剪结果输出目录")
    parser.add_argument("--ext", type=str, default=defaults["ext"], help="图片扩展名，默认 .bmp")
    args = parser.parse_args()
    print(f"生效参数: labels={args.labels}, images={args.images}, output={args.output}, ext={args.ext}")

    crop_from_labels(args.labels, args.images, args.output, args.ext)
def main_crop_from_labels_wmargin(local_overrides: dict | None = None):
    # 主流程默认变量: 不传命令行参数时，直接使用这些值运行
    defaults = {
        "labels": Path("./labels"),
        "images": Path("./images"),
        "output": Path("./outputs"),
        "ext": ".bmp",
    }

    if local_overrides:
        defaults.update(local_overrides)

    parser = argparse.ArgumentParser(description="根据同名 txt 标签裁剪 bmp 图片")
    parser.add_argument("--labels", type=Path, default=defaults["labels"], help="txt 标签文件目录")
    parser.add_argument("--images", type=Path, default=defaults["images"], help="bmp 图片目录")
    parser.add_argument("--output", type=Path, default=defaults["output"], help="裁剪结果输出目录")
    parser.add_argument("--ext", type=str, default=defaults["ext"], help="图片扩展名，默认 .bmp")
    args = parser.parse_args()
    print(f"生效参数: labels={args.labels}, images={args.images}, output={args.output}, ext={args.ext}")

    crop_from_labels_wmargin(args.labels, args.images, args.output, args.ext)

def crop_from_model(image_folder,model_path):
    image_folder = Path(image_folder)
    # 模型路径
    model = YOLO(model_path)
    
    # 保存路径
    save_dir = image_folder.parent / "crops"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = model(image_folder, save=False)

    total_saved = 0
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print(f"未检测到目标: {Path(result.path).name}")
            continue

        best_idx = int(torch.argmax(boxes.conf).item())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

        image = cv2.imread(str(result.path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"图片读取失败: {result.path}")
            continue

        img_h, img_w = image.shape[:2]
        left = _clamp(min(x1, x2), 0, img_w - 1)
        top = _clamp(min(y1, y2), 0, img_h - 1)
        right = _clamp(max(x1, x2), 0, img_w - 1)
        bottom = _clamp(max(y1, y2), 0, img_h - 1)
        if right <= left or bottom <= top:
            print(f"最高分框无效，跳过: {Path(result.path).name}")
            continue

        cropped = image[top:bottom, left:right]
        if cropped.size == 0:
            print(f"裁剪为空，跳过: {Path(result.path).name}")
            continue

        out_name = f"{Path(result.path).stem}.bmp"
        out_path = save_dir / out_name
        cv2.imwrite(str(out_path), cropped)
        total_saved += 1
        print(f"已保存最高分裁剪: {out_path.name}, conf={float(boxes.conf[best_idx]):.4f}")

    print(f"完成，最高分裁剪共保存 {total_saved} 张到: {save_dir}")

def crop_left_400_pixels(src_folder,output_folder):
    src_path = Path(src_folder)
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    if not src_path.exists() or not src_path.is_dir():
        print(f"源目录不存在或不是文件夹: {src_path}")
        return

    bmp_files = sorted(
        p for p in src_path.iterdir()
        if p.is_file() and p.suffix.lower() == ".bmp"
    )
    if not bmp_files:
        print(f"未找到 bmp 图片: {src_path}")
        return

    total_saved = 0
    for img_path in bmp_files:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"跳过（图片读取失败）: {img_path.name}")
            continue

        img_h, img_w = image.shape[:2]
        crop_w = min(400, img_w)
        cropped = image[0:img_h, 0:crop_w]
        if cropped.size == 0:
            print(f"跳过（裁剪为空）: {img_path.name}")
            continue

        save_path = out_path / img_path.name
        cv2.imwrite(str(save_path), cropped)
        total_saved += 1

    print(f"处理完成，共保存 {total_saved} 张图片到: {out_path}")

def draw_bounding_boxes(image_folder,label_folder,image_dst_folder):
    def _color_for_label(label_name: str) -> tuple[int, int, int]:
        # 基于标签名生成稳定颜色（BGR）
        seed = zlib.crc32(label_name.encode("utf-8"))
        b = 60 + (seed & 0xFF) % 160
        g = 60 + ((seed >> 8) & 0xFF) % 160
        r = 60 + ((seed >> 16) & 0xFF) % 160
        return int(b), int(g), int(r)

    def _label_name_from_values(values: list[float]) -> str:
        # 常见标签格式中，第一个值是类别 id
        if len(values) >= 5:
            cls_val = values[0]
            if abs(cls_val - round(cls_val)) < 1e-6:
                return f"class_{int(round(cls_val))}"
            return f"class_{cls_val:g}"
        return "bbox"

    image_folder = Path(image_folder)
    label_folder = Path(label_folder)
    image_dst_folder = Path(image_dst_folder)
    image_dst_folder.mkdir(parents=True, exist_ok=True)
    image_files = sorted(image_folder.glob("*.bmp"))
    missing_label_count = 0
    for image_file in image_files:
        label_file = label_folder / f"{image_file.stem}.txt"
        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        if not label_file.exists():
            missing_label_count += 1
            print(f"跳过（未找到同名标签）: {image_file.name}")
            out_path = image_dst_folder / image_file.name
            cv2.imwrite(str(out_path), image)
            continue
        label = label_file.read_text(encoding="utf-8").splitlines()
        img_h, img_w = image.shape[:2]
        for line in label:
            line = line.strip()
            if not line:
                continue
            try:
                values = [float(x) for x in line.split()]
            except ValueError:
                print(f"跳过（非法数字）: {label_file.name}")
                continue

            label_name = _label_name_from_values(values)
            color = _color_for_label(label_name)
            box = _xyxy_from_values(values, img_w, img_h)
            if box is None:
                print(f"跳过(无法解析坐标): {label_file.name}")
                continue
            left,top,right,bottom = box
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            text_y = top - 8 if top > 18 else top + 18
            cv2.putText(
                image,
                label_name,
                (left, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        out_path = image_dst_folder / image_file.name
        cv2.imwrite(str(out_path), image)
    print(f"未找到同名标签的 bmp 数量: {missing_label_count}")
    return


if __name__ == "__main__":
    
    # MARK 根据模型，剪除瓶子
    # image_folder=r"D:/dissolution/testing_engineer/Apr24/31#_H1_forcrop/Apr24_957476-07-2_64-17-5_A0_back.bmp"
    # model_path="./models/vial_best_Mar25_11pm.pt"
    # crop_from_model(image_folder,model_path)

    # MARK 根据标签，裁剪出瓶子
    # override={
    #     "labels": Path(r"D:\dissolution\vials\vials2\full_image\labels"),
    #     "images": Path(r"D:\dissolution\vials\vials2\full_image\images"),
    #     "output": Path(r"D:\dissolution\vials\vials2\crops\images"),
    #     # "ext": ".bmp",  # 注释掉表示使用默认值
    # }
    # main_crop_from_labels(local_overrides=override)


    # MARK 根据标签，裁剪出带20%边缘余量的瓶子图像
    # override={
    #     "labels": Path(r"D:\dissolution\vials\vials1n2nsuspension\full_image_wcls\labels\val\air"),
    #     "images": Path(r"D:\dissolution\vials\vials1n2nsuspension\full_image_wcls\images\val\air"),
    #     "output": Path(r"D:\dissolution\vials\vials1n2nsuspension\crops_wmargin\val\air"),
    #     # "ext": ".bmp",  # 注释掉表示使用默认值
    # }
    # main_crop_from_labels_wmargin(local_overrides=override)

    # MARK 截取左边400像素图片
    # src_folder=r"D:\dissolution\vials\vials1n2nsuspension\crops_wmargin\val\unclear"
    # output_folder=r"D:\dissolution\vials\vials1n2nsuspension\bottle_cap_crops\images\val\unclear"
    # crop_left_400_pixels(src_folder,output_folder)

    # MARK 根据标签，画出边界框
    # TODO 加上标签名字
    image_src_folder=r"D:\dissolution\vials\vials1n2nsuspension\crops\images\val\air"
    label_folder=r"D:\dissolution\vials\vials1n2nsuspension\crops\labels\val\air"
    image_dst_folder=r"D:\dissolution\vials\vials1n2nsuspension\crops\imgwlabels\val_air_bbox"
    draw_bounding_boxes(image_src_folder,label_folder,image_dst_folder) 
