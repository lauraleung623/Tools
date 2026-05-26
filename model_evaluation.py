import torch
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image


def setup_matplotlib():
    """配置中文字体，并在无 GUI 环境下跳过 plt.show()"""
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def show_figure(fig):
    """有 GUI 时显示图表，否则仅保存后关闭"""
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)


def get_feature_vector(model,image_path):
    """
    提取图片的特征向量（倒数第二层输出）
    """
    # model.embed() 方法直接返回高维特征表示 [citation:4]
    # 支持本地路径或 URL
    results = model.embed(image_path)
    
    # results 是一个列表，取第一张图的特征
    # 特征通常是 PyTorch Tensor 格式，先转为 numpy
    features = results[0].cpu().numpy().flatten()

    # embed() 会持久化 predictor 的 embed 参数，后续 predict 需显式关闭
    if model.predictor is not None:
        model.predictor.args.embed = None
    
    return features

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    # 归一化
    norm1 = vec1 / np.linalg.norm(vec1)
    norm2 = vec2 / np.linalg.norm(vec2)
    # 点积
    similarity = np.dot(norm1, norm2)
    return similarity

#------模型预测差异-----#

def load_and_align_images(img1_path, img2_path, target_size=None):
    """
    加载两张可能尺寸不同的图片，并对齐到相同尺寸
    
    Args:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径  
        target_size: 目标尺寸 (h, w)，如果为None则使用两张图片的平均尺寸
    
    Returns:
        对齐后的两张图片 (numpy arrays, 范围0-1)
    """
    # 加载图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    print(f"原始尺寸 - 图片1: {w1}×{h1}, 图片2: {w2}×{h2}")
    
    # 确定目标尺寸
    if target_size is None:
        # 策略1: 使用最大尺寸（保留所有信息）
        target_h = max(h1, h2)
        target_w = max(w1, w2)
    else:
        target_h, target_w = target_size
    
    print(f"对齐后尺寸: {target_w}×{target_h}")
    
    # 对齐函数
    def resize_to_target(img, target_w, target_h):
        h, w = img.shape[:2]
        if (h, w) != (target_h, target_w):
            # 使用高质量的插值算法
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return img / 255.0  # 归一化到 [0, 1]
    
    img1_aligned = resize_to_target(img1, target_w, target_h)
    img2_aligned = resize_to_target(img2, target_w, target_h)
    
    return img1_aligned, img2_aligned, (target_h, target_w), (h1, w1), (h2, w2)

def interpolate_images(img1, img2, alpha):
    """在两张图像之间进行线性插值"""
    return img1 * (1 - alpha) + img2 * alpha

def analyze_detection_with_size_variation(model, img1_path, img2_path, num_steps=20, target_size=None):
    """
    分析目标检测模型在两张可能尺寸不同的相似图像间的决策边界
    
    特殊处理:
    - 自动对齐图片尺寸
    - 记录尺寸差异对检测的影响
    - 分别测试原尺寸和对齐尺寸的效果
    """
    # 1. 加载并对齐图片
    img1_aligned, img2_aligned, aligned_size, (h1, w1), (h2, w2) = load_and_align_images(
        img1_path, img2_path, target_size
    )
    
    # 2. 可选：同时测试原始尺寸（仅模型推理，不用于插值）
    print("/n--- 原始尺寸推理结果（参考）---")
    original_results = {}
    for path, name in [(img1_path, "图片1"), (img2_path, "图片2")]:
        pred = model.predict(path, embed=None)[0]
        if pred.boxes is not None and len(pred.boxes) > 0:
            top = pred.boxes[0]
            top_cls = int(top.cls.item())
            top_conf = float(top.conf.item())
            print(f"{name}: 检测到 {pred.names[top_cls]}, 置信度 {top_conf:.4f}")
            original_results[name] = {
                'class': top_cls,
                'confidence': top_conf,
                'num_boxes': len(pred.boxes)
            }
        else:
            print(f"{name}: 未检测到目标")
            original_results[name] = None
    
    # 3. 插值实验（使用对齐后的图片）
    print("/n--- 开始插值实验（对齐尺寸后）---")
    results = []
    alphas = np.linspace(0, 1, num_steps)
    
    for alpha in alphas:
        mixed_img = interpolate_images(img1_aligned, img2_aligned, alpha)
        mixed_img_uint8 = (mixed_img * 255).astype(np.uint8)
        
        # 检测预测
        pred = model.predict(mixed_img_uint8, embed=None)[0]
        
        # 提取检测结果
        detections = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):
                detections.append({
                    'class': int(cls.item()),
                    'confidence': float(conf.item()),
                    'bbox': box.tolist()
                })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        top_detection = detections[0] if detections else None
        
        results.append({
            'alpha': alpha,
            'top_confidence': top_detection['confidence'] if top_detection else 0,
            'top_class': top_detection['class'] if top_detection else -1,
            'num_detections': len(detections),
            'has_object': len(detections) > 0,
            'all_detections': detections
        })
        
        # 进度显示
        if int(alpha * num_steps) % (num_steps // 5) == 0:
            print(f"  进度: {alpha:.1%} - 检测到 {len(detections)} 个目标")
    
    return results, alphas, original_results, aligned_size, (h1, w1), (h2, w2)

def diagnose_with_size_awareness(results, alphas, original_results, aligned_size):
    """
    考虑尺寸因素的诊断
    """
    confidences = [r['top_confidence'] for r in results]
    has_obj = [1 if r['has_object'] else 0 for r in results]
    num_dets = [r['num_detections'] for r in results]
    
    # 检测突变点
    sudden_jumps = 0
    jump_positions = []
    for i in range(1, len(confidences)):
        diff = abs(confidences[i] - confidences[i-1])
        if diff > 0.3:
            sudden_jumps += 1
            jump_positions.append((alphas[i], diff))
    
    # 检测目标消失/重现次数
    obj_flips = sum(abs(has_obj[i] - has_obj[i-1]) for i in range(1, len(has_obj)))
    
    print("=" * 60)
    print("目标检测模型决策边界诊断报告（考虑尺寸差异）")
    print("=" * 60)
    print(f"对齐后图片尺寸: {aligned_size[1]}×{aligned_size[0]}")
    print(f"原始尺寸差异: {original_results['图片1'] != original_results['图片2']}")
    print("-" * 60)
    print(f"置信度变化标准差: {np.std(confidences):.4f}")
    print(f"突变点数量 (>0.3跳变): {sudden_jumps}")
    if jump_positions:
        print(f"  突变位置: {jump_positions[:3]}")
    print(f"目标出现/消失次数: {obj_flips}")
    print(f"检测框数量波动: {np.std(num_dets):.2f}")
    
    # 检查原始尺寸推理是否一致
    orig_consistent = False
    if original_results['图片1'] and original_results['图片2']:
        same_class = original_results['图片1']['class'] == original_results['图片2']['class']
        conf_diff = abs(original_results['图片1']['confidence'] - original_results['图片2']['confidence'])
        orig_consistent = same_class and conf_diff < 0.2
        print(f"原始尺寸预测一致性: {'一致' if same_class else '不一致'}")
        print(f"  预测类别: {original_results['图片1']['class']} vs {original_results['图片2']['class']}")
        print(f"  置信度差异: {conf_diff:.4f}")
    
    print("-" * 60)
    
    # 综合诊断（考虑尺寸因素）
    if original_results['图片1'] and original_results['图片2'] and not orig_consistent:
        print("⚠️ **首要问题：原始推理结果就不一致**")
        print("   建议：先检查这两张图是否需要不同标签，或是否存在标注错误")
        print("   如果标签正确，问题可能是：")
        print("   1. 模型对尺寸敏感 → 需要尺寸相关的数据增强（随机缩放）")
        print("   2. 图片本身差异确实较大 → 重新审视相似性判断")
    elif sudden_jumps >= 2 or obj_flips >= 2:
        print("🔴 **诊断结果：决策边界过于尖锐**")
        print("   - 模型存在多处不连续跳变")
        print("   - 微小输入变化导致目标出现/消失")
        if aligned_size[0] > 1000:
            print("   - 注意：对齐到大尺寸会增加计算量，可能放大边界问题")
        print("   - 建议：谱归一化 + 标签平滑 + 更强的数据增强")
    elif np.std(num_dets) > 1.5:
        print("🟡 **诊断结果：对微小扰动过于敏感**")
        print("   - 检测框数量不稳定，在相似图之间波动")
        print("   - 建议：高斯模糊 + 噪声增强 + 降低NMS阈值")
    elif np.std(confidences) > 0.2:
        print("🟠 **诊断结果：中度不稳定**")
        print("   - 置信度波动明显，但无剧烈跳变")
        print("   - 建议：数据增强 + 测试时增强")
    else:
        print("✅ **诊断结果：模型表现相对稳定**")
        if not orig_consistent:
            print("   - 但原始尺寸预测不一致，可能是尺寸差异导致")
            print("   - 建议：使用统一尺寸输入，或在训练中添加随机缩放")
    
    return {
        'sudden_jumps': sudden_jumps,
        'obj_flips': obj_flips,
        'confidence_std': np.std(confidences),
        'det_volatility': np.std(num_dets),
        'original_consistent': orig_consistent,
        'diagnosis': 'sharp' if sudden_jumps >= 2 else ('sensitive' if np.std(num_dets) > 1.5 else 'stable')
    }

# 可视化（考虑尺寸信息的版本）
def visualize_with_size_info(results, alphas, original_results, aligned_size, img1_size, img2_size):
    """生成包含尺寸信息的可视化报告"""
    setup_matplotlib()
    confidences = [r['top_confidence'] for r in results]
    has_obj = [1 if r['has_object'] else 0 for r in results]
    num_dets = [r['num_detections'] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 左列：插值结果
    axes[0, 0].plot(alphas, confidences, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_xlabel('插值系数 α (0=图像1, 1=图像2)')
    axes[0, 0].set_ylabel('最高置信度')
    axes[0, 0].set_title(f'检测置信度变化 (对齐后 {aligned_size[1]}×{aligned_size[0]})')
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim([0, 1])
    
    axes[0, 1].plot(alphas, has_obj, 'g-', linewidth=2, drawstyle='steps-post')
    axes[0, 1].set_xlabel('插值系数 α')
    axes[0, 1].set_ylabel('是否检测到目标 (1=是, 0=否)')
    axes[0, 1].set_title('目标存在性变化')
    axes[0, 1].set_ylim([-0.1, 1.1])
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(alphas, num_dets, 'r-', linewidth=2, marker='s')
    axes[0, 2].set_xlabel('插值系数 α')
    axes[0, 2].set_ylabel('检测框数量')
    axes[0, 2].set_title('检测框数量波动')
    axes[0, 2].grid(True)
    
    # 第二行：原始信息
    h1, w1 = img1_size
    h2, w2 = img2_size
    axes[1, 0].axis('off')
    axes[1, 0].set_title('原始尺寸对比')
    info_text = f"图片1尺寸: {w1}×{h1} (原始)/n图片2尺寸: {w2}×{h2} (原始)/n"
    info_text += f"图片1检测: {original_results['图片1']['confidence']:.3f}/n" if original_results['图片1'] else "图片1: 未检测到/n"
    info_text += f"图片2检测: {original_results['图片2']['confidence']:.3f}" if original_results['图片2'] else "图片2: 未检测到"
    axes[1, 0].text(0.1, 0.5, info_text, fontsize=10, va='center')
    
    axes[1, 1].axis('off')
    axes[1, 1].set_title('诊断建议')
    diagnosis_text = "决策边界突变数量: {}/n目标消失次数: {}/n置信度稳定性: {:.3f}".format(
        sum(abs(confidences[i] - confidences[i-1]) > 0.3 for i in range(1, len(confidences))),
        sum(abs(has_obj[i] - has_obj[i-1]) for i in range(1, len(has_obj))),
        np.std(confidences)
    )
    axes[1, 1].text(0.1, 0.5, diagnosis_text, fontsize=10, va='center')
    
    # 空白区域
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_size_aware_analysis.png', dpi=150, bbox_inches='tight')
    show_figure(fig)


if __name__ == "__main__":
    model = YOLO("D:/ReferenceCode/ultralytics-main/ultralytics-main/runs/detect/train18/weights/best.pt")     

    # Mark 计算两幅图特诊相近程度
    model.eval() # 切换到评估模式
    img1_path = "D:/dissolution/myself/Apr16/19_2818-69-1_64-17-5/Apr16_19_2818-69-1_64-17-5_a2_back.bmp"
    img2_path = "D:/dissolution/myself/Apr15/6_2237-30-1_68-12-2/Apr15_2237-30-1&68-12-2_a3_back.bmp"

    # 提取特征
    vec1 = get_feature_vector(model,img1_path)
    vec2 = get_feature_vector(model,img2_path)

    # 打印维度，确认提取成功
    print(f"特征向量维度: {vec1.shape}")

    # 计算相似度
    sim_score = cosine_similarity(vec1, vec2)

    # 结果解释：范围在 [-1, 1] 之间，越接近 1 代表两张图在高维空间中越相似
    print(f"图片相似度: {sim_score:.4f}")

    # Mark 线性插值探测决策边界 
    img1_path="D:/dissolution/myself/Apr16/19_2818-69-1_64-17-5/Apr16_19_2818-69-1_64-17-5_a2_back.bmp"
    img2_path="D:/dissolution/myself/Apr15/6_2237-30-1_68-12-2/Apr15_2237-30-1&68-12-2_a3_back.bmp"
    
    results, alphas, original_results, aligned_size, img1_size, img2_size = analyze_detection_with_size_variation(
        model, 
        img1_path,   # 可能是 640×640
        img2_path,   # 可能是 641×638
        num_steps=20,
        target_size=None  # None 表示自动对齐到最大尺寸；也可手动指定如 (640, 640)
    )
    
    # 诊断
    diagnosis = diagnose_with_size_awareness(results, alphas, original_results, aligned_size)
    
    # 可视化
    visualize_with_size_info(results, alphas, original_results, aligned_size, img1_size, img2_size)
    
    print(f"/n最终诊断: {diagnosis['diagnosis']}")
