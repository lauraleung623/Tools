"""
对比ultralytics和onnxruntime的分类结果
"""

import onnxruntime as ort
import errors
from pathlib import Path
import json
from typing import Final
import os
import argparse
import csv
from datetime import datetime
import math
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationPredictor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROJECT_ROOT = Path(__file__).resolve().parent


class ClassifyLetterBox:
    """分类 letterbox：等比缩放 + 灰边(114) 填充，行为对齐 Ultralytics ClassifyLetterBox。"""

    def __init__(self, size=(640, 640), auto=False, stride=32):
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto
        self.stride = stride

    def __call__(self, im: np.ndarray) -> np.ndarray:
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)
        h, w = round(imh * r), round(imw * r)

        if self.auto:
            hs = math.ceil(h / self.stride) * self.stride
            ws = math.ceil(w / self.stride) * self.stride
        else:
            hs, ws = self.h, self.w

        top = round((hs - h) / 2 - 0.1)
        left = round((ws - w) / 2 - 0.1)
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(
            im, (w, h), interpolation=cv2.INTER_LINEAR
        )
        return im_out


class LetterBoxClassificationPredictor(ClassificationPredictor):
    """与训练一致的 letterbox(灰边114) + ImageNet normalize 分类推理。"""
    """与Aug13预训练一致"""
    def setup_source(self, source):
        """Use BasePredictor source setup; letterbox is applied in preprocess."""
        from ultralytics.engine.predictor import BasePredictor

        BasePredictor.setup_source(self, source)
        imgsz = self.imgsz[0] if isinstance(self.imgsz, (list, tuple)) else self.imgsz
        self._letterbox = ClassifyLetterBox(size=imgsz)
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def preprocess(self, img):
        """BGR list/ndarray -> letterbox RGB -> normalize tensor."""
        if not isinstance(img, torch.Tensor):
            batch = []
            for im in img:
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                out = self._letterbox(rgb)
                t = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).float() / 255.0
                batch.append(t)
            img = torch.stack(batch, dim=0)
            img = img.to(self.model.device)
            img = (img - self._mean.to(img.device)) / self._std.to(img.device)
        else:
            img = img.to(self.model.device)
        return img.half() if self.model.fp16 else img.float()


class TaskRunner:

    def __init__(self):
        self._ultra_models = {}

    def _resolve_classify_imgsz(self, imgsz=None, export_imgsz=None, default=224):
        """解析分类推理尺寸；显式 imgsz 优先于 ONNX metadata。"""
        if imgsz is not None:
            if isinstance(imgsz, int):
                return imgsz, imgsz
            return int(imgsz[0]), int(imgsz[1])
        if export_imgsz:
            return export_imgsz
        return default, default

    def _get_ultra_model(self, model_path):
        if model_path not in self._ultra_models:
            self._ultra_models[model_path] = YOLO(model_path, task="classify")
        return self._ultra_models[model_path]

    #---------------------------------识别---------------------------------
    def preprocess(self, img, imgsz=640, stride=32, rect=True):
        """
        图片预处理：缩放+归一化+转(1,3,640,640)
        parameters:
        img: 原始图片,cv2(BGR)
        return:
        padded_img: 填充黑边的图片
        scale: 缩放比例
        dw: 填充宽度
        dh: 填充高度
        """
        self.img_height, self.img_width = img.shape[:2]
        if isinstance(imgsz, int):
            target_h, target_w = imgsz, imgsz
        else:
            target_h, target_w = int(imgsz[0]), int(imgsz[1])

        # 对齐 Ultralytics letterbox 细节
        scale = min(target_h / self.img_height, target_w / self.img_width)
        new_w = int(round(self.img_width * scale))
        new_h = int(round(self.img_height * scale))

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        if rect:
            # Ultralytics 默认矩形推理：填充对齐到 stride 的最小矩形
            pad_w = pad_w % stride
            pad_h = pad_h % stride
        pad_w /= 2
        pad_h /= 2

        if (self.img_width, self.img_height) != (new_w, new_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))
        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        padded_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # BGR -> RGB, HWC -> CHW, 归一化
        blob = padded_img[:, :, ::-1].transpose(2, 0, 1)  # MARK 这一步是干嘛的
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, left, top
        
    def postprocess(self, outputs, scale, dw, dh, conf_threshold=0.5, iou_threshold=0.5):
        """ 
        后处理：解码(坐标平移回原图坐标系)+非极大值抑制+缩放+偏移
        parameters:
        outputs:使用填充后的padded_img，预测输出值
        scale/dw/dh: 缩放/填充参数
        outputs:
        results: boxes,scores,class_ids(N,6) (x1,y1,x2,y2,score,cls)
        """
        results = []
        boxes, scores, class_ids = [], [], []

        pred = np.asarray(outputs[0])
        if pred.ndim == 3:
            pred = pred[0]

        # 情况1: 模型已输出 Nx6 (x1, y1, x2, y2, score, cls)
        if pred.ndim == 2 and pred.shape[1] == 6:
            det_boxes = pred[:, :4]
            det_scores = pred[:, 4]
            det_classes = pred[:, 5].astype(np.int32)
        # 情况2: 原始检测头输出，常见为 (nc+4, N) 或 (N, nc+4)
        elif pred.ndim == 2:
            p = pred
            if p.shape[0] < p.shape[1]:
                p = p.T
            if p.shape[1] <= 4:
                return results

            box_xywh = p[:, :4]
            cls_scores = p[:, 4:]
            det_classes = np.argmax(cls_scores, axis=1).astype(np.int32)
            det_scores = cls_scores[np.arange(cls_scores.shape[0]), det_classes]

            # xywh -> xyxy
            x, y, w, h = box_xywh[:, 0], box_xywh[:, 1], box_xywh[:, 2], box_xywh[:, 3]
            det_boxes = np.stack([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], axis=1)
        else:
            return results

        keep = det_scores >= conf_threshold
        if not np.any(keep):
            return results

        det_boxes = det_boxes[keep]
        det_scores = det_scores[keep]
        det_classes = det_classes[keep]

        # 还原到原图坐标
        det_boxes[:, 0] = (det_boxes[:, 0] - dw) / scale
        det_boxes[:, 1] = (det_boxes[:, 1] - dh) / scale
        det_boxes[:, 2] = (det_boxes[:, 2] - dw) / scale
        det_boxes[:, 3] = (det_boxes[:, 3] - dh) / scale

        nms_boxes = []
        for x1, y1, x2, y2 in det_boxes:
            x = float(max(0.0, x1))
            y = float(max(0.0, y1))
            w = float(max(0.0, x2 - x1))
            h = float(max(0.0, y2 - y1))
            nms_boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(nms_boxes, det_scores.tolist(), conf_threshold, iou_threshold)
        if indices is None or len(indices) == 0:
            return results

        for i in indices:
            i = int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i)
            x1, y1, x2, y2 = det_boxes[i]
            x1 = int(np.clip(round(x1), 0, self.img_width - 1))
            y1 = int(np.clip(round(y1), 0, self.img_height - 1))
            x2 = int(np.clip(round(x2), 0, self.img_width - 1))
            y2 = int(np.clip(round(y2), 0, self.img_height - 1))
            results.append([x1, y1, x2, y2, float(det_scores[i]), int(det_classes[i])])
        return results
    
    def infer(
        self,
        image,
        model_path,
        conf_threshold=0.5,
        iou_threshold=0.5,
        imgsz=640,
        stride=32,
        rect=True,
    ):
        """推理的完整流程：预处理->推理->后处理"""
    
        session=ort.InferenceSession(model_path)
        input_name=session.get_inputs()[0].name

        blob,scale,dw,dh=self.preprocess(image, imgsz=imgsz, stride=stride, rect=rect)
        outputs=session.run(None, {input_name: blob})
        results=self.postprocess(
            outputs,
            scale,
            dw,
            dh,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        return results

    #---------------------------------分类---------------------------------
    def _resolve_dim(self,dim_value, fallback_value):
        """把模型输入维度解析为 int，动态维度时回退到 fallback。"""
        if isinstance(dim_value, int):
            return dim_value
        if isinstance(dim_value, np.integer):
            return int(dim_value)
        return fallback_value

    def _onnx_type_to_numpy(self,onnx_type: str):
        """将 onnx 输入类型映射为 numpy dtype。"""
        type_map = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(uint8)": np.uint8,
            "tensor(int8)": np.int8,
            "tensor(int16)": np.int16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
        }
        return type_map.get(onnx_type, np.float32)

    def _read_export_imgsz(self,session):
        """从 ONNX metadata 读取导出时 imgsz，例如 [640, 640]。"""
        try:
            meta = session.get_modelmeta().custom_metadata_map
            imgsz_text = meta.get("imgsz")
            if not imgsz_text:
                return None
            imgsz = json.loads(imgsz_text)
            if isinstance(imgsz, int):
                return imgsz, imgsz
            if isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2:
                return int(imgsz[0]), int(imgsz[1])
        except Exception:
            return None
        return None

    def _resize_and_center_crop(self,image, target_h, target_w):
        """
        对齐 Ultralytics 分类预处理：
        先按短边等比例缩放，再做中心裁剪到目标尺寸。
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Invalid image with zero size.")

        scale = max(target_h / h, target_w / w)
        resized_h = int(round(h * scale))
        resized_w = int(round(w * scale))
        resized = cv2.resize(image, (resized_w, resized_h))

        y1 = max((resized_h - target_h) // 2, 0)
        x1 = max((resized_w - target_w) // 2, 0)
        y2 = y1 + target_h
        x2 = x1 + target_w
        cropped = resized[y1:y2, x1:x2]

        # 极端情况下再兜底 resize，保证输入尺寸与模型完全一致
        if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
            cropped = cv2.resize(cropped, (target_w, target_h))
        return cropped

    def _to_probabilities(self, raw_output):
        """若输出已是概率则直接使用，否则按 logits 做 softmax。"""
        x = np.asarray(raw_output, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        row = x[0]
        if np.all(row >= 0.0) and np.all(row <= 1.0 + 1e-6):
            total = float(row.sum())
            if 0.99 <= total <= 1.01:
                return x.astype(np.float32, copy=False)
        shifted = row - row.max()
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum()
        return probs.reshape(1, -1).astype(np.float32)

    def _topk_from_probs(self, probs, topk=5, class_names=None):
        """从概率向量提取 topk: (class_id, prob, name)。"""
        probs = np.asarray(probs)
        if probs.ndim == 2:
            probs = probs[0]
        k = min(topk, probs.shape[0])
        indices = np.argsort(probs)[::-1][:k]
        result = []
        for idx in indices:
            cid = int(idx)
            prob = float(probs[idx])
            if class_names and cid < len(class_names):
                name = class_names[cid]
            else:
                name = str(cid)
            result.append((cid, prob, name))
        return result

    def _letterbox_classify_preprocess(self, image, target_h, target_w, model_input_dtype):
        """对齐 main.py 训练链路: letterbox + /255 + ImageNet normalize。"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        letterbox = ClassifyLetterBox(size=(target_h, target_w))
        letterboxed = letterbox(rgb_image)
        blob = letterboxed.transpose(2, 0, 1).astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        blob = (blob - mean) / std
        return np.expand_dims(blob.astype(model_input_dtype, copy=False), axis=0)

    def _classify_ultralytics(self, model_path, image_path, imgsz=224):
        """Ultralytics letterbox 分类（支持 .onnx / .pt），返回 class_id / confidence / probabilities。"""
        model = self._get_ultra_model(model_path)
        pred = model.predict(
            source=image_path,
            predictor=LetterBoxClassificationPredictor,
            imgsz=imgsz,
            verbose=False,
        )
        ultra_result = pred[0]
        class_id = int(ultra_result.probs.top1)
        confidence = float(ultra_result.probs.top1conf)
        probabilities = ultra_result.probs.data.cpu().numpy()
        return {
            "class_id": class_id,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def compare_with_ultralytics(
        self,
        model_path,
        image_path,
        class_names=None,
        topk=5,
        imgsz=224,
        onnx_result=None,
        ultra_result=None,
    ):
        """同图对比 ONNXRuntime 与 Ultralytics(letterbox) 分类结果，默认使用同一模型。"""
        print(f"[compare] image={image_path}, model={model_path}, imgsz={imgsz}")

        if onnx_result is None:
            onnx_class_id, onnx_conf, onnx_details = self.classify(
                model_path,
                image_path,
                imgsz=imgsz,
                return_confidence=True,
                return_details=True,
                debug=False,
            )
        else:
            onnx_class_id = onnx_result["class_id"]
            onnx_conf = onnx_result["confidence"]
            onnx_details = onnx_result["details"]

        onnx_topk = self._topk_from_probs(onnx_details["probabilities"], topk=topk, class_names=class_names)

        if ultra_result is None:
            ultra = self._classify_ultralytics(model_path, image_path, imgsz=imgsz)
        else:
            ultra = ultra_result
        ultra_class_id = ultra["class_id"]
        ultra_conf = ultra["confidence"]
        ultra_topk = self._topk_from_probs(ultra["probabilities"], topk=topk, class_names=class_names)

        onnx_name = class_names[onnx_class_id] if class_names and onnx_class_id < len(class_names) else str(onnx_class_id)
        ultra_name = class_names[ultra_class_id] if class_names and ultra_class_id < len(class_names) else str(ultra_class_id)
        class_match = onnx_class_id == ultra_class_id
        prob_abs_diff = abs(onnx_conf - ultra_conf)

        print(f"ONNX top1 -> class={onnx_class_id}({onnx_name}), prob={onnx_conf:.6f}")
        print(f"ULTRA top1 -> class={ultra_class_id}({ultra_name}), prob={ultra_conf:.6f}")
        print(f"top1差值: class_match={class_match}, prob_abs_diff={prob_abs_diff:.6f}")

        print("ONNX topk:")
        for rank, (cid, prob, name) in enumerate(onnx_topk, start=1):
            print(f"  top{rank}: class={cid}({name}), prob={prob:.6f}")
        print("ULTRA topk:")
        for rank, (cid, prob, name) in enumerate(ultra_topk, start=1):
            print(f"  top{rank}: class={cid}({name}), prob={prob:.6f}")

        return {
            "onnx": {"class_id": onnx_class_id, "confidence": onnx_conf, "topk": onnx_topk},
            "ultra": {"class_id": ultra_class_id, "confidence": ultra_conf, "topk": ultra_topk},
            "class_match": class_match,
            "prob_abs_diff": prob_abs_diff,
        }
        

    #---------------------------------主函数入口---------------------------------#
    def classify(
        self,
        model_path,
        image_path,
        imgsz=None,
        return_confidence=False,
        return_details=False,
        debug=False,
    ):
        """
        分类
        parameters:
        model_path: 模型路径
        image_path: 图片路径
        return:
        classification: 分类结果
        """
        model=ort.InferenceSession(model_path)
        input_info=model.get_inputs()[0]
        input_name=input_info.name
        input_shape=input_info.shape
        model_input_dtype = self._onnx_type_to_numpy(input_info.type)
        image=cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # 仅处理常见的 NCHW/NHWC 四维输入
        #if len(input_shape) != 4:
        #    raise ValueError(f"Unsupported input shape: {input_shape}")

        image_h, image_w = image.shape[:2]
        export_imgsz = self._read_export_imgsz(model)
        target_h, target_w = self._resolve_classify_imgsz(
            imgsz=imgsz,
            export_imgsz=export_imgsz,
            default=224,
        )

        if input_shape[1] in (1, 3):  # NCHW
            use_ultralytics_transform = (
                input_shape[1] == 3 and np.issubdtype(model_input_dtype, np.floating)
            )
            if use_ultralytics_transform:
                try:
                    input_data = self._letterbox_classify_preprocess(
                        image, target_h, target_w, model_input_dtype
                    )
                except Exception as exc:
                    print(f"警告: letterbox 预处理失败，回退 OpenCV 路径。原因: {exc}")
                    input_data = self._resize_and_center_crop(image, target_h, target_w)
                    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
                    input_data = np.transpose(input_data, (2, 0, 1))
                    input_data = np.expand_dims(input_data, axis=0)
                    input_data = input_data.astype(np.float32) / 255.0
            else:
                input_data = self._resize_and_center_crop(image, target_h, target_w)
                if input_shape[1] == 3:
                    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
                input_data = np.transpose(input_data, (2, 0, 1))
                input_data = np.expand_dims(input_data, axis=0)
                if np.issubdtype(model_input_dtype, np.floating):
                    input_data = input_data.astype(np.float32) / 255.0
                else:
                    input_data = input_data.astype(model_input_dtype)
        else:  # NHWC
            input_data = self._resize_and_center_crop(image, target_h, target_w)
            if isinstance(input_shape[3], int) and input_shape[3] == 3:
                input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(input_data, axis=0)
            if np.issubdtype(model_input_dtype, np.floating):
                input_data = input_data.astype(np.float32) / 255.0
            else:
                input_data = input_data.astype(model_input_dtype)
        output = model.run(None, {input_name: input_data})
        raw_output = np.asarray(output[0])
        if raw_output.ndim == 1:
            raw_output = np.expand_dims(raw_output, axis=0)

        probs = self._to_probabilities(raw_output)
        class_id = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0][class_id])
        details = None
        if return_details:
            k = min(5, probs.shape[1])
            topk_indices = np.argsort(probs[0])[::-1][:k]
            topk = [
                (int(i), float(probs[0][i]), float(raw_output[0][i]))
                for i in topk_indices
            ]
            details = {
                "raw_output": raw_output[0].astype(np.float32, copy=False),
                "probabilities": probs[0].astype(np.float32, copy=False),
                "topk": topk,  # (class_id, prob, raw)
            }

        if debug:
            print(f"image: {image_path}")
            print(f"input_shape: {input_shape}, input_dtype: {input_info.type}, imgsz: ({target_h}, {target_w})")
            print(f"top1 -> class={class_id}, prob={confidence:.6f}, raw={float(raw_output[0][class_id]):.6f}")
            if details is not None:
                print("top5(prob/raw):")
                for rank, (cid, prob, raw) in enumerate(details["topk"], start=1):
                    print(f"  top{rank}: class={cid}, prob={prob:.6f}, raw={raw:.6f}")

        if return_confidence and return_details:
            return class_id, confidence, details
        if return_confidence:
            return class_id, confidence
        if return_details:
            return class_id, details
        return class_id

    def vial_detection_onnx2(
        self,
        image_path,
        vial_model_path,
        backend="ultralytics",
        conf=0.5,
        iou=0.5,
        imgsz=640,
    )->tuple[np.ndarray,np.ndarray]:
        """
        用于v0.4.0
        parameters:
        image_path: path of image(str)
        vial_model_path: path of vial model(*.onnx)
        return:
        coord_vial: (x_center,y_center,width,height)
        img_vial:crop img of vial
        """
        # 清空数据
        image=None
        model=None
        img_vial=None
        coord_vial=[-1,-1,-1,-1]
        
        image=cv2.imread(image_path)
        if image is None:   # 判断图片是否读取成功5
            raise errors.ImageReadError(f"Failed to read image from {image_path}")

        print(
            "[detection] "
            f"backend={backend}, image_path={image_path}, model_path={vial_model_path}, "
            f"conf={conf}, iou={iou}, imgsz={imgsz}"
        )
        
        if backend == "ultralytics":
            model_yolo=YOLO(vial_model_path, task="detect")
            pred = model_yolo.predict(
                source=image_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
            )
            results = []
            if pred and pred[0].boxes is not None and pred[0].boxes.xyxy is not None:
                boxes_xyxy = pred[0].boxes.xyxy.cpu().numpy()
                boxes_conf = pred[0].boxes.conf.cpu().numpy()
                boxes_cls = pred[0].boxes.cls.cpu().numpy()
                for b, s, c in zip(boxes_xyxy, boxes_conf, boxes_cls):
                    x1, y1, x2, y2 = b.tolist()
                    results.append([x1, y1, x2, y2, float(s), int(c)])
        elif backend == "onnxruntime":
            # 纯 ONNXRuntime 路径，不调用 Ultralytics
            results = self.infer(
                image,
                vial_model_path,
                conf_threshold=conf,
                iou_threshold=iou,
                imgsz=imgsz,
                stride=32,
                rect=True,
            )
        else:
            raise errors.InvalidConfigError(f"Invalid backend: {backend}")

        print(f"检测结果数量: {len(results)}")
        for idx, det in enumerate(results):
            x1, y1, x2, y2, score, cls_id = det
            print(
                f"[{idx}] bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}), "
                f"score={float(score):.2f}, class={int(cls_id)}"
            )

        if len(results) == 0: # 判断识别瓶子是否成功
            print(f"❗Did not detect vial")
            return None,None

        max_conf_idx = max(range(len(results)), key=lambda i: float(results[i][4]))
        best = results[max_conf_idx]
        print(f"[detection] best_score_raw={float(best[4]):.6f}, best_class={int(best[5])}")
        coord_vial_xyxy=[int(x) for x in best[0:4]]
        x1, y1, x2, y2 = coord_vial_xyxy # 坐标对应转换[x1,y1,x2,y2] -> [x_center, y_center, width, height]        
        x = int(max(0,(x1+x2)//2))
        y = int(max(0,(y1+y2)//2))
        w = int(round(max(0, x2 - x1)))   
        h = int(round(max(0, y2 - y1)))
        coord_vial=[x,y,w,h]
        img_vial=image[coord_vial_xyxy[1]-int(round(h*0.2)):coord_vial_xyxy[3]+int(round(h*0.2)),coord_vial_xyxy[0]:coord_vial_xyxy[2]]
        #img_vial=image[coord_vial_xyxy[1]:coord_vial_xyxy[3],coord_vial_xyxy[0]:coord_vial_xyxy[2]]  #MARK v3->v4修改点
        print(f"中心坐标：({coord_vial[0]},{coord_vial[1]}),宽:{coord_vial[2]},高:{coord_vial[3]}") 
        return img_vial,coord_vial

    def classify_main(self):
        #image_folder="D:/dissolution/vials/vials7/bottle_crop400pixles/blue/"
        image_folder="D:/dissolution/AlgoExperiment/Aug17_400pixelsfrommodulev2/all"
        model_path="./cfg/vials1n2n3n7_imgsz224_smodel_HRA1.onnx"
        imgsz=224
        class_names=["air","reagent"]
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        # 数值初始化
        Sample1_SN= None
        num_air=0
        num_reagent=0
        num_unclear=0

        classify_flag=True
        ultralytics_flag=True

        csv_rows = []
        csv_fieldnames = [
            "filename",
            "onnx_class",
            "onnx_class_name",
            "onnx_prob",
            "ultra_class",
            "ultra_class_name",
            "ultra_prob",
            "prob_diff",
            "image_path",
        ]

        for filename in os.listdir(image_folder):
            image_path=os.path.join(image_folder, filename)
            if not os.path.isfile(image_path):
                continue
            if Path(filename).suffix.lower() not in image_extensions:
                continue

            row = {
                "filename": filename,
                "onnx_class": "",
                "onnx_class_name": "",
                "onnx_prob": "",
                "ultra_class": "",
                "ultra_class_name": "",
                "ultra_prob": "",
                "prob_diff": "",
                "image_path": image_path,
            }
            onnx_result = None

            if classify_flag:
                class_idx, confidence, details = self.classify(
                    model_path,
                    image_path,
                    imgsz=imgsz,
                    return_confidence=True,
                    return_details=True,
                    debug=True,
                )
                row["onnx_class"] = class_idx
                row["onnx_class_name"] = (
                    class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
                )
                row["onnx_prob"] = f"{confidence:.4f}"
                onnx_result = {
                    "class_id": class_idx,
                    "confidence": confidence,
                    "details": details,
                }
                print(f"类别序号: {class_idx},类别名称:{class_names[class_idx]},置信度:{confidence:.6f}")
                print("top5细节(类别,概率,raw):")
                for cid, prob, raw in details["topk"]:
                    name = class_names[cid] if cid < len(class_names) else str(cid)
                    print(f"  ({cid}:{name}, {prob:.6f}, {raw:.6f})")
                if class_idx==0:
                    num_air+=1
                elif class_idx==1:
                    num_reagent+=1
                elif class_idx==2:
                    num_unclear+=1
                print(f"air: {num_air}, reagent: {num_reagent}, unclear: {num_unclear}")

            if ultralytics_flag:
                ultra = self._classify_ultralytics(model_path, image_path, imgsz=imgsz)
                ultra_class_id = ultra["class_id"]
                ultra_conf = ultra["confidence"]
                row["ultra_class"] = ultra_class_id
                row["ultra_class_name"] = (
                    class_names[ultra_class_id]
                    if ultra_class_id < len(class_names)
                    else str(ultra_class_id)
                )
                row["ultra_prob"] = f"{ultra_conf:.4f}"

                if classify_flag:
                    row["prob_diff"] = f"{abs(onnx_result['confidence'] - ultra_conf):.4f}"
                    self.compare_with_ultralytics(
                        model_path=model_path,
                        image_path=image_path,
                        class_names=class_names,
                        topk=5,
                        imgsz=imgsz,
                        onnx_result=onnx_result,
                        ultra_result=ultra,
                    )
                else:
                    print(
                        f"ULTRA top1 -> class={ultra_class_id}({row['ultra_class_name']}), "
                        f"prob={ultra_conf:.6f}"
                    )
                print("-" * 80)

            csv_rows.append(row)

        csv_path = PROJECT_ROOT / f"classify_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV已保存: {csv_path}")

        return Sample1_SN,num_air,num_reagent,num_unclear

    def detection_main(self):
        image_path="D:/dissolution/AlgoExperiment/Aug14_FailureCase/2_channel/July29_a1_b3_a2_b3_back.bmpD:/dissolution/AlgoExperiment/Aug14_FailureCase/2_channel/July29_a1_b3_a2_b3_back.bmp"
        model_path="./cfg/vial_HRA1_v1simv2_v2_July31.onnx"
        vial_detection_result=self.vial_detection_onnx2(
            image_path=image_path,
            vial_model_path=model_path,
            #backend="ultralytics",
            backend="onnxruntime",
            conf=0.5,
            iou=0.5,
            imgsz=640,
        )
        return

    def run(self,task_name):
        getattr(self,task_name,lambda:print("未找到任务"))()    


if __name__ == "__main__":
    runner=TaskRunner()
    runner.run("classify_main")  # classify_main,detection_main
