"""
使用onnx推理，和使用ultralytics最接近的版本
"""

import onnxruntime as ort
import errors
from pathlib import Path
import json
from typing import Final
import os
import argparse
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

class TaskRunner:

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

    #---------------------------------主函数入口---------------------------------#
    def classify(
        self,
        model_path,
        image_path,
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
        fallback_h = export_imgsz[0] if export_imgsz else image_h # 优先从模型导出尺寸export_imgsz，没有就用图片尺寸
        fallback_w = export_imgsz[1] if export_imgsz else image_w

        if input_shape[1] in (1, 3):  # NCHW
            target_h = self._resolve_dim(input_shape[2], fallback_h)
            target_w = self._resolve_dim(input_shape[3], fallback_w)

            use_ultralytics_transform = (
                input_shape[1] == 3 and np.issubdtype(model_input_dtype, np.floating)
            )
            if use_ultralytics_transform:
                try:
                    from ultralytics.data.augment import classify_transforms

                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    transform_size = (target_h, target_w) if target_h != target_w else target_h
                    transform = classify_transforms(size=transform_size)
                    transformed = transform(pil_image)
                    input_data = np.expand_dims(
                        transformed.numpy().astype(model_input_dtype, copy=False), axis=0
                    )
                except Exception as exc:
                    print(f"警告: 使用 ultralytics 官方预处理失败，回退 OpenCV 路径。原因: {exc}")
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
            target_h = self._resolve_dim(input_shape[1], fallback_h)
            target_w = self._resolve_dim(input_shape[2], fallback_w)
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
            print(f"input_shape: {input_shape}, input_dtype: {input_info.type}")
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
        image_folder="D:/dissolution/vials/vials1n2nsuspension/bottle_cap_crops/images/train/unclear"    
        model_path="./cfg/Apr29_3pm_bottlecrop.onnx"
        class_names=["air","reagent","unclear"]

        # 数值初始化
        Sample1_SN= None
        num_air=0
        num_reagent=0
        num_unclear=0

        for image_path in os.listdir(image_folder):
            image_path=os.path.join(image_folder,image_path)
            class_idx, confidence, details = self.classify(
                model_path,
                image_path,
                return_confidence=True,
                return_details=True,
                debug=True,
            )
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
            self.compare_with_ultralytics(
                model_path=model_path,
                image_path=image_path,
                class_names=class_names,
                topk=5,
            )
            print("-" * 80)

        return Sample1_SN,num_air,num_reagent,num_unclear

    def detection_main(self):
        image_path="D:/dissolution/hardcase/May26_LowScore_Interface/Apr16_19_2818-69-1_64-17-5_a4_back_img_vial.bmp"
        model_path="./cfg/best_interfacev2_2_Apr30.onnx"
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
    runner.run("detection_main")  # classify_main,detection_main
