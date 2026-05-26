from ultralytics import YOLO
import os
"""
对整个文件夹内图片进行推理，输出分数
"""

class TaskRunner:
    def get_folder_list(self, folder_path):
        folder_list = []
        for file in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, file)):
                folder_list.append(os.path.join(folder_path, file))
        return folder_list

    def get_image_list(self, folder_path):
        image_list = []
        for file in os.listdir(folder_path):
            if file.endswith("back.bmp") :
                image_list.append(os.path.join(folder_path, file))
        return image_list


    def inference(self, parent_folder_path, **predict_kwargs):
        num_pic=0
        model = predict_kwargs.pop("model")
        #output_path = predict_kwargs.pop("output_path", os.path.join(parent_folder_path, "inference_results.txt"))
        output_path = "inference_results.txt"
        sub_folder_list = self.get_folder_list(parent_folder_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for sub_folder_path in sub_folder_list:
                image_list = self.get_image_list(sub_folder_path)
                f.write(f"folder_path: {sub_folder_path}/n")
                for image_path in image_list:
                    image_name = os.path.basename(image_path)
                    num_pic += 1
                    result = model.predict(image_path, **predict_kwargs, verbose=False)
                    if len(result[0].boxes) == 0:
                        f.write(f"{image_name} 未检测到目标/n")
                        print(f"{image_name} 未检测到目标")
                        continue
                    if len(result[0].boxes) == 1:
                        f.write(
                            f"class={int(result[0].boxes.cls)},score={float(result[0].boxes.conf):.2f}    {image_name}/n"
                        )
                        print(f"class={int(result[0].boxes.cls)},score={float(result[0].boxes.conf):.2f}    {image_name}")
                    else:
                        f.write(f"检测到多个目标: {image_name}/n")
                        for box in result[0].boxes:
                            f.write(f"class={int(box.cls)},score={float(box.conf):.2f}    {image_name}/n")
                            print(f"class={int(box.cls)},score={float(box.conf):.2f}    {image_name}")
        print(f"结果已保存: {os.path.abspath(output_path)},共{num_pic}张图片")
    

    def run(self, task_name, *args, **kwargs):
        getattr(self, task_name, lambda *a, **k: print("未找到任务"))(*args, **kwargs)

if __name__=="__main__":
    model = YOLO("D:/Practise/Tools/models/val2_Apr22.pt")
    #parent_folder_path = "D:/dissolution/myself/Apr16/13_564483-18-7_64-17-5"  
    #parent_folder_path = "D:/dissolution/myself/Apr17"  
    parent_folder_path="D:/dissolution/testing_engineer/Apr24"
    task_runner = TaskRunner()
    task_runner.run("inference",parent_folder_path=parent_folder_path,model=model)
