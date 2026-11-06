深度学习目标检测中使用YOLOv8对草莓病害目标检测数据集进行目标检测任务 _对草莓病害数据集进行构建和训练模型。
@[toc]

以下文字及代码仅供参考。
草莓病害目标检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9f80e781ed44447ba16198514d62be22.png)

共9080张图像，，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c410d2e831f9475aa2aa52deca43fe80.png)
6类

['Angular Leafspot', 'Anthracnose Fruit Rot', 'Gray Mold', 'Leaf Spot', 'Powdery Mildew Fruit', 'Powdery Mildew Leaf']

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f463b9cdc49d400e8db4057fec2a3a45.png)

['角斑病', '炭疽果腐病', '灰霉病', '叶斑病', '果实白粉病', '叶片白粉病']
数据集划分:训练集、验证集、测试集【7925:577:578】
YOLO标注格式

使用深度学习模型对草莓病害进行目标检测，采用YOLOv8作为目标检测框架。YOLO格式的标注文件，并且有明确的划分（训练集、验证集和测试集），构建和训练模型。

### 1. 数据集准备

假设你的数据集结构如下：

```
strawberry_disease_detection/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img1.jpg
│   │   └── ...
│   └── test/
│       ├── img1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── ...
    ├── val/
    │   ├── img1.txt
    │   └── ...
    └── test/
        ├── img1.txt
        └── ...
data_strawberry_disease.yaml
```

`data_strawberry_disease.yaml` 文件内容示例：

```yaml
train: ./strawberry_disease_detection/images/train/
val: ./strawberry_disease_detection/images/val/

nc: 6  # 类别数量：Angular Leafspot, Anthracnose Fruit Rot, Gray Mold, Leaf Spot, Powdery Mildew Fruit, Powdery Mildew Leaf
names: ['Angular Leafspot', 'Anthracnose Fruit Rot', 'Gray Mold', 'Leaf Spot', 'Powdery Mildew Fruit', 'Powdery Mildew Leaf']
```

确保每个标签文件的内容符合YOLO格式，即每行包含一个对象的类别ID、中心点x, y坐标（归一化到0-1），宽度和高度（同样归一化）。

### 2. 安装依赖库

确保安装了必要的库：

```bash
pip install ultralytics opencv-python-headless tensorboard
```

### 3. 模型训练

创建一个Python脚本来开始训练过程。这里我们以YOLOv8为例说明如何训练模型。

#### 训练脚本

```python
from ultralytics import YOLO

def main_train():
    # 加载预训练的YOLOv8n模型或从头开始定义模型
    model = YOLO('yolov8n.yaml')  # 或者直接加载预训练权重，如 'yolov8n.pt'
    
    results = model.train(
        data='./data_strawberry_disease.yaml',
        epochs=100,  # 根据需要调整
        imgsz=640,
        batch=16,
        project='./runs/detect',
        name='strawberry_disease_detection',
        optimizer='SGD',
        device='0',  # 使用GPU编号
        save=True,
        cache=True,
        verbose=True,
    )

if __name__ == '__main__':
    main_train()
```

### 4. 推理与结果可视化

训练完成后，我们可以利用训练好的模型对新图片进行预测，并将结果可视化。

#### 推理脚本

```python
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO('./runs/detect/strawberry_disease_detection/weights/best.pt')

def detect_objects(image_path):
    results = model.predict(source=image_path)
    img = cv2.imread(image_path)
    for result in results:
        boxes = result.boxes.numpy()
        for box in boxes:
            r = box.xyxy
            x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            label_id = int(box.cls)
            label = result.names[label_id]
            confidence = box.conf
            if confidence > 0.5:  # 设置置信度阈值
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制矩形框
                cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

# 示例调用
result_image = detect_objects('your_test_image.jpg')  # 确保测试图像路径正确
Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).show()  # 使用PIL显示图像
```

### 5. 可视化界面

为了监控训练过程，可以使用TensorBoard。在训练脚本中添加 `tensorboard=True` 参数，然后运行以下命令启动TensorBoard：

```bash
tensorboard --logdir runs/
```

然后在浏览器中访问 `http://localhost:6006` 查看训练进度和结果。

### 6. 运行步骤总结

1. **数据集准备**：确认数据集已按要求组织好，并创建`data_strawberry_disease.yaml`。
2. **安装依赖**：通过提供的命令安装所需的Python库。
3. **训练模型**：运行训练脚本开始训练过程。
4. **推理与可视化**：使用训练好的模型进行推理，并可视化结果。
5. **监控训练**：使用TensorBoard监控训练过程。
使用YOLOv8对草莓病害目标检测数据集进行目标检测任务。请根据实际需求调整相关参数和代码。如果有任何问题或需要进一步的帮
