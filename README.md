# Complex neural network to detect and semantic segmentation
There are many problems in modern ecology that could be solved with the use of
modern technologies. Among the important problems in our country in this area,
the need for constant monitoring of fish biodiversity in various rivers should be
singled out separately. 
Complete and timely information about fish migrations
allows scientists and relevant competent specialists to extract useful data and
predict the behavior of fish, as well as receive information about their condition in
real time from the corresponding biogeocenosis.


## Prerequisites
The implementation is GPU-based. You need CUDA for optimal and fast training. We recomended run model on colab or another GPU-based server.
```
git clone https://github.com/justfollowthesun/ML2022_Project
cd ML2022_Project
pip install -r requirements.txt 
```
After downloading the dataset from Roboflow (it will load inside the yolov5 folder in folder DigitalSea), you need to move the train, test, val folders to the same directory with the yolov5 folder (that is, "in parallel"), and the data file.yaml - move from dataset to yolov5 root folder

## Data labeling 
There are three types of classes in the prepared dataset.
The class of single fish is simply called a "fish", it is highlighted by the corresponding rectangular bounding box, while marking this particular class was associated with the greatest difficulties and time costs due to:
a) the multitude of single objects in the photo, which may be similar to the target class
b) different water transparency, photo quality (locally), different shadows
c) requirements for maintaining high marking accuracy for successful training
This is the task of detecting objects.
The other two classes relate to solving the problem of semantic segmentation of shoals of fish of different densities (sparse shoals - "scatter" and dense shoals - "shoal"), the data markup of these classes was made much faster, but the classes are very similar to each other and differ
a) in the density of fish
b) the presence of single fish in the stream
c) the presence of cyclic congestion and intersections
(d) implicit class boundaries

## Parameters
hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0

#custom model
# parameters
nc: {num_classes}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

## Results
In the course of the work, intermediate results were obtained for object detection using Yolov5, the results obtained on a small part of the dataset indicate that more data of target classes is needed (now one of the main difficulties - the disproportionality of the division of classes by photos, both in number and density of representatives of different classes), which, when cutting the original 8K photos into many small ones to a format suitable for Yolov5 training (in this case 640 pixels), creates a lot of "empty" photos that do not qualitatively increase accuracy, but are capable of making inappropriate noise (however, the study of the heterogeneity of objects in the photos showed that objects that are similar to the target objects of the class are not so common).
In total, 77 large-format images were used for Yolo, divided into 12x12 images, the number of class targets is estimated at 10-12 thousand.

## Built With
PyTorch, Linux, CVAT, CUDA, Colab

 
## Authors
- [**Ivan Anisimov**](https://github.com/justfollowthesun)
- [**Artemy Tsirkulenko**](https://github.com/Prometei6969)
- [**Natalia Shmigelskaya**](https://github.com/NataliaShmigelskaya)
