# Diagnosis Of Bone Fractures From X-ray Images Using YoloV8n and Faster-RCNN
 
## Introduction
This project builds and compares two Computer Vision models (YOLO and Faster R-CNN) in the problem of bone fracture recognition on the [Bone Fracture Detection](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project/data) dataset from Kagge.

## This project has 2 model
### Comparison Between YOLOv8n and Faster R-CNN
In this project, I experimented with two prominent object detection models — YOLOv8n and Faster R-CNN — to detect bone fractures in X-ray images.

I began with YOLOv8n, a real-time object detection model known for its speed and simplicity. Since I had prior experience with YOLO and the dataset was already pre-processed, I expected it to perform reasonably well. However, the results were disappointing — the model struggled to detect fine-grained fractures and showed low accuracy across most test cases.

After discussing the issue with my instructor, we concluded that YOLOv8n might not be the best fit for medical imaging tasks that require high precision. This led me to explore Faster R-CNN, a region-based convolutional neural network designed for more accurate object detection.

Faster R-CNN integrates a Region Proposal Network (RPN) with a classification network, allowing it to focus on key areas within the image and make more accurate predictions. I fine-tuned a pre-trained Faster R-CNN model on the bone fracture dataset, and the performance improved significantly. The model was able to detect small, subtle cracks that YOLO often missed, proving to be much more effective for this medical imaging problem.

While Faster R-CNN required longer training times and more computational resources, it delivered much better accuracy and consistency in fracture detection, making it a more suitable choice for this project.
