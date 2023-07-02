# Artificial-Intelligence-Externship
SmartBridge externship in Artificial Intelligence


## Assignments

### Assignment 1

#### Problem statement
Exploring basic python libraries used for ML/DL - like numpy, pandas, etc.

### Assignment 2

#### Problem statement
Classification of drugs using CNN model

#### Dataset
Link for the kaggle dataset: https://www.kaggle.com/datasets/prathamtripathi/drug-classification/download?datasetVersionNumber=1

#### Model used
Convolutional Neural Networks (CNNs) are a class of deep learning models widely used for image and video analysis tasks. CNNs are designed to automatically learn and extract hierarchical patterns and features from input data through a series of convolutional layers, pooling layers, and fully connected layers. The convolutional layers use filters to scan the input data, capturing local patterns and spatial relationships. The pooling layers downsample the feature maps, reducing their spatial dimensions while retaining important information. The fully connected layers integrate the high-level features and make predictions based on the learned representations.

### Assignment 3

#### Problem statement
Bird species classification using CNN model and Transfer learning.

#### Dataset
Link for the kaggle dataset: https://www.kaggle.com/datasets/akash2907/bird-species-classification/download?datasetVersionNumber=1

#### Model(s) used
The models used for this task are CNN and Transfer learning model - ResNet50.

Transfer learning is a technique in deep learning where a pre-trained model, such as ResNet50, is used as a starting point for a new task or dataset. ResNet50 is a popular pre-trained convolutional neural network architecture that consists of 50 layers. It was originally trained on a large dataset (e.g., ImageNet) and has learned to recognize a wide range of visual features. In transfer learning, the pre-trained ResNet50 model's lower layers are retained, acting as a feature extractor, while the final layers are replaced or fine-tuned to adapt to the new task. This approach leverages the knowledge and representations learned by the model on a large dataset and applies it to a different but related task, even with limited data. By using transfer learning with ResNet50, it becomes possible to achieve good performance and reduce the training time for new tasks, especially in scenarios where limited labeled data is available.


## Project: People Counting and Tracking System

### Problem Statement 
People counting and tracking are crucial tasks in various domains, including retail, transportation, and security. Traditional methods of manual counting and tracking are time-consuming, error-prone, and limited in their ability to handle large crowds. 
To overcome these challenges, this project aims to develop an AI-based system for automated people counting and tracking. By leveraging advanced computer vision techniques and deep learning algorithms, the system will accurately detect and track individuals in real-time, 
allowing for efficient and reliable analysis of crowd behavior and movement patterns.

The primary objective of this project is to create a robust and scalable solution that can count and track people in various scenarios, such as shopping malls, airports, and public events. The system will be trained to handle occlusions, changes in lighting conditions, 
and complex scenes, ensuring accurate and reliable people detection and tracking. By achieving high precision and real-time performance, the project aims to provide valuable insights for crowd management, resource allocation, and security enhancement, thereby improving 
overall operational efficiency and safety in crowded environments.

#### Model(s) Used

The model used in this task is MobilenetSSD.

detection tasks on resource-constrained devices such as mobile phones or embedded systems. It combines MobileNet, a low-complexity convolutional neural network (CNN), with a Single Shot MultiBox Detector (SSD) framework.

MobileNet is specifically designed to be computationally efficient and have a small memory footprint, making it suitable for real-time applications on devices with limited processing power. It achieves this by employing depthwise separable convolutions, 
which factorize standard convolutions into a depthwise convolution followed by a pointwise convolution. This significantly reduces the number of parameters and computations required, while still capturing meaningful features.

The SSD framework, on the other hand, is a popular object detection approach that utilizes a set of predefined anchor boxes at different scales and aspect ratios to detect objects. It performs convolutional feature extraction at multiple scales and 
predicts the presence of objects within each anchor box, along with their corresponding class labels and bounding box coordinates.

By combining MobileNet and SSD, MobileNetSSD achieves a good balance between accuracy and efficiency. It allows real-time object detection on mobile and embedded devices, enabling applications such as real-time video analysis, surveillance systems, and augmented reality. 
