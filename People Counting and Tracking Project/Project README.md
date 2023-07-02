# Problem Statement 
People counting and tracking are crucial tasks in various domains, including retail, transportation, and security. Traditional methods of manual counting and tracking are time-consuming, error-prone, and limited in their ability to handle large crowds. 
To overcome these challenges, this project aims to develop an AI-based system for automated people counting and tracking. By leveraging advanced computer vision techniques and deep learning algorithms, the system will accurately detect and track individuals in real-time, 
allowing for efficient and reliable analysis of crowd behavior and movement patterns.

## Dataset

The primary objective of this project is to create a robust and scalable solution that can count and track people in various scenarios, such as shopping malls, airports, and public events. The system will be trained to handle occlusions, changes in lighting conditions, 
and complex scenes, ensuring accurate and reliable people detection and tracking. By achieving high precision and real-time performance, the project aims to provide valuable insights for crowd management, resource allocation, and security enhancement, thereby improving 
overall operational efficiency and safety in crowded environments.

## Model(s) Used

The model used in this task is MobilenetSSD.

detection tasks on resource-constrained devices such as mobile phones or embedded systems. It combines MobileNet, a low-complexity convolutional neural network (CNN), with a Single Shot MultiBox Detector (SSD) framework.

MobileNet is specifically designed to be computationally efficient and have a small memory footprint, making it suitable for real-time applications on devices with limited processing power. It achieves this by employing depthwise separable convolutions, 
which factorize standard convolutions into a depthwise convolution followed by a pointwise convolution. This significantly reduces the number of parameters and computations required, while still capturing meaningful features.

The SSD framework, on the other hand, is a popular object detection approach that utilizes a set of predefined anchor boxes at different scales and aspect ratios to detect objects. It performs convolutional feature extraction at multiple scales and 
predicts the presence of objects within each anchor box, along with their corresponding class labels and bounding box coordinates.

By combining MobileNet and SSD, MobileNetSSD achieves a good balance between accuracy and efficiency. It allows real-time object detection on mobile and embedded devices, enabling applications such as real-time video analysis, surveillance systems, and augmented reality. 
