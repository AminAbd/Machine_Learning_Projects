## Project: 100 Sports Image Classification

### Objective
The primary objective of this project is to classify images into 100 different sports categories accurately, leveraging the inherent characteristics and context present within each image.
### Dataset:
The dataset used for this project is sourced from the following Kaggle dataset:
https://www.kaggle.com/datasets/gpiosenka/sports-classification
### Inputs
Each image in the dataset, with dimensions of 224x224x3, serves as an input to the classification model. These images cover a broad range of sports, providing diverse visual data for the model to learn from.

### Models Used
**EfficientNetB0:
Model Training and Implementation
The model employed for this task is EfficientNetB0, known for its efficiency and effectiveness in handling image classification tasks. The training process is enhanced with data augmentation techniques like random flipping and zooming to improve the model's generalization capabilities.

Data Augmentation:

Random Horizontal Flipping
Random Zooming (up to 20%)
Model Architecture:

Base Model: EfficientNetB0 pre-trained on ImageNet, with the top layer removed.
Global Average Pooling 2D layer.
Dense layer with 256 units and ReLU activation.
Batch Normalization layer.
Dropout layer with a rate of 0.1.
Output Dense layer with 100 units (one per sport category) with softmax activation.
Training Overview:

Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Training and Validation Performance:

Training and Validation Performance:
 
| Epoch | Loss   | Accuracy | Val Loss | Val Accuracy |
|-------|--------|----------|----------|--------------|
| 1     | 1.0074 | 75.47%   | 0.2390   | 93.40%       |
| 2     | 0.3166 | 91.11%   | 0.2046   | 92.60%       |
| 3     | 0.2045 | 94.34%   | 0.1837   | 94.20%       |
| 4     | 0.1428 | 95.76%   | 0.1799   | 94.20%       |
| 5     | 0.1190 | 96.51%   | 0.1920   | 94.00%       |
| 6     | 0.1046 | 96.86%   | 0.1947   | 94.40%       |
| 7     | 0.0840 | 97.44%   | 0.1503   | 95.20%       |
| 8     | 0.0803 | 97.51%   | 0.1881   | 95.00%       |
| 9     | 0.0794 | 97.56%   | 0.1924   | 93.60%       |
| 10    | 0.0771 | 97.81%   | 0.1986   | 93.60%       |
