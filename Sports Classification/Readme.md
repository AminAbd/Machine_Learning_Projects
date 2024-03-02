## Project: 100 Sports Image Classification

### Objective
The primary objective of this project is to classify images into 100 different sports categories accurately, leveraging the inherent characteristics and context present within each image.
### Dataset:
The dataset used for this project is sourced from the following Kaggle dataset:
https://www.kaggle.com/datasets/gpiosenka/sports-classification
### Inputs
Each image in the dataset, with dimensions of 224x224x3, serves as an input to the classification model. These images cover a broad range of sports, providing diverse visual data for the model to learn from.
### Data Augmentation:
1- Random Horizontal Flipping
2- Random Zooming (up to 20%)
### Models Used
#### EfficientNetB0:
EfficientNet represents a groundbreaking approach in the design of convolutional neural network (CNN) architectures, achieving superior accuracy with minimal computational resources. Developed by Mingxing Tan and Quoc V. Le, EfficientNet's core lies in its innovative compound scaling method, which scales the network dimensions—depth, width, and resolution—harmoniously.
**Key Components:**
Baseline EfficientNet-B0: The foundation model, EfficientNet-B0, integrates MBConv blocks, known for their efficiency and effectiveness, combining depthwise separable convolutions with squeeze-and-excitation blocks for enhanced feature representation.

Compound Scaling: This unique scaling approach applies a compound coefficient (ϕ) to scale depth, width, and resolution uniformly, facilitating the creation of more substantial and more accurate networks without excessive computational load.

EfficientNet Variants: The EfficientNet family ranges from B1 to B7 models, offering a spectrum of capabilities, each scaled optimally to balance accuracy and efficiency.

Technological Innovations: EfficientNets leverage MBConv blocks, squeeze-and-excitation optimization, and the Swish activation function, contributing to their remarkable efficiency and performance.

EfficientNets have gained widespread adoption across various domains, particularly where computational efficiency is paramount, such as mobile and edge computing. Their scalability and performance have set new benchmarks in the field of deep learning.

---
Training Overview:

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy


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

#### Resnet50:

**Key aspects of ResNet50 include:**

**Residual Blocks:** These core components help maintain information flow across layers, supporting the training of deep networks.

**Efficiency:** Despite its depth, ResNet50 is computationally efficient due to its bottleneck design, reducing dimensionality within its residual blocks.

**Versatility:** As part of the broader ResNet family, which includes various models like ResNet101 and ResNet152, ResNet50 provides a balanced option, widely applicable across numerous tasks and industries.

**Impact:** ResNet50 has significantly influenced deep learning, offering a robust framework for various applications, from image processing to advanced recognition tasks.

ResNet50 remains a cornerstone in deep learning architectures, celebrated for its depth and efficiency, proving crucial in advancing the field and inspiring future network designs.



Training Overview:
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy

| Epoch | Loss   | Accuracy | Val Loss | Val Accuracy |
|-------|--------|----------|----------|--------------|
| 1     | 1.4631 | 64.66%   | 0.6411   | 80.00%       |
| 2     | 0.5680 | 85.01%   | 0.4902   | 85.00%       |
| 3     | 0.3881 | 89.29%   | 0.3741   | 88.00%       |
| 4     | 0.2923 | 91.96%   | 0.3831   | 88.20%       |
| 5     | 0.2406 | 92.83%   | 0.4852   | 84.60%       |
| 6     | 0.2073 | 93.96%   | 0.3792   | 88.20%       |
| 7     | 0.1717 | 95.16%   | 0.3804   | 89.00%       |
| 8     | 0.1501 | 95.50%   | 0.4334   | 88.00%       |
| 9     | 0.1474 | 95.61%   | 0.4240   | 87.80%       |
| 10    | 0.1375 | 95.88%   | 0.3479   | 89.20%       |


