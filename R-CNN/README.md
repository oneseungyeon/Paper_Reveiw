# Review for R-CNN (Regions with CNN features)
## Object detection
![image](https://user-images.githubusercontent.com/74392995/128475565-00990815-7440-4eb8-ae69-f439f1b52a80.png)

## Abstract
1. One can apply high-capacity **CNN** to **bottom-up region proposals** in order **to localize and segment objects**
2. When labeled training **data is scarce**(데이터의 양이 부족할때),**supervised pre-training** for an **auxiliary task**, followed by **domain-specific fine-tuning**, yields a significant performance boost.
## Introduction
### Object detection system overview
![image](https://user-images.githubusercontent.com/74392995/128470816-f7e324e1-2035-4f4e-a024-d4b80373e655.png)

1.  Takes an input image
2.  Extracts around 2000 bottom-up region proposals -> 이때 모든 region proposal의 크기를 맞춰주는데, 그 이유는 output의 이미지를 같게 해야하기 때문이다. => warp(=resize) 실행
3.  Computes features for each proposal using a large convolutional neural network (CNN)
4.  Classifies each region using class-specific linear SVMs
### Two problems: localizing objects with a deep network and training a high-capacity model with only a small quantity of annotated detection data.
* Localizing objects with a deep network
1. localization as a regression problem -> But it have very large receptive fields (195 × 195 pixels) and strides (32×32 pixels) in the input image
2. **Recognition using regions** : extracts a fixed-length feature vector from each proposal using a CNN, and then classifies each region with category-specific linear SVMs, simple technique (affine image warping) to compute a fixed-size CNN input from each region proposal, regardless of the region’s shape.
* Training
4. **Fine-tuning** : use unsupervised pre-training, followed by supervised fine-tuning on a large auxiliary dataset (ILSVRC), followed by domainspecific fine-tuning on a small dataset (PASCAL), is an effective paradigm for learning high-capacity CNNs when data is scarce
## Object detection with R-CNN
1. The first generates category-independent **region proposals**.
2. The second module is a large **convolutional neural network** that extracts a fixed-length feature vector from each region.
3. The third module is a set of **classspecific linear SVMs**.
### Region Proposals -> Selective Search
1. 비슷한 색, 질감 등을 갖는 인접 픽셀들의 다양한 크기로 window를 생성 -> Generate initial sub-segmentation
2. Window들을 Bottom-up방식으로 합쳐 더 큰 window 생성 -> Recursively combine similar regions into larger ones
3. (2)를 반복하여 최종적으로 2000개의 region proposal을 생성 -> Use the generated regions to produce candidate object locations
![image](https://user-images.githubusercontent.com/74392995/128473651-c46cdb1d-18a5-4daf-8e96-6304020069a0.png)


### Feature extraction
1. We extract a 4096-dimensional feature vector from each region proposal
2. Features are computed by forward propagating a mean-subtracted 227 × 227 RGB image through five convolutional layers and two fully connected layers.
3. In order to compute features for a region proposal, we must first convert the image data in that region into a form that is compatible with the CNN (its architecture requires
inputs of a fixed 227 × 227 pixel size
4. Prior to warping, we dilate the tight bounding box so that at the warped size there are exactly p pixels of warped image context around the original box (we use p = 16) -> object와 인접한 16개의 픽셀도 포함시킴
5. fine-tuning을 적용
![image](https://user-images.githubusercontent.com/74392995/128475282-26a007b0-b3db-48e0-8ab8-6209d2db28b8.png)

## Test
* Process
1. We run selective search on the test image to extract around 2000 region proposals.
2. We warp each proposal and forward propagate it through the CNN in order to compute features. 
3. Then, for each class, we score each extracted feature vector using the SVM trained for that class. 
4. Given all scored regions in an image, we apply a greedy non-maximum suppression -> 박스 중 가장 스코어(IoU)가 높은 박스만 남기고 나머지 박스는 제거
* Run-time analysis : all CNN parameters are shared across all categories. + feature vectors computed by the CNN are low-dimensional when compared to other common approaches ->  efficient
## Training 
1. Superviese pre-training : we discriminatively pre-trained the CNN on a large auxiliary dataset (ILSVRC2012 classification) using image-level annotations only
2. Domain-specific fine-tuning : we continue stochastic gradient descent (SGD) training of the CNN parameters using only warped region proposals
3. Object category classifiers : selecting this threshold carefully is important. Positive examples are defined simply to be the ground-truth bounding boxes for each class 
### SVM classifier
1. The feature matrix is typically 2000×4096 and the SVM weight matrix is 4096×N, where N is the number of classes. -> CNN으로 추출한 4096 차원의 특징 벡터를 SVM으로 분류
### Bounding Box Regression
1. SVM으로 분류된 bounding box 중 ground truth와 가장 IoU가 높은 bounding box를 ground-truth box와 비슷하게 조정 ->  region proposal P 와 정답 위치 G가 존재할 때, P를 G로 mapping할 수 있는 변환을 학습

![image](https://user-images.githubusercontent.com/74392995/128666416-42b5fd2d-d078-4cfd-8bf8-722bb810c0d0.png)

![image](https://user-images.githubusercontent.com/74392995/128666397-3194d0cc-9737-46d6-a069-23212b38ebed.png)

2. bounding box regression을 한 실험이 더 성능이 좋았음
![image](https://user-images.githubusercontent.com/74392995/128484335-c3dce5dd-e577-4043-80e3-f630b8042444.png)

### Appendix
* Object proposal transformations

![image](https://user-images.githubusercontent.com/74392995/128666235-29bc3ebf-56ee-41f0-9e08-4bcac1bec31b.png)

* 2000개의 bounding box를 convolution 연산을 하기 때문에 시간과 비용이 많이 소요된다 => Fast R-CNN, Faster R-CNN 
