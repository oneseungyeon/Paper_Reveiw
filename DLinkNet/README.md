# D-LinkNet
### Review for D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction
## Abstract
D-LinkNet which adopts encoder-decoder structure, dilated convolution and pretrained encoder for road extraction task
  -  Dilation convolution is a powerful tool that can enlarge the receptive field of feature points without reducing the resolution of the feature maps : 전체적인 특징을 잡아내기 위해 필터 사이즈를 늘려야 하지만 이는 연산량이 크게 늘어나고 오버피팅의 우려가 있다. 이에 필터 내부에 zero padding을 추가해여 큰 범위를 보지만 부분적으로 정보를 수집한다.

![image](https://user-images.githubusercontent.com/74392995/126930831-061b0fe4-764d-48a9-9d04-af6bd5106bb1.png)

In the CVPR DeepGlobe 2018 Road Extraction Challenge, our best IoU scores on the validation set and the test set are 0.6466 and 0.6342 respectively
## Introduction
Boltzmann machines + classification network to assign each patch extracted + FCN architecture and employed a Unet with residual connections using DCNN
#### Road segmentation from high resolution satellite images
  1. The input images are of high-resolution, so networks for this task should have **large receptive field** that can cover the whole image. 
  2. Roads in satellite images are often slender, complex and cover a **small part** of the whole image.
  3. Roads have **natural connectivity and long span**.
### D-LinkNet Architeture
![image](https://user-images.githubusercontent.com/74392995/126932769-d770f368-a235-4f3a-a29d-dcbfeeff197e.png)
- D-LinkNet uses Linknet [15] with pretrained encoder as its backbone and has additional dilated convolution layers in the center part
- Linknet is an efficient semantic segmentation neural network which takes the advantages of **skip connections**, **residual blocks and encoder-decoder architecture**. 
- Linknet has shown high precision on several benchmarks, and it runs pretty **fast**.
- It generally has two types, **cascade mode and parallel mode**, both modes have shown strong ability to increase the segmentation accuracy We take advatages of both modes, using shortcut connection to combine these two mode.
- **Transfer learning** is a useful method that can directly improve network preformance in most situation, especiall when the training data is limited.
## Method
### Network Architecture
- D-LinkNet uses ResNet34 pretrained on ImageNet dataset as its encoder. -> ResNet34 : 256*256 // satellite images : 1024*1024
- Considering the narrowness, connectivity, complexity and long span of roads, it is important to increase the receptive field of feature points in the center part of the network as well as keep the detailed information. 
- Using pooling layers could multiply increase the receptive field of feature points, but may reduce the resolution of center feature maps and drop spacial information.
- Dilated convolution can be stacked in cascade mode

![image](https://user-images.githubusercontent.com/74392995/126934597-c0656138-63f4-479b-ac33-550a9fffd039.png)

- The output feature map will be of size 32 × 32 && last center layer will see 31 × 31 points.-> D-LinkNet takes the advantage of multi-resolution features

![image](https://user-images.githubusercontent.com/74392995/126934025-abd7e3c5-ff84-4442-b989-8b7bc64757d5.png)

### Pretrained Encoder
- Transfer learning is an efficient method for computer vision, especially when the number of training images is limited. 
- Using ImageNet pretrained model to be the encoder of the network is a method widely used in semantic segmentation field.
## Experiments
- Datasets: DeepGlobe Road Extraction dataset -> 1024*1024
- BCE(binary cross entropy)
- Dice coefficient loss

![image](https://user-images.githubusercontent.com/74392995/126935119-c4b2144b-af8a-4401-9d65-4d8f1f92cfbb.png)

- Adam optimizer
- Lr 2e-4 and reduced by 5 for 3 times
- 4 batch size
- 160 epochs
## Result
![image](https://user-images.githubusercontent.com/74392995/126935406-60d97249-183c-4e9e-a716-4a3065ad6fe7.png)

- Our baseline Unet had larger receptive field but had no pretrained encoder and the center feature map’s resolution was 8 × 8, which is too small to preserve detailed spacial information.
- LinkNet34 had pretrained encoder which made the network has better representation, but it only had 5 downsampling layers, hardly covering the 1024 × 1024 images
- We found that although LinkNet34 was better than Unet while judging an object to be road or not, it had road connectivity problem
- By adding dilated convolution with shortcuts in the center part, D-LinkNet can obtain larger receptive field than LinkNet as well as preserve detailed information at the same time
### Analysis
- Test time augmentation(TTA) : 0.029
- BCE + dice coefficient loss : 0.005(is better than BCE+IoU loss)
- Pre-trained : 0.01
- Dilated conv : 0.011
- Ambitious data augmentation : 0.01
