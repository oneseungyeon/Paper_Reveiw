# LinkNet
## Review for LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
### Abstract
- Pixel-wise semantic segmentation for visual scene understanding not only needs to be accurate, but also efficient in order to find any use in real-time application.
- As a result they are huge in terms of parameters and number of operations; hence slow too. 
- We propose a novel deep neural network architecture which allows it to learn without any significant increase in number of parameters.
### Introduction
- Inspired by auto-encoders, most of the existing techniques for semantic segmentation use encoder-decoder pair as core of their network architecture.

![image](https://user-images.githubusercontent.com/74392995/126939906-d7b21887-4444-448d-afe5-cd1f0262217e.png)

- Here the encoder encodes information into feature space, and the decoder maps this information into spatial categorization to perform segmentation
### Related work
- Spatial information is lost in the encoder due to pooling or strided convolution is recovered by using the pooling indices or by full convolution. Howerever, Semantic segmentation's retaining spatial information becomes utmost important.
- The generator either uses the stored pooling indices from discriminator, or learns the parameters using convolution to perform upsampling.
  - encoder = disriminator
  - decoder = generator
 - Recurrent Neural Networks(RNNs) were used to get contextual information and to optimize CRF. But the use of RNN in itself makes it computationally very expensive.
 ### Network Architecture
 -The generator either uses the stored pooling indices from discriminator, or learns the parameters using convolution to perform upsampling. Moreover,encoder and decoder can be either symmetric (same number of layers in encoder and decoder with same number of pooling and unpooling layers), or they can be asymmetric.
 
![image](https://user-images.githubusercontent.com/74392995/126941600-664b2ce8-9f84-42f9-8be1-23f71a328ac9.png)

  - Batch normalization
  - ReLU non-linearity
  - Encoder : 7*7 and a stride 2
  - Spatial max-pooling in an area of 3*3 with a stride of 2. 
  - Batch size : 10
  - Lr : 5e-4
  
![image](https://user-images.githubusercontent.com/74392995/126941416-5e604d27-3de6-442e-94db-c0fb7c49c426.png)
![image](https://user-images.githubusercontent.com/74392995/126941539-5899fd56-85fb-4b07-b021-d13518344bad.png)

- By performing multiple downsampling operations in the encoder, some spatial information is lost. It is difficult to recover this lost information by using only the
downsampled output of encoder. 
- Linked encoder with decoder through pooling indices, which are not trainable parameters. Other methods directly use the output of their
encoder and feed it into the decoder to perform segmentation
- we aim at recovering lost spatial information that can be used by the decoder and its upsampling operations. In addition, since the decoder is sharing knowledge learnt by the encoder at every layer, the decoder can use fewer parameters. 
### Result
- The classes present in all the datsets are highly imbalanced; we use a custom class weighing scheme defined as 
![image](https://user-images.githubusercontent.com/74392995/126943548-476f6c76-3443-4743-887e-ddb9aac1e412.png)

- This class weighing scheme has been taken and it gave us better results than mean average frequency
- We use intersections over union (IoU) and instance-level intersection over union (iIoU) as our performance metric instead of using pixel-wise accuracy.
### Conclusion
- Our main aim is to make efficient use of scarce resources available on embedded platforms, compared to fully fledged deep learning workstations. 
- Our work provides large gains in this task, while matching and at times exceeding existing baseline models, that have an order of magnitude larger computational and memory requirements.
