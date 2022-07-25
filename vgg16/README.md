
# VGG16
Review for Very Deep Convolutional Networks for Large-Scale Image Recognition
## ARCHITECTURE
- Input : 224 * 224 RGB image
- Preprocessing : subtracting the mean RGB value from each pixel
- Filter : 3 * 3 small receptive field -> which is the smallest size to capture the nothion of left/right, up/down, center
- Linear transformation : 1 * 1 convolution filter with stride : 1
- Max pooling : 2 * 2 pixel window, with stride 2
- Fully-Connected layers : 4096(Fully-Connected) + 4096(Fully-Connected) + 1000(soft-max : final)
- Hidden layer : ReLU non-llinerity contain LRN(Local Response Normalisation)
![image](https://user-images.githubusercontent.com/74392995/123589557-5b78d580-d824-11eb-942d-16f9927f91db.png)
## DISCUSSION
- 3 * 3 conv.layers instead of a single 7 * 7 layer? : three non-linear rectification laters instead of a single one + decrease the number of parameters
![image](https://user-images.githubusercontent.com/74392995/123590849-34230800-d826-11eb-8b6b-f5e261418c65.png)
- 1 * 1 conv.layers : increase the non-linearity of the decision funcion
- Deeper depth
## TRAINING
- The training is carried out by optimising the multinomial logistic regression using mini-batch(256) gradient descent with momentum(0.9)
- Weight decay : L2 penalty set to 5 * 1e-4
- Dropout regularisation for the first two fc layer(dropout : 0.5)
- Learning rate(1e-2) is decrease 3 times, when val acc stopped imporving
- The initialisation of the network weights is important : we initialised the first four convolutional layers and the last three fullyconnected layers with the layers of net A
- We did not decrease the learning rate for the pre-initialised layers, allowing them to change during learning.
 ![image](https://user-images.githubusercontent.com/74392995/123596095-ad255e00-d82c-11eb-9bd4-338608322280.png)
- 224 * 224 input images : randomly cropped from rescaled training images + horizontal flipping + random RGB colour shift
### TRAINING IMAGE SIZE
- Let S be the smallest side of an isotropically-rescaled training image, from which the ConvNet input is cropped -> 더 작은 쪽을 256으로 맞추고 가로 대 세로 배율은 같게
- Training scale(1) : S = 256, 384(single scale)
- Training scale(2) : S = 256 ~ 512(multi-scale) ->  For speed reasons, we trained multi-scale models by fine-tuning all layers of a single-scale model with the same configuration, pre-trained with fixed S = 256, 384(single scale)
## TESTING
- Q(test scale) is not necessarily equal to the training scale S(train scale) -> using several values of Q for each S leads to improved performance
- The first FC layer to a 7 × 7 conv. layer, the last two FC layers to 1 × 1 conv. layers -> 오버피팅을 막기 위해 학습때와 다른 구조를 사용
- Multi-crop evaluation is complementary to dense evaluation due to different convolution boundary conditions
## SINGLE SCALE EVALUATION
- Using local response normalisation (A-LRN network) does not improve on the model A without any normalisation layers(A = A-LRN)
- The additional non-linearity does help(C > B)
- The classification error decreases with the increased ConvNet depth(D > C)
- It is also important to capture spatial context by using conv -> 3 * 3 > 1 * 1
- Deep net with small filters outperforms a shallow net with larger filters.
- Training set augmentation by scale jittering is indeed helpful for capturing multi-scale image statistics.
![image](https://user-images.githubusercontent.com/74392995/123765233-859fc580-d900-11eb-83d6-91576ba9295a.png)
## MULTI-SCALE EVALUATION
- Scale jittering at test time leads to better performance
![image](https://user-images.githubusercontent.com/74392995/123754998-70259e00-d8f6-11eb-8174-741cdad35863.png)
## MULTI-CROP EVALUATION
- Using multiple crops performs slightly better than dense evaluation, and the two approaches are indeed complementary, as their combination outperforms each of them.
![image](https://user-images.githubusercontent.com/74392995/123755158-9a775b80-d8f6-11eb-894c-6f6f3b6105cb.png)
