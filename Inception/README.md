# Inception V1 : "Dense 한 Fully-Connected 구조에서 Sparsely Connected 구조로 바꾸는 것"
## 다양한 필터 사이즈
### 문제점
1. 같은 카테고리라고 해도 각 이미지 안의 타겟의 위치 분포는 다양함
![image](https://user-images.githubusercontent.com/74392995/125406583-f1e7f200-e3f3-11eb-8e83-4dda1b99d47c.png)
### 아이디어 : "인셉션 모듈"
1. 다양한 필터 사이즈(1x1, 3x3, 5x5)를 사용하자! + 더 "깊게"가 아닌 더 "넓게"
![image](https://user-images.githubusercontent.com/74392995/125408465-e4cc0280-e3f5-11eb-9975-593b0a94bf00.png)
2. 1x1 필터 사용 : 입력 채널의 수를 줄일 수 있으며 cov 연산 후 추가되는 ReLU를 통해 비선형적 특징을 더 추가 가능
![image](https://user-images.githubusercontent.com/74392995/125408812-40968b80-e3f6-11eb-80be-d34bdde74492.png)
## 더 깊은 층
### 문제점
1. 층을 깊게 쌓을수록 파라미터 수가 증가해 계산 복잡도(연산량)가 증가하며 오버피팅의 우려가 생김
2. 층이 깊어질수록 Vanishing Gradient Problem 발생
### 아이디어 
1. Two auxiliary classifiers : 중간 layer에 auxiliary classifier를 추가하여, 중간중간에 결과를 출력해 추가적인 역전파를 일으켜 gradient가 전달될 수 있게끔 하면서도 정규화 효과가 나타나도록 함
![image](https://user-images.githubusercontent.com/74392995/125410471-df6fb780-e3f7-11eb-8cb3-9bb8196803ad.png)
2. 최종 loss를 계산할 때 모델의 중간 부분의 loss값을 포함
![image](https://user-images.githubusercontent.com/74392995/125410586-ff06e000-e3f7-11eb-846a-df67b0894225.png)
# Inception V2,3
## 더 작은 필터 사이즈
### 문제점
1. Representational Bottleneck : 차원을 줄일수록 정보 손실이 커지는 문제 발생
### 아이디어
1. 5x5 convolution을 두개의 3x3 convolution으로 (=Factorization into smaller convolutions)
![image](https://user-images.githubusercontent.com/74392995/125412676-16df6380-e3fa-11eb-9a5d-c664fffe07ff.png)
2. nxn을 1xn과 nx1로 쪼개기 (=Spatial Factorization into Asymmetric Convolutions)
![image](https://user-images.githubusercontent.com/74392995/125414699-fd24bdfa-ecab-4f9a-9da9-1b871df7e1d3.png)
3. nxn을 1xn과 nx1로 쪼개기 + Wider

![image](https://user-images.githubusercontent.com/74392995/125552694-a284a3d2-b413-41d7-ba50-b9b4275eba75.png)
## Auxiliary Classifiers
### 문제점
1. Auxiliary classifiers은 학습초기에 수렴성을 개선시키지 못함
### 아이디어
1. Auxiliary classifiers에 drop out이나 batch normalization을 적용했을 때, main classifiers의 성능이 향상 -> Auxiliary classifiers는 정확도 향상이 아닌 정규화 목적에 더 부합

![image](https://user-images.githubusercontent.com/74392995/125564513-10d40353-989b-47a1-b053-62a4f9f15d92.png)

## Efficient Grid Size Reduction
### 문제점
1. CNN은 pooling 연산(feature map의 사이즈 감소)과 필터 수 증가(representational bottlenet을 피하기 위해)하는데 이는 연산량을 감소시키지만 신경망의 표현력(representation)도 감소시킴
### 아이디어
1. pooling layer와 conv layer를 병렬로 사용

![image](https://user-images.githubusercontent.com/74392995/125562798-78954946-58e6-44a4-86d5-1de36573ba66.png)
##  Label Smoothing
### 문제점
1. 잘못 라벨링된 데이터가 정답에 크게 영향을 줌
### 아이디어
1. label 을 0또는 1이 아니라 smooth 하게 부여
ex) label이 [0, 1, 0, 0]이면 레이블 스무딩을 거쳐 [0.025, 0.925, 0.025, 0.025]로 변환
![image](https://user-images.githubusercontent.com/74392995/125563473-07bee041-37d8-4675-b9ff-f62a55bb57e4.png)
### Architecture
![image](https://user-images.githubusercontent.com/74392995/125562306-85aa467f-85c7-44d3-9bc8-6a9ba9142873.png)

# Inception V4 : "Uniform"
### 문제점
1. 모듈이 너무 복잡해서 형식화(uniform)할 필요가 있음
### 아이디어
![image](https://user-images.githubusercontent.com/74392995/125422217-6b07a81c-9684-4cc1-a949-76924aefd592.png)

