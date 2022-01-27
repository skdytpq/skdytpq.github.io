---
title: Res NEt & Dense Net
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 딥러닝
key: 20220127
tags: 
  -딥러닝
use_math: true
---

# Res Net & Dense Net

### 들어가기 앞서

앞선 게시물에서 CNN에 대한 기본적인 구조를 살펴보았다. 이미지 인식 분야에서는 ImageNet Challenge 라는 대회가 있는데 이 곳에서는 여러가지 모델들이 실험되며 경연을 통해 우승 모델들이 결정된다. 

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/Resnet1.png?raw=true">
  <br>
  그림 1. 성능 연대표
</p>

위 그림은 ImageNet 에서 좋은 성능을 보인 모델들을 연대표로 정리한 자료인데, 여기서 2012년 AlexNet 이 에러율을 큰 폭으로 낮춘 이후 여러가지 모델들이 나오기 시작했다.

그 이후로 2015년 ImageNet 2015에선 100개 이상의 layer를 가진 ResNet이 우승을 차지했다. 

딥러닝은 기본적으로 연산량이 매우 많은 Task인데, 점차 GPU의 성능이 향상되며 기존 컴퓨팅 파워로는 연산될 수 없었던 여러 Deep 한 모델들(layer의 수가 많은) 이 우수한 성능을 보였다.

이러한 흐름은 VGG, GoogLeNet, ResNet을 거치며 더욱 가속화 되었다. 하지만 단순히 layer를 쌓는 것 만으로 CNN모델의 성능을 높일 수 있을까?

결과적으로는 Plain 한 Network 에서 단순히 Layer 만 쌓는다고 모델의 성능은 향상되지 않는다고 여러 실험이 말해주고 있다. 

이번 게시글에서는 CNN 모델 중 ResNet 과 그것과 결이 비슷한(?) Dense Net 의 논문을 review 해 보며 layer를 늘리는 것 뿐만 아니라 어떠한 과정을 거치며 모델의 성능을 높이는지 알아보자.

## ResNet

### layer를 쌓으면 성능이 올라가나?

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/Resnet2.png?raw=true">
  <br>
  그림 2. Layer 깊이와 성능
</p>

위 그림은 ResNet 논문 서두에 나와있는 그림이다.  

AlexNet, VGG 와 같은 경우 기존 CNN구조에서 변화된 것은 없다. 다만 convolution layer를 거칠 때 얼마만큼 거칠 것인지, filter size를 얼마만큼 할 것인지와 같이 튜닝하며 Convolution layer 를 쌓는 방식으로 진행했다.

GoogLeNet과 같은 경우 Inception module이 있지만 구글넷 논문에서의 흐름도 결국 layer를 쌓는 과정을 보다 효율적으로 하기 위해 여러 장치를 추가한 것이라고 이해 할 수 있다.

ResNet 연구진은 네트워크의 Depth 를 늘이는 것만으로도 쉽게 성능을 향상 시킬 수 있는가? 라는 질문에 의문을 갖고 그림 2와 같은 실험을 진행했다. 

여기서 알 수 있는 사실은 Training error, Test error 가 오히려 Plain 한 Network 에서는 오히려 더 떨어진다는 것이다. 

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/Resnet3.png?raw=true">
  <br>
  그림 3. Plain Network 의 구조
</p>

Plain Network 란 위 그림 중앙에 위치한 표 처럼 단순히 Conv, pooling 을 거친 layer 이다. 이 네트워크는 그림 왼쪽에 있는 VGG Net 에서 영감을 받아 생성 한 것이다.

이 네트워크의 Design은 저 흐름을 그냥 받아들이면 될 것 같다. 여기서의 요는 단순이 layer를 쌓는 것이 모델 성능을 향상시키는 데 능사가 아니란 것이다.

단순한 모델의 구조는 ResNet 논문이나 다른 자료를 참고한다면 쉽게 이해 할 수 있을 것이다.

### Residual Block

실제 위 그림 2를 보면 특정 깊이 까지는 모델의 성능이 향상되지만 56 layer라고 해서 20 layer 의 모델보다 성능이 좋지않다.

그렇다면 그 이유는 무엇인가?

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/Resnet4.jpeg?raw=true">
  <br>
  그림 4. Layer 가 깊어질 때의 문제
</p>

위 그림이 시사하는 바 처럼 layer가 너무 깊게 쌓인다면 Back-Propagation 을 통해 학습하고 활성화 함수를 수정한다고 하더라도 학습 시 기울기가 소실될 수 있는 문제가 있다.

input 으로 들어오는 값 $x$ 가 있다면 이 $x$ 가 여러 layer를 통과 할 때 학습이 진행되며 x가 갖고 있는 영향력은 점점 미비해지고 층이 깊어 질 수록 이러한 input 혹은 중간 어떤 값 $\hat{x}$​ 의 영향력이 0에 수렴하는 것이다.

Back-Propagation이란 결국 기울기를 통해 학습을 진행하는 것인데 여러 값에서 기울기가 소실된다면 당연히 안정적으로 학습이 진행되기 어려울 것이다.  

그렇게 해서 나온 것이 **Deep Residual Learning** 이다.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/Resnet5.png?raw=true">
  <br>
  그림 5. Residual Block 구조
</p>

위 그림에서 볼 수 있듯 기존 $x$ 를 통해 출력값 $H(x)$​ 를 받는 방식은 layer를 거치고 비선형 함수를 거치는 방식으로 진행된다.

Deep Residual Learning 은 이러한 기존 방식으로 학습하는 것을 틀어 오른쪽과 같은 방식으로 학습을 진행하는데, 여기서 차이점은 입력 값을 출력 값에 더해주는 것이다. 

이렇게 input $x$ 값을 더해주는 과정을 우리는 short cut 이라 하자.

그렇다면 이렇게 설계해서 얻고자 하는 방향성은 무엇인가? 

### Deep Residual Learning 의 방향성

기존 방식에서 우리는 $x$ 값이 학습을 통해 $H(x)$ 를 우리가 예측하고자 하는 타겟 값 $y$​​ 와 가까워 지길 원한다. 다시 말해 기존의 신경망은 입력값 $x$ 를 타겟값 $y$ 로 매핑하는 함수 $H(x)$ 를 얻고자 하는 목적을 지닌다.

하지만 여기서 $H(x)$ 를 $x$ 에 특정한 값 $F(x)$ 를 더한 값이라고 생각해보자. 다시 말해 원래 input 값인 $x$ 라는 값이 있고 여기에 $F(x)$ 라는 소스를 뿌려 $H(x)$ 라고 만든 것이다. 

그렇다면 관점을 전환 해 보자.  위 그림 5에서 기존 방식에서의 $x$​​ 값의 변화를 수식으로 한번 확인 해 보자면 $F(x) = \sigma_2(W_2(\sigma_1(W_1x)))$​​​ 는 저 그림에서의 layer를 통과한 output이 된다. 

오른쪽 그림에서 Residual block 을 거친 output $y$​ 의 수식은 어떻게 될까? $y= F(x,{W_i})+W_sx$​ 가 된다. 여기서 $W_s$​ 는 혹여나 $ F(x,{W_i})$​ 이 값이 $x$​ 와 차원이 맞지 않을 경우를 대비해 차원을 맞게 교정 해주는 장치라고 생각하면 된다.

저렇게 쌩으로 봤을 때 $ y =  \sigma_2(W_2(\sigma_1(W_1x)))$​ 이 식과 $y= F(x,{W_i})+W_sx$​  이 식 중 어느 것이 미분하여 기울기를 갱신하기 쉬울까? 후자라고 생각하기 쉽다.

또 다르게 생각해보자. 앞서 이야기 한 것 처럼  $x$​ 라는 값이 있고 여기에 $F(x)$​ 라는 소스를 뿌려 $H(x)$​​ 라고 만든 것일 때 우리는 $x$​ 로 $H(x)$​ 를 바로 구하는 것 보다 $x + F(x)$​ 에서의  $F(x)$​ 를 구하는 것이 더 쉽지 않을까?

물론 아니라고 이야기 할 수 있다. 논문의 저자도 명백하게 저 방법이 더 낫다고 이론적으로 증명했다기 보다는 경험적으로 저렇게 $F(x)$​ 를 구하는 방향성으로 학습하는 것이 더 낫다고 이야기한다.

$H(x)$ 를 학습하는 것 보다 $F(x)$​ 를 학습하는 것이 더 효율적이라는 가정에서 출발 한 것이다.

여기서 Residual 은 $H(x) - x$ 를 잔차로 보아 이 $F(x)$ 의 값을 학습하는 의미라고 이해 할 수 있다.

최적화 관점에서도 위 Residual block 을 사용하면 얻는 장점이 있다. 예를 들어 $x$ 가 타겟값 $y$​​ 를 매핑하기에 매우매우 좋은 input 값이라고 해보자. 그렇다면 우리는 $F(x)$ 를 추가적으로 학습 할 양이라고 생각 할 수 있다. 

그렇다면 $F(x)$​ 는 0에 수렴 할 것이다. 왜냐하면 이미 $x$​ 는 너무 잘 학습되어 있는 input 값이라 추가적으로 학습 할 것이 없게 되기 때문이다!

**다시말해 input 값 $x$ 에 대한 정보인 information flow 를 output에서 이어 갈 수 있는 것이다.**

### 기울기 소실 문제 해소

보통 활성화 함수를 우리는 ReLU 를 쓴다 그 이유는 Gradient vanishing 문제를 해소하기 위함이었다. 하지만 이러한 ReLU 도 input 값이 0보다 작다면 0을 출력하기 때문에 Back Propagation 을 거칠 때 기울기 소실 문제를 피할 수는 없다.

우리는 앞서 잔차 블록의 output 값을 $y= F(x,{W_i})+W_sx$ 라고 정의했다. 여기서 활성화 함수를 통과한 값은 $F(x,{W_i})$ 이다. 

여기서 설령 ReLU를 거쳐 저 산출 값이 0이 되었다 하더라도 $y$를 $x$​ 에 대해 미분한다면 적어도 1의 값이 나온다.($W_s$는 잠시 무시하자) 

그렇기 때문에 기울기가 안정적이지 않아 학습이 잘 진행되지 않는 문제를 일정 부분 해소 할 수 있다.

### Experiment

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/resnet6.png?raw=true">
  <br>
  그림 6. ResNet의 구조
</p>

위 그림은 Residual block 을 거친 Network가 다층의 layer에서 어떻게 구성 돼 있는지 나타낸 표 이다.

50개의 layer가 있을 때에 대해 간단히 살펴보면 input 이미지를 $7\times  7$​ 의 convolution layer 를 거친 후 풀링을 거친다. 

이후 4개의 convolution blcok 을 거치는데, 여기서 층이 깊어지고 다수의 Channel 을 얻는다면 파라미터의 양이 증가하기 때문에 bottle neck 을 이용했다.

이후 average pooling 을 거친 후 소프트 맥스 함수를 통해 결과 값을 얻는다.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/Resnet7..png?raw=true">
  <br>
  그림 7. ResNet 성능 표
</p>

위 표에서 ResNet 의 결과를 확인 할 수 있는데 상당히 고무적인 결과는 ResNet 을 사용 시 layer가 152개로 가장 깊을 때의 성능이 가장 좋다는 것이다.

위 결과가 시사하는 바는 short cut 을 거쳐 각 블록이 input 정보에 대한 연산 흐름에 추가하여 학습 하는 것이 layer 를 깊게 쌓을 때 발생하는 여러 문제들을 해결해 줘 학습을 안정적으로 진행하게 했다는 것이다.

## Dense Net

DenseNet 은 ResNet 보다 적은 파라미터를 사용하여 더 높은 성능을 지닌 모델이다.

우리는 앞선 ResNet 에서 이전 block 의 $x$ 값을 이후 block의 output 값에 더해 줌으로써 정보의 흐름을 유지하고 기울기 소실 문제를 일정 부분 해결한다는 것을 알았다.

그림 3에서 가장 우측 네트워크 그림이 ResNet 에 대한 그림인데 그림에서 알 수 있듯 short cut 을 거치기 때문에 각 block 이 위 아래 block 에 연결 된 것 처럼 보인다.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/densenet1.png?raw=true">
  <br>
  그림 8. DenseNet 구조
</p>

위 그림은 DenseNet 에 구조를 나타낸 그림인데, 이 그림과 ResNet 의 그림과의 차이점은 첫번째 layer의 값이 뒤에 있는 모든 layer에 연결 돼 있다. 

이렇게 된다면 마지막 layer 는 직전 layer 의 값만 받는 것이 아닌 이전 모든 layer 의 값을 받게 된다.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/%EB%8D%B4%EC%8A%A4%EB%84%B71.png?raw=true">
  <br>
  그림 9. ResNet 과 DenseNet 연산 형태
</p>

위 그림은 L 번째 layer의 $x$ 값을 표현 한 것인데 ResNet 의 경우 바로 직전 L-1 layer 의 정보를 단순히 더한 형태이지만 DenseNet 의 경우 단순히 더한 것이 아닌 아예 블록 연산의 input 에 모든 이전 $x$ 값을 넣는다.

DenseNet 논문의 저자는 $H_l$ 에서 기존 ResNet 처럼 단순히 이전 $x$ 값을 summation 하는 형태는 network 에서 information flow 이 방해된다고 이야기 한다. 

그렇기 때문에 DenseNet 의 저자는 이전 layer를 모든 다음 layer에 직접적으로 연결하는 것이 information flow 를 향상하는 데  더 효과적이라 이야기 한다.

### Channel wise concat

DenseNet 은 이렇게 layer를 연결하는 과정을 단순하게 가져간다.

channel wise conecat 을 통해 각 layer 를 연결하는데, channel wise conecat란  이전 layer 에서 나온 Channel 을 이후에 모두 이어 붙인다는 것이다. 

channel wise conecat 을 수행하기 위해서는 피쳐맵의 크기가 동일해야 한다. 하지만 Feature map 을 줄이는 연산은 CNN 과정에서 반드시 필요하기 때문에 pooling 연산은 수반되어야 한다.

이러한 문제를 해결하기 위해 DenseNet은 Dense Block 의 개념을 도입한다.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/%EB%8D%B4%EC%8A%A4%EB%84%B72.png?raw=true">
  <br>
  그림 10. DenseNet 구조
</p>

위 그림에서 볼 수 있듯 각 Dense Blcok은 block 안에서 일단 channel wise concat 을 진행하는데, 이후 BN, , 1x1conv, 2x2 avg_pooling 을 진행한다. 이렇게 된다면 Feature map 의 사이즈를 줄이며 이전 블록에서의 Channel 을 이후 block에 모두 이어 붙일 수 있는 것이다.

이렇게 Feature map 의 사이즈를 줄이는 각 과정을 거치는 층을 transition layer 라고 한다. 이러한 transition layer 에는 $\theta$ 라는 파라미터가 존재하는데, 이 파라미터는 transition layer 가 출력하는 채널 수를 조절한다.

예를 들어 입력값의 채널 수가 100 이고 $ \theta$​ 가 0.5 라면 output 으로 나오는 채널 수는 50개가 되는 것이다. 이렇듯 DenseNet 에서 transition layer 는 Feature map 의 크기와 Channel 수를 감소시킨다.

DenseNet은 더한다는 것이 아니라 네트워크의 output을 k 개의 Channel로 유지를 하고 앞에 있는 input에 대한 채널을 계속해서 중첩시키는 것이다. 굉장히 많은 Channel이 쓰여서 파라미터가 많을 거 같지만 사실은 아니다!

기존 CNN 은 한 layer 의 Channel 이 다음 layer 에서 중첩되지 않기 때문에 보통 128,256,512 와 같이 상당한 수의 channel 을 가져간다.

하지만 이렇게 되면 DenseNet 에서는 연산량이 기하급수적으로 많아지기 때문에 k 개의 Channel 을 사용한다. 여기서 k 는 Growth rate 라고 한다.

논문에서는 k=12 를 사용하는데, 이 Growth rate 는 각 layer 가 전체에 어느 정도 기여를 하는지 생각 할 수 있다. (해당 논문에서는 실험 결과 12 까지만 하더라도 충분히 Data set을 표현 가능하다 함)

DenseNet은 이렇게 모든 layer 가모델에  k 만큼을 기여하기 때문에 이 네트워크를  “collective knowledge” 라고 칭한다.

### Feature reuse

앞서 말했듯 dense net 의 경우 단순하게 피처맵을 층을 지날 때 마다 이어 붙인다. 

기존 CNN 모델은 각 layer는 전 후 layer 에만 영향을 주기 때문에 많은 layer를 통과하게 되면 처음 layer 의 Feature map 에 대한 정보는 사라지기 쉽다. 

이러한 문제를 feature reuse 라고 하는데, DenseNet의 경우 끝단의 최종 layer 에도 처음 layer 의 Feature map 에 대한 정보가 있기 때문에 정보가 소실되는 것을 일정부분 방지 할 수 있다.

또한 초기 값을 마지막까지 직접적으로 전달하기 때문에 Back propagation을 진행한다 하더라도 가중치 값이 직접적으로 전달된다. 

따라서 기울기 소실 문제도 완화되며 다양한 layer의 Feature map 을 연결하여 학습하기 때문에 정규화의 효과도 있다고 한다.

### Architecture & Code

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/Densnet%EA%B5%AC%EC%A1%B0.png?raw=true">
  <br>
  그림 11. DenseNet 아키텍쳐
</p>

다음 그림은 DenseNet의 구조이다. 앞서 이야기 했듯이 Dense Block을 거치고 Transition layer 를 통과하는 과정을 거친다. 

위 그림에서 알 수 있듯 각 Dense Block 은 bottle neck 과정을 여러번 거친다. 이러한 결과에서 연산이 많아 파라미터의 수가 많다고 생각 할 수 있지만 Growth rate 인 k 가 기존 Channel 연산을 진행하는 여타 CNN 모델들 보다 상당히 작기 때문에 연산 과정에서의 파라미터 수는 ResNet 보다 적다.

```python
import torch
import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.num_classes = num_classes
        self.growth_rate = 32
        self.base_feature = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                          )

        self.dense_layer1 = nn.Sequential(nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, self.growth_rate * 4, 1, bias=False),

                                          nn.BatchNorm2d(self.growth_rate * 4),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(self.growth_rate * 4, self.growth_rate, 3, padding=1, bias=False),
                                          )

        self.dense_layer2 = nn.Sequential(nn.BatchNorm2d(96),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(96, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer3 = nn.Sequential(nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer4 = nn.Sequential(nn.BatchNorm2d(160),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(160, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer5 = nn.Sequential(nn.BatchNorm2d(192),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(192, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.dense_layer6 = nn.Sequential(nn.BatchNorm2d(224),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(224, 128, 1, bias=False),

                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, 3, padding=1, bias=False),
                                          )

        self.transition1 = nn.Sequential(nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 128, 1, bias=False),
                                         nn.AvgPool2d(kernel_size=2, stride=2)
                                         )

    def forward(self, x):

        # 6
        x = self.base_feature(x)
        x1 = self.dense_layer1(torch.cat([x], 1))
        x2 = self.dense_layer2(torch.cat([x, x1], 1))
        x3 = self.dense_layer3(torch.cat([x, x1, x2], 1))
        x4 = self.dense_layer4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.dense_layer5(torch.cat([x, x1, x2, x3, x4], 1))
        x6 = self.dense_layer6(torch.cat([x, x1, x2, x3, x4, x5], 1))
        x = self.transition1(torch.cat([x, x1, x2, x3, x4, x5, x6], 1))

        return x
```

위 코드는 DenseNet 121 에 대한 코드인데, forward 를 진행할 때 x 값의 input 이 계속해서 증가하는 것을 볼 수 있다.

위 코드는 Growth rate 를 32로 설정하여 layer 를 통과할 때 마다 각 block 에서는 $4\times k$ 개의 Channel 을 통해 학습한 뒤 다시 Growth rate 크기 만큼 Feature map Channel 을 줄인다.

### Experiment

<p align = "center">
  <img width = "800" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/RE%2CDE/DenseNet%EC%8B%A4%ED%97%98%EA%B2%B0%EA%B3%BC.png?raw=true">
  <br>
  그림 11. DenseNet 실험 결과
</p>

그림 12에서 볼 수 있듯 ResNet 대비 더 적은 param 을 통해 더 좋은 성능을 끌어냈고 DenseNet 은 기존 ResNet 의 신경망이 매우 깊어진 것에 더해 264 층 까지 더 깊어짐에도 성능이 향상 됐다는 것을 확인 할 수 있다.

