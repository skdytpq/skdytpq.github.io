---
title: CNN(Convolutional Neural Network)
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 딥러닝
key: 20220123
tags: 
  -딥러닝
use_math: true
---

# CNN(Convolutional Neural Network)

## 들어가기 앞서

기본적으로 딥러닝을 배울 때 우리는 단층 퍼셉트론, 다층 퍼셉트론에서 어떻게 학습이 이루어지는지에 대해 배웠고 또한 어떻게 학습을 잘 할 수 있는지에 대한 알고리즘, 최적화 기법, 초기화 기법 등을 공부했다. 

딥러닝의 문제점으로 지적 받을 수 있는 기울기 소실 문제(Gradient-Vanishing) 과 Overfitting, Internal Covariance shift와 같은 문제를 어떻게 하면 더 최소화 시킬 수 있는지에 대해 Activation Function을 조정 해보고 Drop out, Batch nomalization과 같은 방법론과 알고리즘을 통해 해결할 수 있는지 찾아봤다. 

그렇다면 이제 이미지 관련 분야에서 사용되는 대표적인 딥러닝 모델인 CNN에 대해서 살펴보자.이번 글은 수식을 최소화 하고 구조와 흐름을 파악하는 데 집중해서 써보겠다.

## Image

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-06%20165516.png?raw=true">
  <br>
  그림 1. 신경망 구조
</p>

우리는 그림 1 과 같은 신경망 구조를 배워왔다. 여기서 유념해야 할 점은 각 노드가 받는 input의 값은 1차원 벡터 형태라는 것이다. 

이제 이미지에 대해서 생각해보자.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/CNN1/%EA%B7%B8%EB%A6%BC1.jpeg?raw=true">
  <br>
  그림 2. 귀여운 강아지
</p>

그림 2는 귀여운 강아지에 대한 그림이다. 이 강아지의 눈의 위치는 숫자로 어떻게 표현할 수 있을까? 우리에게 익숙한 데카르트 좌표계를 사용한다면 (x,y)라고 손쉽게 표현할 수 있을 것이다. 

다시말해 이러한 이미지를 표현하기 위한 축(기저)는 2개 이기 때문에 이 이미지는 2차원이라고 할 수 있다. 

우리는 이 그림을 보고 강아지인지 고양이 인지 분류를 하기 위한 모델을 만든다고 하기 위해 우선 모델에 이 데이터를 input으로 넣어줘야 한다.

그러기 위해 신경망 모델에 넣기 전  $n\times k$​​​ 꼴의 저 ​이미지를 1차원 벡터의 형태로 flatten 시켜줘야 한다. 

하지만 1차원 벡터로 이미지를 넣는다면 문제점은 없을까?

예를 들어 강아지의 얼굴을 생각해 본다면, 대부분의 강아지는 눈 밑에 코가 있고 코 밑에 입이 있다. 또 한번 눈에 대해서만 생각해 본다면, 눈에 해당하는 픽셀 하나하나들은 각각 비슷한 값을 지녔을 것이다. 검은 눈의 강아지라면 눈의 픽셀 하나하나가 비슷하게 검은색일 것이다.

**이렇듯 이미지의 경우 인접 변수간 높은 상관관계가 있다.**

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/CNN1/%EA%B7%B8%EB%A6%BC2.jpg?raw=true">
  <br>
  그림 3. 귀여운 강아지2
</p>

또 이 강아지의 사진을 보자. 이 강아지도 마찬가지로 귀엽지만 여기서 주의깊게 살펴볼 점은 이미지 자체로 봤을 때 앞서 그림2의 강아지와 눈의 위치가 다르다. 또한 자세와 표정도 사뭇 다르다.

 쉽게 말해 좌표로 찍어봤을 때 눈에 해당되는 좌표가 그림 2의 강아지와는 사뭇 다르다. 하지만 이 강아지도 어쨋든 눈, 코, 입이 똑같이 있다.

이러한 눈, 코, 입을 갖고 있다는 강아지의 **부분적인 특성은 고정된 위치에 등장하지 않는다.**

굵은 글자로 표시된 두가지가 이미지 데이터의 주요한 특징이라고 할 수 있다. 

하지만 이미지 데이터를 1차원 벡터로 단순하게 flatten 해 학습을 진행한다고 하면 이러한 두가지 특징에 대한 정보는 소멸 될 가능성이 높다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/CNN1/%EA%B7%B8%EB%A6%BC4.png?raw=true">
  <br>
  그림 4. 이미지 Y 의 flatten
   출처.https://wikidocs.net/62306
</p>

그림 4에서 직관적으로 살펴볼 수 있듯이 y라는 2차원의 문자를 1차원으로 바꾼다면 우리가 알아보기 힘든 모양으로 바뀐다. 

이렇듯 기계에서도 2차원 데이터를 1차원으로 변환하며 변환 전에 갖고 있던 공간적인 구조에 대한 정보가 유실된 상태에서 학습을 하기 때문에 제대로 학습 하기 어려울 것이다.

그렇다면 우리는 공간적 정보의 유실을 최소화 한 1차원 데이터를 학습 시킨다면 더 좋은 성능을 낼 수 있다고 기대할 수 있을 것이다.

이러한 생각에서 나온 모델이 합성곱 신경망(Convolutional Neural Network)이다.

## CNN의 구조

이미지의 픽셀 하나하나는 실수이다. 또한 색을 표시하기 위해 우리는 RGB 3개의 실수로 표현하는데, 다시말해 하나의 원본 이미지는 Red, Green, Blue 세 가지 필터를 거쳐 나오는 이미지라고 할 수 있다.

예를 들어 $100\times 100$ 의 크기를 갖는 이미지가 있다고 한다면 이미지의 색상을 표기 하기 위해 총 shape는 (100,100,3) 인 3차원 텐서가 되는 것이다. 여기서 이 3이라는 숫자를 우리는 **채널**이라 부르자.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/CNN1/Convolution_schematic.gif?raw=true">
  <br>
  그림 5. image filter feature map
</p>

이제 위 그림에 대해서 이야기 해보며 CNN 의 아이디어에 대해 알아보자.

위 그림에서 image는 실제 이미지의 각 픽셀 값을 나타낸 것이라고 하자.  위 image 의 사이즈는 $5\times 5$ 이다. 여기서 움직이는 $3\times 3$ 의 판은 **filter(or kernel)**이라 하는데, 이미지를 계속해서 훑어 가며 합성곱 연산을 수행한다. 

쉽게 생각해보자면 필터의 사이즈인 $3\times 3$​​​의 판을 전체 이미지에 대어 훑어가며 연산을 진행하는 것이다. 

연산의 과정은 쉽게말해 판에서 해당하는 각 값과 판에 덧붙여진 이미지의 각 수치 값들에 행렬곱을 진행한 후 각 행렬 원소를 더하는 것이다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/CNN1/%ED%95%A9%EC%84%B1%EA%B3%B1%20%EA%B3%84%EC%82%B0%20%EC%A0%88%EC%B0%A8.png?raw=true">
  <br>
  그림 6. Feature map 연산 과정
  출처.http://taewan.kim/post/cnn/
</p>

위 그림을 보면 좀 더 이해가 될 것이다. 이후 합성곱을 진행한 각 원소들을 넣어 놓는 곳을**Feature Map** 생각할 수 있다.

그렇다면 이 Feature map에서 들어가는 정보는 무엇일까? 우리는 필터를 통해 읽어가면서 필터 사이즈 영역에 있는 이미지의 값들을 Feature map에 있는 하나의 원소로 표현할 수 있게 되었다. 

즉 다시말해 공간적인 특성에 대한 정보도 함축해서 Feature map 에 넣었기에 단순히 1차원 벡터로 flatten 하는 것 보다 공간적인 정보를 더 가져갈 수 있을 것이라 기대할 수 있다.	

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/CNN1/%EA%B7%B8%EB%A6%BC_8.jpg?raw=true">
  <br>
  그림 7. Feature map 생성 과정
  출처. http://taewan.kim/post/cnn/
</p>

위 그림은 Feature map이 생성되는 과정이다. 여기서 각 step은 필터가 훑는 횟수를 의미한다 할 수 있는데, 필터가 총 9번을 훑었기 때문에 Feature map 의 사이즈는 $3\times 3$ 이 된 것이다.

위 그림에서는 필터가 이미지를 오른쪽으로 한 칸씩 이동하며 이미지를 훑는데, 우리는 하이퍼 파라미터를 통해 이 이미지를 몇 칸씩 이동하며 훑을지 정할 수 있다. 이 값을 우리는 **stride**라고 이야기하자.

위 그림들은 채널이 하나일 때의 이미지를 나타낸 것이다. 하지만 우리가 실생활에서 보는 이미지는 보통 다수의 채널을 갖고있기 때문에 우리는 각 채널별로 필터를 훑은 후 나온 n개의 Feature map 을 최종석으로 합쳐 최종 Feature map 을 생성한다. 아래의 그림은 3개의 채널을 가진 이미지의 Feature map 생성을 나타낸 것이다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/CNN1/3%EC%B1%84%EB%84%90_%ED%95%A9%EC%84%B1%EA%B3%B1.png?raw=true">
  <br>
  그림 8. 3개의 channel에 대한 feature map 생성
  출처.http://taewan.kim/post/cnn/
</p>

여기서 유념해야 할 사실은 위 연산에서 사용되는 필터의 수를 3개라고 하는 것이 아닌 각 채널 당 1개의 필터를 갖고 연산을 진행한다는 것이다.

우리는 여기서 다수의 Feature map 을 생성할 수 있다. 예를 들어 이미지에 대한 정보를 여러 측면에서 얻고 싶다면 필터의 수를 늘리면 된다.다양한 관점에서 이미지를 보겠다는 것이다. 

예를 들어 64개의 Channel 을 갖는다고 하는 것은 64가지의 filter를 사용한다는 것으로 64가지의 각기 다른 연산으로 이미지를 파악하겠다는 이야기이다.

그렇다면 필터 당 하나의 Feature map 이 생성된다. 따라서 Feature map 의 수 는 필터의 수라고 생각 할 수 있다.

이렇게 필터를 거쳐 Feature map 의 생성되는데, 우리가 filter로 이미지를 훑는 이유는 이미지의 공간적인 정보도 추가하여 학습하기 위함이라고 앞서 말한 바 있다.

그렇기 때문에 CNN의 학습 때의 가중치는 각 filter 들의 값이라 할 수 있다. 즉, 가중치가 갱신되는 것은 각 filter 들의 값들이 변하는 것이다. 

또한 합성곱 연산은 원소들을 선형결합 하는 연산의 의미도 띈다고 앞서 말했기 때문에 이러한 연산을 선형 결합이라 생각한다면, 이 선형 결합의 결과물들이 Feature map의 하나하나의 원소가 되는 것이다.

그렇다면 우리는 Feature map 을 ReLU와 같은 적당한 활성화 함수에 넣어 비선형성을 부여할 수 있다. 

따라서 CNN은 filter 값으로 표현되는 가중치, 선형 결합의 결과물인 Feature map 을 활성화 함수에 집어 넣는 과정을 거친다. 

하지만 이러한 과정을 거쳐 이미지를 학습하기에는 문제가 있다. MLP 와 같은 신경망을 사용하여 연산이 진행될 때 우리는 비용함수 측면에서 매우 많은 파라미터를 학습해야 한다는 계산 복잡도의 문제점에 봉착하게 된다.

CNN 에서 이런 문제는 더 심해진다. 통상 CNN 과정에서 하나의 Channel 즉, 다시말해 한가지 관점에서 데이터를 바라보는 필터로 학습을 진행하기엔 성능 향상의 기대치가 낮아지기 때문에 성능을 올리기 위해 다수의 채널을 사용한다. 9

하지만 여기서 모든 filter의 weight 즉, 가중치가 다르다면 Channel 이 늘어날 수록 혹은 CNN의 층이 깊어질 수록 파라미터가 기하급수 적으로 증가한다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/CNN1/CNN%20%EC%9B%A8%EC%9D%B4%ED%8A%B8%20%EC%85%B0%EC%96%B4%EB%A7%81.png?raw=true">
  <br>
  그림 9. 파라미터 개수의 증가
  출처. https://ireneli.eu/2016/02/03/deep-learning-05-talk-about-convolutional-neural-network%EF%BC%88cnn%EF%BC%89/
</p>

위 그림을 예로 보자. $1000\times 1000$​의 2차원 이미지를 $10\times 10$​​의 filter로  Feature map을 만들며 이 값을 받는 hidden unit 의 수는 백만 개 라고 해보자.

만약 여기서 각 filter의 유닛이 가중치가 다 다르고 모든 픽셀에서 고유한 가중치를 갖는 다고 한다면 $10\times 10\times 1000\times 1000$​개의  파라미터가 생긴다. 그렇다면 총 연산할 때 계산해야 하는 파라미터의 수는 $10^{12}$ 개가 되는 것인데, 이러한 연산은 거의 불가능 한 연산이다.

하지만 filter가 옮겨가며 연산 할 때마다 동일한 가중치를 가지고 연산하게 된다면 어떻게 될까? 그렇다면 파라미터의 개수는 $10\times 10$ 이 되는 것이고 백만 개의 hidden unit 이 있다고 하더라도 총 연산량은 $10^{8}$​ 이 된다. 약 1만 배 정도 차이가 나는 것이다.

실제 연산에서는 다수의 채널을 사용하기 때문에 이렇게 filter 별 가중치를 동일하게 하지 않는다면 계산량은 기하급수적으로 증가 할 것이다. 

이렇게 filter 별 가중치를 동일하게 하는 것은 가중치를 공유한다는 의미로 **weight sharing** 이라 한다. 

또한 계산량을 줄이기 위해 한가지 연산을 추가적으로 사용하는데 이 것을 pooling(풀링) 이라 한다. 풀링은 Feature map 을 Down sampling 하여 계산량을 줄이는 방법이다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-06%20165516.png?raw=true">
  <br>
  그림 10. Max pooling
</p>

풀링은 위 그림 처럼 진행되는데 위 그림과 같이 $2\times2$ 의 사이즈의 풀링을 거친다면 총 Feature map 의 크기에 $\frac{1}{4}$​ 크기가 되는 것이다. 위 연산은 총 4개의 값 중 최댓값을 추출하는 Max Pooling 이라는 방법인데 이 외에도 평균값을 추출하는 Average pooling 등 다양한 pooling 기법이 있다.

또한 풀링을 시행하는 또 다른 이유가 있는데, CNN 에서 여러 layer를 거치면 학습해야 하는 파라미터의 수는 매우 많아진다. 따라서 Over fitting의 우려가 있다.  따라서 우리는 풀링 과정을 거치며 Feature 의 사이즈를 줄여 Over fitting 의 우려도 일정부분 해소하려는 목적이 있다.

### Pytorch

이제 파이토치로 구현한 간단한 코드를 보며 CNN 중 VGG의 구조에 대해 추가적으로 설명해 보겠다.

```python
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        
        self.features = features #convolution
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )#FC layer
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x) #Convolution 
        x = self.avgpool(x) # avgpool
        x = x.view(x.size(0), -1) #
        x = self.classifier(x) #FC layer
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```

```python
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # 16 +3 =vgg 19
    'custom' : [64,64,64,'M',128,128,128,'M',256,256,256,'M']
}
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
                     
    return nn.Sequential(*layers)
```

위 코드는 CNN 중 VGG에 대한 코드인데 이번 챕터에서는 VGG에 대한 설명 보다는 CNN이 어떤 식으로 적용되는지에 대해 알아보겠다.

위 코드는 1000개의 이미지를 분류하기 위한 CNN 모델인데, 우선 Class 내부에서 VGG에 대한 내용을 상속받는다.

우선 두번째 코드펜스에서의 make_layers에 대해 살펴보자. 여기서 input channel 은 이미지가 기존에 가지고 있는 3(RGB)를 의미한다. 

그리고 위에서 정의한 cfg의 리스트를 입력받아 레이어를 추가하는데, 여기서 A 에 대해서 이야기 해보자. 여기서 각 cfg는 VGG11, 13,16,19 의 모델 구조를 의미한다.

for 문으로 들어간 v 는 [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'] 형태의 구조를 받게 된다.

첫번째 64는 ‘M’이 아니기 때문에 convolution 층을 생성하게 된다.

처음으로 nn.Conv2d를 살펴보자. 우리가 앞서 정의한 in_channels=3 에서 v 인 64개의 채널이 만들어지게 된다. 그렇게 되면 **Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))** 와 같은 결과값이 생긴다.

이 결과값이 의미하는 것은 3개의 채널을 가진 이미지를 size가 $3\times 3$ 인 filter 로 64개의 Feature map 을 생성한다는 의미이다. 여기서 stride 는 1이고 padding 도 1이기 때문에 Feature map의 크기는 input 이미지의 사이즈와 똑같이 된다.

이후 우리는 batch_norm=False로 지정해놓았기 때문에 Batch Nomalization 과정은 거치지 않는다. 그 이후 ReLU를 거치면 output의 Channel 은 64가 되기 때문에 in_channel 의 값을 3이 아닌 64로 갱신해준다.

이후 M 이라면 Max pooling을 진행 해 주는데 여기서 size 가 $2\times 2$​ 이고 stride 도 2 이기때문에 총 4개 값 중 최댓값을 뽑고 이 연산 중 중복되는 원소는 없다는 의미이다. 따라서 여기서 실제 이미지의 size 인 $n\times n$ 에서 $\frac{n}{2} \times \frac{n}{2}$ 크기의 Feature map 을 생성하게 된다.  channel 의 수는 변화가 없기 때문에 64 채널이 동일하게 유지된다.

이러한 또 같은 과정을 input channel 이 128, 256, 512 과정을 거치며 층을 쌓게 된다. 이 연산의 층은 nn.Sequential(*layers) 에 저장되어 모델링 작업이 다 끝나게 된다. 

```python
conv = make_layers(cfg['A'], batch_norm=True)
CNN = VGG(make_layers(cfg['A']), num_classes=10, init_weights=True)
```

이후 앞서 정의한 VGG 클래스에 위 A를 넣어주게 된다면 VGG 11 모델이 생성된다. 

여기서는 self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) 를 통해 $7\times 7$ 의 크기로 average pooling을 진행한다. 이후 신경망 학습을 진행하게 되는 것이다.

여기서 유념해야 할 점은 실제 input image 의 크기는 커지지 않는다는 것이다. 여기서 변하는 것은 channel 인데, 이 channel 이란 filter 가 얼마나 다양하게 훑는지에 대한 것이다. 

