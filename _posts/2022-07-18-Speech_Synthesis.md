---
title: 음성합성 With DeepLearning
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 딥러닝
key: 2022-07-18
tags: 
  -딥러닝
  -음성합성
use_math: true
---

# Speech Synthesis - With Deep learning

### Introduction

HMM 의 구조는, Input 이 들어오면 Cluster Sequence 로 바꾸고 해당 시퀀스를 피처 시퀀스로 바꾸는 과정이 진행된다. 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/speech%20with%20deep/audio_%EA%B7%B8%EB%A6%BC1.png?raw=true">
  <br>
  그림 1.HMM의 Step
</p> 

해당 그림과 같이 Step 이 진행될 때 우선 Decision Tree 를 통해 Text 를 매핑시키는 과정이 있는데, 해당 과정에서 복잡한 Context 정보를 포착하기 어렵다는 단점이 있다.

또한 Speech의 경우 Observed data가 이전 Step 의 데이터와 독립이라고 판단할 수 없기 때문에 HMM 의 가정을 적용시키는 데에 어려움이 있다. 

따라서 해당 step 들에서 시간에 따른 변화를 포착하고 Clustering 과 같이 Input space 를 Sub space 로 분기함으로써 오는 정보 손실을 잡기 위해 Neural Network 를 사용하는 시도가 활발히 진행되었다.

초반에는 cluster to feature mapping 과정만 DNN 을 사용하여 진행하였지만, 이후 Input 을 Feature 로 mapping 시키는 전체 과정에 DNN 을 사용하는 방식으로 진행되었다.

음성 합성의 경우에는 Duration prediction 인 Regression Model 이기 때문에 초기에 tanh를 통한 Feed forward 방식의 Regression DNN 모델을 사용하게 되었다.

하지만 DNN 의 경우 각 Frame by Frame 이므로 Contextual 정보들을 활용할 수 없었다. 

그렇기 때문에 LSTM 등의 RNN 계열 모델을 통해서 음성 합성 모델링을 하게 되었다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/speech%20with%20deep/audio_%EA%B7%B8%EB%A6%BC2.png?raw=true">
  <br>
  그림 2.LSTM 을 사용한 Modeling
</p> 

음성의 경우 양방향 Sequence 의 영향을 받기 때문에 단반향이 아닌 Bi-directional Modeling 을 주로 하게 된다. 

HMM의 경우 Clustering 을 통해 Speech를 Mapping 하는 방식을 적용하기 때문에 고유한 ‘ㅏ’의 Pitch 및 여러 문맥적 정보를 담고 있다고 하더라도 Sub space 에서 Sampling이 되기 때문에 정보적 손실이 발생한다.

이 경우를 해소하기 위해 Bi-LSTM을 사용하게 된다면 더이상 Sub-space 로 분기하는 것이 아니며, 각 Script 별 동일한 Text더라도 Sequence 의 특징을 반영하기 때문에 정보적 손실이 Clustering 보단 덜 하다.

하지만 LSTM 의 연산은 Auto Regressive 하기 때문에 Computa-tion이 커짐을 확인할 수 있다.

### Tacotron

타코트론의 경우 처음으로 제안된 End to End 음성 합성 모델이다. 

Tacotron의 경우 Text를 집어넣게 되면 Vocoder 직전까지의 Mel spectrogram이 나오게 된다. 

<p align = "center">
  <img width = "800" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/speech%20with%20deep/audio_%EA%B7%B8%EB%A6%BC3.png?raw=true">
  <br>
  그림 3.Tacotron 아키텍쳐
</p> 

전반적인 구성 요소는 다음과 같으며 위 그림과 같은 아키텍쳐를 따른다.

- CBHG : convolution layer + highway network + GRU 
- Attention : Text 와 Speech 의 Alignment 를 담당하는 부분으로 텍스트와 스피치의 유사도를 추정하는 부분이 된다.
- Encoder : Pre-net을 이용하여 모델의 수렴과 최적화를 위해 선형 Layer와 Drop out 을 거친다.
- Decoder : Align된 attention 들을 통해 Mel spectrogram을 추정하는 곳
- Post processing 의 경우 추정된 Mel spectrogram의 성능을 높이기 위한 Post filter의 역할을 한다.

해당 과정에서 나온 Mel Spectrogram 을 학습하기 위해 학습 후 나온 Mel 과 정답 Mel 에 대해 MSE 를 구하여 학습이 진행된다.

Tacotraon 2의 경우 Stop token이 추가된다. 

1의 경우 최대 길이만큼 Auto-Regressive 하게 계속 돌리게 되어 불필요한 부분을 잘라버리는 과정으로 계산 과정이 비효율 적이다.

하지만 Stop Token을 추가함으로써 언제 문장 합성을 마치는지 판단하여 계산을 보다 효율적으로 수행할 수 있다.

### Auto Regressive

여기서 Auto Regressive의 Concept에 대해서 짚고 넘어가야 확실할 것 같다.

자동 회귀라는 식으로 해석할 수 있는데 보통 예측하는 모델을 회귀 모델이라 부른다.

자동회귀 모델은 이렇게 예측하는 과정이 순회적으로 자동적 수행하는 것이라고 할 수 있는데, 보통의 RNN 모델이 다 이렇게 Auto Regressive Model이다.

RNN의 경우 Input 정보를 활용하여 Output 을 뱉고 또 해당 Output 을 Input 으로 넣어 계산하게 되는데, 이 과정 자체를 Auto Regressive라고 하는 것이다.

이 Auto Regressive 모델을 생각 해보면 (활성화 함수 제외) $y_n = W_n(W_{n-1}(W_{n-2}\dots ((W_1)x)$ 의 형태로 중첩되어 계산이 되는 형태이다.

이러한 연산 과정은 시퀀스($n$)이 증가 할수록 연산이 중첩되어 계산 량이 많아지게 된다.

또한 해당 과정에서 필연적으로 $n$ 에서 멀리 떨어진 Sequence 의 정보는 손실이 발생하기 때문에 시퀀스가 매우 길다면 해당 모델은 잘 작동하지 않을 수 있다.

따라서 Sequence 가 긴 모델에서 Stop token 을 지정해 주는 것은 유효한 효과를 낼 수 있으며 이러한 Auto Regressive의 문제를 해결하기 위해 Transformer Model을 음성 합성에 사용하려는 시도도 있다.

#### 음성 합성 Attention

음성합성의 Attention 의 경우 Text 간의 Attention 과 흡사하면서도 다른 부분이 존재한다.

Text 와 Speech의 Attention 의 경우 Speech 1 ,2 ,3 이 ‘ㄱ’이라는 단어를 표현가능하며 이 흐름은 CTC의 Self loop 와 동일하다.

<p align = "center">
  <img width = "800" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/speech%20with%20deep/audio_%EA%B7%B8%EB%A6%BC4.png?raw=true">
  <br>
  그림 4.음성합성 Attention
</p> 

음성 합성의 Attention 의 경우 Align을 위해서 Attention 을 해준다고 하였는데, 위 그림의 맨 왼쪽과 같이 Attention 이 적절하게 된 과정은 Speech의 여러 노드가 Text 하나의 Alignment 와 일치하는 것을 보면 쉽게 파악이 가능하다.

적절한 합성이 진행되기 위해 해당 과정처럼 자연스러운 매핑 과정이 필요한데, 해당 그림의 가장 우측 그림과 같이 가중치가 모호한 상황이라면 해당 Align이 모호해져 합성에서 좋은 성능을 내지 못하게 된다.

기본적으로 Attention 이란 유사도를 구하는 과정이기 때문에 해당 발음이 Text의 어디에서 오는지 적절하게 찾는 과정이라고 할 수 있다.

보통 Attention 의 예시로 정방 행렬을 사용해서 헷갈리는 측면이 있지만, 실제 Attention 은 정방행렬이 아닌 경우가 많다.

위 그림에서도 정방행렬이 아닌 이유는 Speech 와 Text의 align 단위와 Speech의 Duration 문제를 파악하기 위함이다.

따라서 어디서 Duration 이 진행되는지, 어디서 Speech가 바뀌는지에 대한 Alignment 를 학습하기 위해 이러한 Attention 과정이 진행되는 것이라고 할 수 있다.

### Transformer TTS

앞서 이야기한 Auto Regressive한 RNN 계열의 모델이 갖는 한계를 해결하기 위해 Transformer를 기반으로 한 TTS 모델이 등장하게 된다.

<p align = "center">
  <img width = "800" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/speech%20with%20deep/audio_%EA%B7%B8%EB%A6%BC.png?raw=true">
  <br>
  그림 5.Transformer TTS
</p> 

Auto Regressive 연산을 Multi-head Attention 으로 바꾸어 Encoder, Decoder 를 Parallel하게 학습하게 하였다.(평행하게 즉, 순차적 계산이 아닌 행렬 한번에 계산)

하지만 Transformer 의 경우에도 학습은 Parallel하게 행렬 연산으로 진행되지만 예측 과정에서는 Token 을 하나씩 넣기 때문에 Auto Regressive의 문제에서 벗어날 순 없다. 

또한 교사 강요를 사용하기 때문에 Exploration Bias 가 생기게 되며, Transformer를 사용했다고 해서 드라마틱한 성능 향상보단 학습 시간 단축에 의미를 두게 된다.

## Personalize

사람은 각각의 스타일이 있으며, 이러한 스타일은 고유하다고 할 수 있다. 

TTS에서는 이렇게 사람마다의 스타일을 성분화 하여 어떠한 요소가 변화하는지를 파악하여 각 개인의 고유한 목소리 스타일로 합성하고자 하는 시도가 진행되고 있다. 

### Multi - Speaker Speech Synthesis

- D-vector based : 한 사람의 대표되는 성격을 Vector로 만들어 Speaker Encoding Network 를 따로 Training 하여 특성 벡터를 Concat 하는 형태이다. 
- Glbal Style Token : Style을 GST모델이 Soft한 Label을 만드는 것이다. 
  - 해당 모델은 인코더에서 고정된 길이의 벡터로 Input을 압축시킨 후 Attention 을 통과하여 각 Token(labeling 된) 간의 유사도를 추정한다. (각 토큰의 역할 학습)
  - 이후생성된 토큰을 Weighted Sum을 통해 Vocoder Encoder의 넣는다.

- VAE - Tacotron2 : Style을 Latent vector 를 통해 파악하겠다는 뜻이다. 
  - VAE Tacotron의 경우 기존 Fixed 된 Vector 를 뽑는 것과 다르게 Latent Vector 를 뽑기 때문에 보다 Stochastic하게 추정한다.
  - Unsupervised Learning 을 사용하기 때문에 이또한 어떤 Vector가 어떤 역할을 하는지 학습할 때는 알 수 없다.
#### 참고자료
https://www.youtube.com/watch?v=8MlntMp0OFM&list=PL9mhQYIlKEhfyZxdateDkmmpXbTLy_-MN&index=5
