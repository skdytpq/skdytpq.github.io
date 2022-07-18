---
title: Speech Synthesis - Without Deep learning
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 
key: 20220718
tags: 
  -딥러닝
  -음성합성
use_math: true

---

# Speech Synthesis - Without Deep learning

### Introduction

Text 가 입력일 때 그 것을 Speech로 바꾸어 주는 것이다.  

이 음성합성의 목적은 크게 두가지가 있다. 

- Intelligibility : 얼마나 음성이 또박또박 발음되는지
- Naturalness : 얼마나 사람의 목소리와 비슷한지

기존에의 음성합성은 Rule-based 로 진행되었는데 주로 Formant의 위치를 잡거나 조음 기관들의 모양을 복사하거나 하는 방법이었다.

하지만 이러한 방법은 성능과 비용적 측면에서 매우 복잡하기 때문에 잘 사용하지 않는다.

Concatenative synthesis의 경우 Speech 데이터를 매우 작은 Unit 으로 쪼개어 사용하는데, DB의 크기에 영향을 많이 받으며 음성의 변화같은 분야에서는 로버스트 하지 않다.

통계적 합성 측면에서의 SPSS 는 HMM 기반으로 음성을 합성하게 된다.

이 모델은 Text Analysis Module과 Text 와 Speech 의 길이를 맞춰주는 alignment 측면의 Duration Module 이 존재한다. 

이러한 Duration Module을 통과하여 Acoustic Model 이 음성 피처를 추정하며 마지막으로 Feature 를 음성 파형으로 바꿔주는 작업을 거치게 된다. 뉴럴 기반의 SPSS는 Acoustic Feature 의 처리를 RNN 계열의 모델을 통해 진행되게 된다.

하지만 이러한 통계 기반 접근 방식은 Linguistic Domain 의 지식이 매우 많아야 하므로 가장 활발하게 연구되는 분야는 End-to-End의 Speech Synthesis 이다.

- Utterance to Encoder : Text embedding
- Attention : alignment
- Decoder : Mel reconstruction
- Vocoder : to Wave form

의 형태로 진행되게 된다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/speechsynthesis/audio_%EA%B7%B8%EB%A6%BC4.png?raw=true">
  <br>
  그림 1.class 별 별점
</p>


### Unit Selection

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/speechsynthesis/audio_%EA%B7%B8%EB%A6%BC3.png?raw=true">
  <br>
  그림 2. Unit Selection
</p>

Unit Selection 이란 근거를 통해 조그만한 Speech Unit 을 쪼개고 이어붙이는 작업이다.

해당 작업은 실제 음성을 사용하기에 음질의 퀄리티는 좋지만, DB의 규모에 따라 quality의 영향을 많이 받는다.

Unit Selection 은 긴 Syllable데이터 부터 di-phone 조각까지 사용하며, 여러 조각들로 흩뿌려진 각 unit 들은 Viterbi algorithm, Beam-search 를 통해 Selection 된다.

비터비 알고리즘의 경우 여러 후보군 중 가장 확률이 큰 후보군을 선택하는 알고리즘인데, 이 알고리즘을 사용하기 위해 Forward, Backward Prob를 다 계산하며, Beam-search의 경우 가지치기 형태로 선택하는 것이라고 생각할 수 있다.

이 Selection의 근거가 되며 Parameter 를 조정하는 식으로는 Concatenation Cost와 Target Cost 가 있는데, 이 부분에 대한 수식적인 표현은 Deep learning 의 Loss Function 과 비슷하다.

이 Unit Selection 은 한계가 매우 명확하기 때문에 거의 연구되지 않고 있다.

### HMM Based

HMM 에 대한 설명은 이전 게시글에 있으니 기본적인 설명만 하겠다.

HMM의 두가지 가정에 의존한 모델이다

- 현 상태는 바로 이전 상태에만 영향을 받는다.
- Output(Observation)은 독립이다.

이러한 가정은 모델링과 계산을 편리하게 하지만, 이러한 Assumption 은 Sequential 인 Audio 에 정보를 처리할 때 발생할 수 있는 Error 가 있다.

Speech Synthesis 의 HMM 모델은 Speech가 한 방향으로 흐른다는 사실에 의거하여 Left to Right 모델이 된다. 

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/speechsynthesis/audio_%EA%B7%B8%EB%A6%BC5.png?raw=true">
  <br>
  그림 3. HMM based speech Synthesis
</p>

우선 HMM 의 Training 부터 살펴보자.

HMM의 경우에는 Speech DB를 통해 Training 을 진행하게 되는데, 우선 해당 Data에 대해 Labeling 이 들어가게 된다. 

여기서 라벨링은 인간이 명시적으로 align 해주는 것이기 때문에 성능의 차이가 있을 수 있다.

이후 Speech 에 대해 Feature 를 뽑게 되는데 여기서 우리가 Observed Data라고 취급하는 것은 speech이며, text가 해당 작업의 Hidden state 가 된다. 

여기서 트레이닝 이후 Tree based Clustering 이 진행되는데, 해당 과정은 Speech 데이터에서 모든 발음을 커버할 수 없기 때문에 DB에 없던 억양이나 발음에 대한 정보에 대해서도 Robust 해지기 위해 진행하는 작업이다.

해당 Training 을 마치게 된다면 Hidden state 의 $\lambda$ 가 Update 가 된 상태가 된다. 이후 해당 모델을 통해 Synthesizing 작업이 진행되게 된다.

해당 작업에서는 우선 Text가 들어오면 해당 Text를 Speech 에 맞게 analysis 하는 작업이 진행된다. 

이 과정은 단어의 어디까지가 Speech data에 있는 어떤 형태로 발음이 되는지에 대한 Alignment 과정이라고 생각할 수 있다.

이 alignment 과정에서는 각 음소정보, Context 정보(초성, 중성, 종성, 파열음인지에 대한 음운학적 정보) 와 Symbol 을 참고한다.

이후 Tokenization , Token to words(약어를 읽는 것, 12.3 -> 십이점 삼) , Word to Syllable(Phonetic rule), syllable to phonemes 과정을 거치게 되어 Text Feature 를 얻는다.

이후 이러한 Text 를 토대로 HMM 을 통해 계산한 우도 함수의 Argmax 인 $\hat{o}$ 를 뽑는 형식이 된다.

- training : $\hat{\lambda} = argmax_\lambda p(o \vert x,\lambda)$ 
  - text  에 맞는 Speech 가 나올 확률을 최대화 하는 Parameter $\lambda$ 를 최적화
- synthesis : $\hat{o} = argmax_o p(o\vert x,\hat{\lambda })$
  - 최적화된 $\lambda $ 를 토대로 hidden state $x$ 가 주어졌을 때 가장 확률이 높은 $o$ 출력

#### 참고자료

 https://velog.io/@tobigsvoice1516/2%EC%A3%BC%EC%B0%A8-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B8%B0%EB%B0%98-%EC%9D%8C%EC%84%B1%ED%95%A9%EC%84%B11

 https://www.youtube.com/watch?v=KvoQm7kGXKU&list=PL9mhQYIlKEhfyZxdateDkmmpXbTLy_-MN&index=4
