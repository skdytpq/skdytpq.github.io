

---
title: Fastspeech
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:

  - 기초통계
key: 20220822
tags: 
    -  음성합성
use_math: true

---

# Fastspeech

## Introduction

TTS의 경우 딥러닝의 발전으로 최근 Attention 기법을 활용한 모델이 많이 나오고 있다. 예를 들어 Tacotron1,2 DeepVoice,ClariNet 등이 존재한다.

해당 모델들은 주로 Text를 Input으로 받은 후 Auto regressive하게 mel spectrogram을 만든 후 해당 스팩트로그램으로 Vocoder를 활용해 음성을 합성한다.

신경망 기반 TTS는 mel-spectrogram을 autoregressive하게 만들게 되는데, 일상적으로 mel 을 만드는 sequence는 길기 때문에 아래와 같은 여러가지 문제에 봉착하게 된다.

- 낮은 추론 학습 속도를 갖게 된다. CNN, Transformer 기반 TTS 모델의 경우 RNN기반의 모델보다 속도를 상승시킬 수 있으나, 모든 모델은 Sequence 학습을 진행하고 해당 길이는 매우 길기 때문에 본질적으로 속도가 낮다.
- Attention의 잘못된 할당과 학습의 에러 때문에 생성된 spectrogram은 word skipping repeating 문제를 겪게된다.(낮은 robustness)
- 합성의 경우 조작 가능성이 매우 낮다. 이전의 auto regressive 한 모델은 text와 speech의 명시적 할당 없이 spectrogram을 1대1 대응을 시켰다. 결론적으로 직접적으로 voice speed와 운율적인 느낌을 조작하기 힘들다. 

위 도전적인 문제 중 핵심은 낮은 추론 속도와 낮은 강건함이다. 이러한 문제를 해결하기 위해 Fastspeech 를 제안하였으며, Fastspeech의 경우 mel-spectrogram을 auto regressive하지 않게 생성한다. 

또한 mel-spectrogram의 경우 실제 phoneme 의 sequence보다 훨씬 길기 때문에 phoneme과 spectrogram 두 sequence의  mismatch 를 해결하기 위해 Fast speech는 Length Regulator를 도입하였다. 

이 regulator의 경우에는 phoneme의 upsamples를 진행한다. 이 과정은 각 phoneme의 duration을 토대로 진행되는데, regulator의 경우 이러한 phoneme의 duration 을 예측하는 duration predictor 위에 Built 되게 된다.

앞선 세가지 큰 도전들에 대해 Fast Speech는 다음과 같은 효과를 볼 수 있다.

- Auto-regressive하지 않고 phoneme 과 Parallel하게 생성되는 spectrogram은 음성 합성의 속도를 매우 높여준다.
- auto-regressive 한 모델에서 attention을 기반으로 한 allignment 와 다르게 Phoneme duration predictor 의 경우 Phoneme과 Mel- spectrogra 간의 강한 allignment를 보장한다. 해당 방식으로 얻을 수 있는 이득은, attention 과정에서 잘못 학습되는 문제를 피할 수 있으며 해당 문제에서 야기되는 단어 생략, 단어 반복의 문제를 피할 수 있다.
- Length regulator의 경우 단순하게 phoneme 의 각 duration을 줄이거나 늘림으로써 voice speed에 따라 쉽게 변화할 수 있으며, 음율적인 부분도 각 phoneme 사이에 break를 더해주어 조작할 수 있다.

결과적으로 Fast speech의 경우 음성 합성에서 보다 큰 유연성을 가질 수 있으며 phoneme의 attention 을 통해 mel 을 만드는 것이 아니기 때문에 attention 학습에서 파생될 수 있는 issue를 피할 수 있다. 

논문의 저자가 밝히기에 Fastspeech의 경우 270배의 spectrogram greneration 속도를 달성하였으며 38배 속도의 음성 합서을 이루었다. 

## Architecture

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/fastspeech/fastspeech_1.png?raw=true">
  <br>
  Fastspeech 전체 아키텍쳐
</p>

Fastspeech의 아키텍쳐는 다음과 같이 크게 4 부분으로 나눌 수 있다. 

### Feed Forward Transformer

Fastspeech의 아키텍쳐는 Self attention 기반의 Transformer 모델과 1D Convolution Layer로 구성되어있다. 논문의 저자는 이 구조를 Feed-Forward Transformer(FFT)로 부르는데, 해당 블록은 Phoneme, Mel-spectorgram Side 별로 N 개가 중첩되어있다.

다시 말해 Phoneme 간의 attention 이 이루어진 후 Length Regulator를 통과한 이후엔 mel-spectrogram 의 측면 즉, spectrogram 생성 관점에서의 attention 이 진행되게 된다고 생각할 수 있다. 그 이유는 Length Regulator 를 통과한 Vector는 Mel-spectrogram 과 같은 Length를 갖고 있기 때문이다. 

모든 FFT 블록은 (b)에서 보듯 self-attention 과 1D Conv 로 구성되어 있는데, 이 Self-attention을 진행하여 각 위치간 정보를 추출한다. 

또한 특징적인 것으로 Attention 이후 FC-layer 를 거치는 것이 아닌, 1D convolution 을 거치게 되는데, 이렇게 convolution 을 진행한 이유는 phoneme 과 mel-spectrogram은 근처에 있는 hidden state의 영향을 많이 받기 때문이다. 

사실 auto regressive한 모델도 각 Sequence 간의 연관성을 위해 진행된 것인데, 해당 논문은 중첩 계산을 하기보단 Convolution을 통해 Sequence 정보를 묻힌 것이라고 생각할 수 있다.

### Length Regulator

사실 해당 논문에서 이 Part가 Fastspeech의 특징 중 핵심이라고 할 수 있다.

해당 Block 은 phoneme 과 mel-spectrogram의 mismatch 를 해결하기 위한 part이다. 대체적으로 phoneme의 sequence가 mel-spectrogram의 sequence보다 더 작으며 같은 phoneme이 여러번 중첩되어 mel-spectrogram으로 나타나게 된다. 

해당 과정에 대해 설명하기 위해 우선 phoneme 과 mel-spectrogram의 legnth가 맞다고 가정하자. phoneme A 의 duration 을 $d$ 라고 할 때 length ragulator는 phoneme A 의 hidden state를 d 배로 늘린다.  이렇게 된다면 결과적으로 mel-spectrogram과의 길이가 동일해지게 된다.(모든 phoneme에 대해 동일하게 진행.)

phoneme의 sequnece를 $H_{pho}  = [h_1,h_2,\dots , h_n]$ 이라고 하며 각 phoneme의 Duration sequence 를 $D = [d_1,d_2,\dots , d_n]$ 이라고 할 때, mel-spectrogram의 sequence의 총 길이는 $\sum_{i=1}^n d_i = m$ 이 되는 것이 자연스럽다. 

Length Regulator을 통해 mel 을 만들기 위해 $LR$ (length regulator)모듈을 정의한다면
$H_{mel} = LR(H_{pho} , D , \alpha)$ 로 표현할 수 있는데, 여기서 $\alpha$ 는 voice speed 를 조정하기 위한 Hyper parameter이다.

예를 들어 $H_{pho} = [h_1,h_2,h_3,h_4]$ 이고 각 phoneme의 duration 이 $[2,2,3,1]$ 일 때 , $\alpha$ 는 이 Duration 에 곱해주게 된다. $\alpha$가 2라면 원본 속도보다 두배 느린 voice speed가 될 것이다. 왜냐하면 각 phoneme 의 duration 이 그만큼 길어져 mel-spectrogram도 길어지게 되기 때문이다.

mel-spectrogram의 경우 duration이 $[2,2,3,1]$ 일 때  $[h_1,h_1,h_2,h_2,h_3,h_3,h_3,h_4]$ 로 만들어지게 된다. 각 duration 은 정수형이기 때문에 $\alpha$ 를 곱해준 후 정수형으로 반올림 해주게 된다.

Length Regulator 는 생각보다 간단하게 정의되는데, 이 과정에서 우리는 임의적으로 각 duration 사이에 공간,공백을 넣어주며 운율적인 느낌을 살릴 수 있다. 

### Duration predictor

해당 구조는 앞선 Length Regulator의 Duration을 predict하기 위해 만들어진 모듈이다. 근본적으로 해당 모델은 Duration 에 대한 Truth 값이 없기 때문에 auto regressive 한 Transformer 기반 TTS 모델을 학습을 시킨 후 해당 모델의 학습 결과를 Duration predictor의 Truth 값으로 간주한다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/fastspeech/fastspeech_2.png?raw=true">
  <br>
  Fastspeech Duration Predictor
</p>

우선 구조는 위와 같이 되어있는데, $H_{pho}$ 를 input으로 받은 후 Convolution layer 를 두번 통과한다. 이후 Linear Layer를 통과하여 단일 Scalar 값을 갖게된다. 

이 scalar 값을 phoneme duration 으로 사용하는데, 앞서 말했듯 Auto regressive한 TTS 모델에서 얻은 각 phoneme 의 Duration 을 토대로 MSE Loss를 구하며 학습이 진행되게 된다.

논문에서 제시한 학습 과정은 다음과 같다.

- 기존의 auto regressive TTS 모델을 학습한다.
- 각 모델에서 attention alignment 를 추출하는데 해당 과정은 multi head attention이기 때문에 다중의 alignment 가 나오게 되지만 모든 attention 결과가 diagonal한 특성을 갖고 있지는 않기 때문에(해당 특성은 대충 잘 매칭 되었다고 판단 가능한 가벼운 척도) attention head 가 diagonal 한지 측정하는 척도 F 를 가정하여 해당 값이 가장 큰 attention head 값을 구한다.
- 최종적인 Duration $D$ 를 구하는데, 각 element 인 $d_i$는 앞선 과정에서 선택된 head 에서 mel-spectrogram 에 phoneme 이 각각 얼마나 참여했는지의 최댓값이다. 

## Experimental

LJspeech(2.6GB)의 sample 13100개를 사용하였으며 train으로 12500, validation 300, test 300 개의 sample을 사용하였다.

실험을 위한 Set-up 은 다음과 같다.

- Wave to mel 에서 Frame size 는 1024 이며 hop size 는 256이다. 

- 6개의 FFT block 사용, punctuation 을 포함한 51개의 phoneme vocab, self-attention 과 1D conv 의 384차원  , 2-layer의 Conv 는 kernel size 가 각각 3으로 동일하며 384 to 1536 , 1536 to 384 두 layer 로 구성됨.
- 마지막 output layer 에서 384 dimension이 80 dimension 의 mel-spectrogram 으로 변환되며 Duration predictor 의 경우 Conv layer 가 2개이며 각 Layer 의 channel 은 384로 동일하다.
- Auto-regressive 모델을 사용하였으며, 해당 모델은 6개의 encoder, decoder 로 되어있으며, 해당 모델 또한 FFN 대신 1D convolution 을 적용하였으며, 전반적인 parameter setting 은 Fastspeech 모델과 비슷하다.
- Adam optimizer 를 사용하였으며, Vocoder의 경우 WaveGlow를 사용하였다.

## Conclusion



<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/fastspeech/fastspeech3.png?raw=true">
  <br>
  Fastspeech Result
</p>

Fastspeech의 경우 MOS 지표를 토대로 비교해보았을 때 audio quality 의 경우 tacotron2와 비슷한 성능을 보이는 것을 확인할 수 있다. 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/fastspeech/fastspeech4.png?raw=true">
  <br>
  Fastspeech Result 속도
</p>

Fastspeech의 특징으로 speed 의 경우 tacotron과 비교해보았을 때 매우 빠른 성능을 냄을 확인할 수 있으며 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/fastspeech/fastspeech5.png?raw=true">
  <br>
  Fastspeech Mel, Speed
</p>

Transformer TTS의 경우 mel 의 length 와 합성 속도간의 강한 선형 관계가 있는 것과 별개로 Fastspeech의 경우는 mel의 길이와 합성 간의 선형관계가 약함을 알 수 있다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/fastspeech/fastspeech6.png?raw=true">
  <br>
  Fastspeech Legnth Regulator
</p>

또한 앞서 Length regulator 의 파라미터인 $\alpha$를 조정함에 따라 mel-spectrogram의 길이를 쉽게 조절할 수 있는데, 논문의 저자는 Pitch의 변화가 거의 없이 부드럽게 voice speed 가 조정됨을 확인할 수 있다고 한다. 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/fastspeech/fastspeech7.png?raw=true">
  <br>
  Fastspeech 운율 조정
</p>

또한 phoneme 중간에 간단하게 공백 구간 duration을 적용하여 쉽게 음성의 운율을 추가할 수 있다. 위 그림에서 빨간 박스 친 부분이 Break 를 주어 운율적 요소를 추가한 것이다.

#### 결과

결과적으로 Fastspeech의 경우 아예 auto regressive를 떼어버린 모델은 아니지만 간접적인 autoregressive 를 활용하여 mel-spectrogram 생성 시 non-auto regressive 성질을 띄게 했다고 할 수 있다.

이 과정에서 각 mel spectrogram은 중첩되어 계산되는 것이 아니기 때문에 속도가 빠르다는 장점이 있을뿐만 아니라 duration module 을 직접 튜닝할 수 있기 때문에 부분적으로 합성 결과를 control 할 수 있다. 

하지만 vocoder, transformer TTS를 추가적으로 사용하기 때문에 end to end 모델이라고 할 수는 없다. 하지만 Pararell 하게 Mel 을 Generate 하기 때문에 속도적인 측면에서 이득을 본 것은 확실하다고 말할 수 있다.