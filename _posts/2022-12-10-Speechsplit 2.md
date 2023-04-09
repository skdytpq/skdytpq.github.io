---
title: Speechsplit2
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 딥러닝
key: 20221210
tags: 
  - 딥러닝 
use_math: true

---

# Speech-Split 2

## Introduction

음성인식, 음성 합성 등의 다양한 task 들은 음성에서 뽑아낼 수 있는 다양한 feature 중 특정 feature 에 주목하는 경향이 있다. speech 의 경우 다양한 성분으로 풀어 분석하는 것은 중요하며  Voice conversion 의 경우 주로 timbre(음색)에 집중한다.

음성 변환에 목표로 두는 것은 기존에 주로 탐구되어 Speech를 분해하기 위해 탐구되어져 왔던 linguistic한 특성은 그대로 유지하면서 voice 의 음색을 변경하는 것이다. 

대부분의 voice conversion 의 경우 timbre의 변경에 집중하는데 여기서 대화의 맥락과 음색의 분리가 핵심적인 문제이다.

기존에 VAE를 통한 음색 분리도 시도되었으며, GAN의 경우 bottle neck dimension 을 tuning 하며 여러 특성적 요소를 Speech에서 분리하는 시도를 하였다.

선행 모델인 Speech split 의 경우에도 content, rhythm, pitch, timbre 를 세심하게 조정된 bottle neck 으로 구성되어 있는 3개의 encoder를 사용하여 특성적 요소를 분리하는 시도를 하였는데, 세심하게 튜닝하는 것에 차이가 존재하며 다시 튜닝할 때 마다 다른 데이터를 필요로 한다는 단점이 존재하였다.

Speechsplit 2 의 경우 효율적인 신호처리 기술로 기존 모델의 구조 변경 없이 tuning 과정을 보다 완화시켰다. encoder의 input 을 전처리하여 model 로 들어가는 정보의 흐름을 제어하여 병목 현상으로 인해 감소된 각 구성 요소에 대한 분리된 표현을 학습할 수 있음을 보여주었다.

## Method

### Speechsplit 1

Speechsplit 의 경우 audo-encoder based 생성 모델이다. 해당 모델은 speech 를 4개의 구성요소로 분해하였다.(리듬, 맥락, 피치, 음색)

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/pics/speechsplit2.png?raw=true">
  <br>
  그림
</p>

또한 세개의 인코더로 구성되어 있는데 각 인코더는 리듬을 담당하는 인코더 $E_r$, 맥락을 담당하는 $E_c$, Pitch에 대한 인코딩을 담당하는 $E_f$로 되어있으며,  AE 의 각 Latent Vector $Z$는 다음과 같이 정의된다.
$Z_r = E_r(S), ~ Z_c = E_c(R(S)), ~ Z_f = E_f(R(P))$

여기서의 $S = [s_1, s_2, \dots , s_T]^T$ 인 mel spectrogram의 형태이며 $P = [p_1,p_2,\dots, 	p_T]^T$ 는 각 화자마다의 동일한 평균과 분산으로 정규화 한 pitch의 contour(등고선) 의 one-hot encoding 표현이다. 

또한 R 이란, 전체 시간간 random resampling 에 대한 것이며, $Z_r, Z_c , Z_f$ 는 각 encoder 의 output 이고, decoder D 의 경우 one hot embedding 으로 구성된 벡터 $u$와  이 벡터를 통해 생성하는 output spectrogram $\hat{S}$ 가 나온다.
$\hat{S} = D(Z_r,Z_c,Z_f,u)$ 로 정의할 수 있다.

### Speechsplit 2

처음에 pitch information을 발화 데이터 x 에서 제거한다. 이 과정에서 world라는 보코더를 사용하여 신호를 분석하는 pitch 스무더를 구현한다.

이후 비주기성 스펙트로 envelope를 World 분석기로 추출한다. 그리고 모든 음성 프레임 $f$를 화자 각각의 음성 평균으로 교체하여 신호를 다시 합성시킨다.

여기서 나온 $\hat{x}$를 스펙트로그램 $\hat{S}$를 기반으로 만들어진 pitch 정보가 없는 발화 데이터라고 할 수 있는데, 그 이유는 모든 음성 프레임의 정보를 pitch contour의 평균으로 대체했기 때문에 pitch의 역동성이 없어진 것이다.  이 과정을 PS 라고 한다.

이후 음색적 정보를 배제하기 위해 VTLP를 사용한다. 해당 식은 $\tilde{x} = H(\hat{x}, \alpha)$  로 표현하는데, 해당 $\tilde{x}$ 를 perturb 발화 데이터라고 하며 이것과 대응되는 $\tilde{S}$ 를 perturb mel이라고 한다. 여기서 $\alpha $ 값은 $U(0.9,1.1) $ 을따르는데, 각 훈련 데이터에서 쉽게 음색 정보를 복구하지 못하도록 무작위로 설정하기 위함이다.

pitch 와 음색에 대한 정보를 모두 제거한 $\tilde{S}$ 와 pitch contour 로 생성한 one-hot pitch 표현을 단순히 concat하여 pitch encoder의 input 으로 집어넣는다.