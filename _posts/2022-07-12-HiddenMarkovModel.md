---
title: Hidden Markov Model
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 딥러닝
key: 20220712
tags: 
  -통계학
  -딥러닝
use_math: true
---

# Hidden Markov Model

순차 데이터란 시간적 특성이 있는 데이터라고 할 수 있다. 예를 들어 어떤 염기서열, 날씨, 음성 등등이 이러한 Sequence 데이터라고 할 수 있다.

HMM 은 순차 데이터를 확률적으로 모델링하는 생성 모델이라고 할 수 있다 (Generative Model) . 

## Markov Model

State로 이루어진 Sequence를 상태 전이 확률 행렬로 표현하는 것이 Markov Model 의 정의라고 할 수 있는데

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/HMM/Markov_%EC%83%81%ED%83%9C%EC%A0%84%EC%9D%B4%ED%96%89%EB%A0%AC.png?raw=true">
  <br>
  그림 1. 상태 연속에 대한 행렬
</p>

그림과 같이 어떠한 상태가 연속적으로 되어있는 확률을 행렬로 표현하는 것이라고 생각하면 된다.

Markov 가정은 시간 $t$에서 관측은 가장 최근 $r$개의 관측에만 의존한다는 가정이며, 이러한 과정은 시퀀스 데이터의 너무 과거 사건은 영향력을 무시하고 이전 시점을 강력히 생각하자는 것이다.

### Hidden Markov

같은 시간에 발생한 두 종류의 State sequence 각각의 특성과 그들의 관계를 이용해 모델링을 하는 것이다.

숨겨진 시퀀스와 관측 가능한 시퀀스가 있는데, 관측 가능한 시퀀스 데이터로 관측 불가능한 시퀀스 데이터를 예측하겠다는 것이다.

이러한 관측 가능한 시퀀스는 Hidden Sequence에 종속되어있다. 

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/HMM/Markov_%EA%B7%B8%EB%A6%BC.png?raw=true">
  <br>
  그림 2. Hidden state의 종속
</p>

종속되어 있다는 말은 위 그림과 같이 관측 가능한 State($o_n$)는 Hidden State($s_n$) 의 영향을 받고 나온다는 소리이다.

음성인식에서 우리가 음성 데이터를 관측할 수 있지만, 이러한 음성 데이터에서 음소&단어 데이터는 관측할 수 없다. 여기서 음소&단어가 Hidden State 라고 할 수 있다.

이러한 HMM의 파라미터는 상태전이확률 행렬과 방출확률 행렬이다. 전이 확률이란, 이전 시점의 Hidden State 에서 현 시점의 Hidden State 로 옮겨질 확률이다.  방출 확률이란 각각의 상태에서 Observed된 데이터가 나올 확률을 이야기하는 확률이다.

또한 $\pi$ 도 Parameter 라고 할 수 있는데 이 것은 Hidden State 의 초기 확률이라고 할 수 있다.

- 전이확률 $a_{ij} = p(q_{t+1} = s_j \vert q_t = s_i), 1 \leq i, j \leq n$ $\sum_{j=1}^n a_{ij} = 1$
- 방출확률 $b_j(v_k) = P(o_t = v_k \vert q_t = s_j), 1\leq j \leq n , 1\leq k \leq m$ 이며 은닉상태 $b_j$ 에서 관측치$v_k$ 가 도출될 확률이다.
- $\pi$ 는 각 $s_i$ 에서 시작할 확률

위와 같은 식으로 각각의 확률 index 로 정의할 수 있다. (모든 parameter 은 행렬로 정의된다.)

### HMM 의 문제

우리는 히든 마르코프 모델을 어떠한 관측 데이터로 학습을 시킨다고 할 때, 여기서 은닉 상태의 확률값 즉, 그 상태를 예측하는 Decoding 을 시행할 수도 있고 Observed 데이터가 들어왔을 때 그 데이터의 확률을 예측하는 문제를 풀 수 있다.

하지만 이러한 확률이 어떻게 되는지에 대해서 우리는 알 수 있는가?

우선, Observed 데이터의 시퀀스를 우리가 알아낼 수 있는가? 데이터 시퀀스 자체가 만약 길어진다면 이러한 경우의 수는 급수적으로 많아진다. 

예를 들어 3가지의 Hidden State 값이 있는 경우  2 개의 시퀀스를 가진 O 의 확률을 알고싶을 때 이 확률은 9가지의 경우의 수를 모두 더한 경우가 된다. 

만약 10개의 시퀀스의 확률을 알고싶다면? 이 경우는 $3^{10}$ 의 경우의 수를 모두 더해주어야 하기 때문에 시퀀스가 길고 Hidden State 의 수가 많다면 매우 연산량이 많은 작업이 될 수 있다.

총 상태의 개수 가 $N$ 이라고 하고, Sequence 의 길이가 $T$ 라고 할 때 총 경우의 수는 $N^T$ 라고 할수있다. 

이러한 문제는 우리가 풀 수가 없다. 그래서 조금 더 Smart 한 방법을 통해 풀겠다는 것이 Forward Algorithm 이다.

### Forward Algorithm

예를 들어 어떠한 Sequence $O = (o_1 = a , o_2 = b , o_3 = c )=?$ 의 확률을 알고싶다고 할 때  Hidden state 가 2개 라면 우리는 각 계산 확률을 저장해 두었다가 중복된 계산은 꺼내 쓸 수 있게 만들 수 있다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/HMM/Markov_Forward.png?raw=true">
  <br>
  그림 3. HMM Forward
    </p>

그림에서 $q1$ 과 $q2$ 는 각각의 Hidden state 상태이다. $O_i$ 는 각각 상태가 나올 확률이라고 본다면, 우리가 지정한 초기 상태 $\pi$ 에서 우선 초기 State가 분기된다.

여기서 Hidden state 상태에만 집중한다면, 첫번째 $a_1(2)$ 의 확률은 $\pi \times b(o_1)$ 이 된다. 

이후 해당 결과를 여기서 저장하여 이후 시점 계산에 활용하게 되는데, 예를 들어서 $a_2(2) = (a_1(2)a_{22} + a_1(1)a_{12})b(o2) $ 이다. 여기서 $a_{ij}$는 i 번째 hidden state 에서 j 번째 hidden state 로 전이될 전이확률이다.

이 Hidden Markov model 을 살펴보면, 어떤 특정 시점 $t$에 대한 정보가 $r$시점 전까지의 정보만을 참고하는 흐름이라고 하지만 모든 시점은 이전 시점에 영향을 받기 때문에 $t-1$시점의 정보는 그 이전까지의 모든 과거 정보가 담겨져 있다고 할 수 있다.

이러한 과정을 Forward algorithm 이라고 할 수 있는데, 어떠한 계산을 새롭게 하는 것이 아니라 중복된 것을 메모리에 저장해두었다가 필요할 때 꺼내서 계산하는 식의 알고리즘이라고 할 수 있다. 다시 말해 Loop 를 도는 반복적인 계산을 중복해서 수행하는 것이 아닌, 계산들을 메모리에 저장해두어 필요할 때마다 꺼내 쓴다는 Idea이다. 

단적인 예로 $a_2(2)$ 에 대해서 계산하고자 할 때 이전 노드의 정보를 참고할 수 있다. 이렇게 된다면 중첩된 계산을 피할 수 있어 효율성이 높아진다.

여기서 각 $a$노드는 전방확률(forward probability) 라고 한다.

해당 과정을 통해 우리가 얻을 수 있는 통찰은 우리가 HMM 에서 각 전이확률과 방출 확률을 안다면, 어떤 결과값 $K$가 나왔을 때 무수히 많은 HMM 모델을 대조하여 $K$가 나옴직한 모델을 선택할 수 있게 된다는 것이다.

즉, Output 데이터가 주어졌을 때 어떤 모델을 신뢰하는 것이 바람직한지 알려주는 것이다. 하지만 이 과정에서의 전재는 우리가 Hidden state 의 상태와 각 State 에서의 방출 확률을 알고 있다고 가정한다.

**Forward probability**

- $p(O \vert \lambda) = \sum_{j=1}^n \alpha_T(j)$  (모든 노드에서 j 가 나올 확률)
- $\alpha_1(i) = \pi_ib_i(o_1), 1\leq i \leq n$ (초기 노드의 값)
- $\alpha_t(i) = [\sum_{j=1}^n \alpha_{t-1}(j)\alpha_{ji} \times b_i(o_t)], 2\leq i\leq T , 1\leq i \leq n$ (이전 노드의 확률을 모두 포함)
-  Forward Probability 는 주어진 Sequence O 가 HMM에 속할 확률이다.
- 이러한 idea 는 Sequence classification 에서 활용 가능하다. ( 어떤 Sequence 가 어느 HMM 모델에서 나왔는지 판단 가능)

### Backward probability

Backward probability 란 전방확률을 계산하는 것이 아닌, 후방 확률을 계산하는 것이라고 할 수 있는데, 똑같은 문제에 대해서 (Sequence 확률 판별 문제) 뒤에서 앞으로 계산을 진행하면 된다. 

방향만 다른 것이기 때문에 Forward Prob 와 계산 과정은 다르지만 결과 값은 같게 나올 수 밖에 없다.

### Hidden Markov Model - Decoder

앞선 Forward Prob로 어떤 시퀀스가 해당 HMM 에서 나올 확률을 찾는 문제와 별개로 모델 $\lambda$ 와 시퀀스 $O$ 가 주어졌을 때 가장 확률이 높은 은닉상태의 시퀀스$Q$ 를 찾는 것이 디코딩이다.  

이 것은 HMM 의 핵심이라고 할 수 있는데 예를 들어 음성인식 문제에서는 입력 음성 신호의 연쇄를 가지고 음소(단어) 시퀀스를 찾는 문제에 적용할 수 있다. 

이 Decoding 과정에서는 비터비(Viterbi Algorithm) 이 주로 사용된다.

$v_t(j) = max_i^n[v_{t-1}(i) \times a_{ij} \times b_j(o_t)]$ 의 수식으로 정의되는데, $t$ 시점의 $j$번째 은닉 상태의 비터비 확률을 가리키는 수식이다. 

해당 과정의 식을 살펴본다면 $a_{ij}$ 는 i 번째 노드에서 j 번째 노드로 가는 확률을 의미한다. 이후 $v_{t-1}$ 은 이전 i 번째 노드의 확률이며, $b_j(o_t)$는 해당 노드에서 어떤 특정 시퀀스 $o_t$가 나올 확률이라고 할 수 있다.

이 과정에서 우리가 찾고자 하는 것은 저 비터비 확률을 최대화 하는 i 번째 인덱스를 찾는 것이라고 할 수 있는데, 다시말해 이전 $t-1$시점 노드에서 어떤 Hidden state 노드와 연결되었을 때 확률이 가장 높은지 찾는 것이라고 할 수 있다.

확인해보면 비터비 확률을 구할 때에도 Forward Algorithm 이 적용된 것을 확인할 수 있다.

그렇게 나온다면 비터비 확률을 통한 모델에서는 특정 j 번째 노드에서는 가장 확률이 높을 i 번째 인덱스를 찾고, Max 값을 출력하게 된다. 그렇다면, 해당 $t$시점의 Hidden state 의 j 번째 노드에서의 최댓값을 찾을 수 있으며 여기서 가장 Max 값이 Hidden state 의 노드를 찾게 되면 그 것을 해당 시퀀스가 나타났을 때의 Hidden state 라고 할 수 있다.

$\tau_t(i) = ^{argmax}_{1\leq \leq n}[v_{t-1}(j)a_{ji}]$ 라고 수식적인 표현이 가능하다.

Forward 알고리즘과의 차이는 Sum 과 Select 의 차이이다.

### Parameter Learning

하지만 위와 같은 추적을 하기 위해서 우리가 알아야 하는 것은 결국에 각각의 $b,a$ 의 확률들이다. 

지금까지는 이러한 확률 즉 전이확률과 방출 확률을 임의로 지정하고 해당 과정을 거쳐서 어떻게 Decoding 과 Evaluation 이 이루어지는지 알아보았지만 결국에 이러한 HMM 모델을 만들기 위해서 가장 우선시 되어야 하는 것은 각각의 확률이다.

즉 다시말해 우리는 가장 먼저 $HMM(\lambda^*) = argmax_{\lambda} P(O\vert \lambda)$ 를 찾아야 한다.

HMM 을 Learning 시키기 위해 우리는 적절한 $A,B,\pi$  행렬을 찾아야 하는데, 전체적인 파라미터는

1. HMM 초기화
2. 적절한 방법으로 $P(HMM(\lambda^{new})) > P(O\vert HMM(\lambda))$ 찾기
3. $\hat\lambda = HMM(\lambda^{new})$ 로 설정하고 중지하거나 2 반복

으로 학습이 진행된다.

이 Learning 방법은 Baum-Welch Algorithm 으로 진행된다. 이 것은 EM 알고리즘과 비슷하다고 할 수 있다.

학습 때 사용하는 파라미터는 $\gamma , \xi $ 이다.

- $\gamma $ 는 $t$ 시점에서 $s_i$ 에 있는 것이고 $\xi$ 는 $t$시점 상태가 $s_i$, $t+1$ 시점에서는 $s_j$ 일 확률이다.

### 확률구하기

$\gamma_t(i) = p(q_t = s_i \vert O, \lambda) = \frac{\alpha_t(i)\beta_t(i)}{\sum_{j=1}^n\alpha_t(j)\beta_t(j)}$ 라는 수식으로 나타낼 수 있다. 위 수식이 의미하는 바는 직관적으로 모든 State 가 나타날 확률을 구하고 $s_i$ 가 나타날 확률을 구한 것이라고 할 수 있다.

자세히 들여다 보면 Forward , Back ward 계산 식을 곱해주는 형태로 나타낼 수 있는데, 두 식을 곱해줌으로써 해당 시점에서 어떤 State 의 확률을 계산할 수 있다.

$\xi_t(i,j) $ 는 위 감마 식과 비슷하지만 분모와 분자 사이에 $a_{ij}b_j(o_{t+1})$ 을 곱해준다. 이 것의 의미는 $i$ 번째 노드에서 $j$번째 노드로 갈 전이 확률과 $j$번 째 노드에서 상태가 나타날 확률을 고려한 것이라고 할 수 있다.

HMM에서 Leaning 을 시키기 위해 우리는 $\alpha,\beta$ 를 계산하여 감마와 크사이를 우선 먼저 구한다.

이후 구한 값으로 파라미터를 업데이트 하여 확률을 높이는 방향으로 학습이 진행된다.

이 감마와 크사이는 3 개의 파라미터에 어떤 영향을 주는 것일까?

- $\pi_i^{new} $ 는 t가 1일 때 $s_i$에 있을 확률 즉, $\gamma_{t=1}(i)$ 라고 표현할 수 있다.

- $a_{ij}^{new} = \frac{s_i 에서 s_j로 전이할 기댓값}{s_i 에서 전이할 기댓값}$ 이라고 나타낼 수 있다. 위 의미를 간단하게 보면 매우 자연스럽다. 우선 노드 i 에서 j 로 전이 할 확률을 알기 위해 우리는 $P(s_j \vert s_i)$ 를 구하면 되는데, $s_i$ 에 대한 확률을 우선적으로 구해야 하기 때문이다. 

  이러한 연산은 모든 시점을 통틀어서 진행해야 하기 때문에 앞선 감마와 크사이의 정의를 다시 돌이켜 본다면 

  $\frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}$ 라는 수식을 통해 우리는 모든 $a$ 를 구할 수 있다.

- $b_i(v_k)^{new}$ 는 $s_i$ 에서 $v_k$ 를 관측할 확률 즉, 방출 확률을 새롭게 업데이트 하는 수식이라고 할 수 있다.

  해당 수식에서 우리는 또한 모든 시점에서 State 가 $s_i$ 일 확률을 우선적으로 구한 후 $o_t = v_t$ 인 모든 $t$ 에 대해 $s_i$에 있을 확률을 구하게 되면 해당 방출 확률을 구할 수 있다.

  수식적으로 $\frac{\sum_{t=1,so_t = v_t}^{T-1}\gamma_t(i)}{\sum_{t=1}^{T-1}\gamma_t(i)}$ 이라고 표현할 수 있다.

결국 우리는 Parameter 3 개를 모두 감마와 크사이를 통해 표현할 수 있었다. 

### PIPELINE

우리의 HMM 의 모델의 인풋은 어떠한 임의의 모델이다. 

이후 우리는 HMM 에 임의로 파라미터를 넣어 해당 모델의 감마와 크사이를 구하게 된다. 이후 우리의 Observed Data 를 집어넣어 해당 모델의 Prob를 구할 수 있게 되는데, 이 Prob 를 비교함으로써 파라미터를 Update 시키게 되는 것이다. 

## Hidden Markov Model의 의의

한 상태(state) $q_i$ 가 나타날 확률은 단지 그 이전 상태 $q_{i-1}$ 에만 의존한다는 것이 Markov 성질의 핵심이다.

$P(q_i \vert q_1, \dots , q_{i-1}) = P(q_i \vert q_{i-1})$ 과 같은 수식으로 표현된다. 

$P(q_{i+1} = x \vert q_i = y) = P(q_2 = x \vert q_1 =y) = P_{xy}$ 보통의 마코프 체인에서는 모델링을 간소화하기 위해 전이 확률 값이 전이 시점에 관계 없이 상태에만 의존한다고 가정한다. (시간안정성 과정)

은닉마르코프 모델은, 각 상태가 마르코프 체인을 따르지만, 은닉상태 인 것이라고 가정한다. 

즉, 다시말해 현재 보유한 데이터를 실제 어떤 은닉 상태(true hidden state )가 Noise가 낀 형태로 표현된 것이라고 생각하는 것이다.

예를 들어 $B$를 예측하고 싶을 때 $A$의 정보만 있다면 $P(A \vert B)$ 즉, $B$일 때 $A$의 확률을 최적화 하여 $B$ 를 구하자는 것이다

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/HMM/Markov_1.png?raw=true">
  <br>
  그림 4. HMM
</p>

해당 그림에서 알 수 있듯이, 우리가 알고싶은 확률을 $A$ 즉, 날씨가 더웠는지 추웠는지에 대해 알고싶다고 할 때 해당 날씨에서 아이스크림을 먹을 횟수$B$ 를 가지고 추론하자는 것이다.

해당 Chain 에서 $A$ 는 전이 확률(transition probability) 이며, $B$ 는 방출확률이 된다. 전이 확률이란, 우리 눈에 보이지 않는 Hidden State 에서 각 상태로 전이할 확률을 나타내며, 각 $B$ 는 해당 상태에서 특정한 확률값을 방출(emission)한다고 해석할 수 있다.

여기서 $\pi$ 란 초기 상태 분포를 가르킨다. 우리가 가정한 $A$ 는 2가지 상태에 대한 확률이기 때문에 [0.8,0.2] 등으로 표현될 수 있다.

이러한 은닉 마르코프 모델에서 알고싶은 것은 이러한 $A,B$ 에 대한 확률이라고 할 수 있는데, 현재 알고있는 데이터로 부터 추청하는 과정이라고 할 수 있다.


