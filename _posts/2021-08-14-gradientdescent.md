---
title: 경사하강법
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 기초통계
key: 20210814
tags: 
  -통계학
use_math: true
---

# Gradient Descent의 목적

gradient descent는 기본적으로 함수의 최솟값을 찾는 것이 목적이다. 

“”선형회귀에서 정규방정식을 만족하는 해를 찾는 것처럼 해석적인 방법을 이용하면 되지 않겠느냐?” 라는 물음은 자연스러운 물음이다.

* 하지만 데이터 분석에서 맞딱드리는 함수는 보통 미분계수를 계산하기가 어렵다. 

* 또한 앞서 선형회귀 포스팅에서 말했던 것처럼 계산복잡도 또한 실제 데이터에서 분석을 진행할 때 해석적인 근을 찾는 경우 더 커진다
* 추가적으로 데이터의 양이 매우 큰 경우 Gradient Descent와 같이 iterative한 방법으로 해를 구하면 계산 측면에서 더 효율적이다.

그렇기 때문에 보통 함수의 최솟값을 찾을 때 경사하강법과 같은 iterative한 방법을 많이 쓴다.

# 경사하강법?

## 경사하강법과 learning rate

경사하강법은 임의의 $\theta$값을 배정한 후에 한번에 조금씩 비용함수가 감소되는 방향으로 진행하는데, 비용함수에서 Random하게 배정한 그 point 에서의 기울기를 계산한다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gradient_1.png?raw=true">
  <br>
  그림 1. 경사하강법 모델의 과정
</p>

위 그림을 보면 직관적으로 이해할 수 있을 것이다. 여기서 빨간점은 Random하게 배정된 point이고 거기서의 기울기를 구한뒤, 그 기울기가 음수인 곳 즉, $y=x^2$의 최솟값을 갖게하는 $x$값으로 이동하는 것이다.

이 경사하강법의 주요한 Parameter은 스텝의 크기로, **학습률**이라는 하이퍼파라미터로 결정된다. 이 학습률이 너무 작으면 최솟값을 찾으려하는 과정 반복이 많아지므로 시간이 오래걸린다. 

쉽게말해 한발짝을 내딛을 때 보폭을 넓게 할 것인지, 좁게 할 것인지 선택하는 것이다.

만약 학습률이 크다면 어떻게 될까?

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gradient_3.png?raw=true">
  <br>
  그림 2. 학습률에 따른 진행과정
</p>

위 그림에서 왼쪽 그림은 학습률이 작을 때 포인트의 이동은 느릴 것이다. 오른쪽 그림은 어떤가? 학습률이 매우 크다면 골짜기를 가로질러 반대편으로 건너뛰게 되어 더 큰값으로 발산할 수도 있다. 

우리는 지금까지 2차원 볼록함수의 비용함수 그래프만 보았는데, 모든 비용함수가 이렇게 매끈한 모양이면 좋겠지만, 안그런 경우도 많다. 

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gradient_4.png?raw=true">
  <br>
  그림 3. 다양한 비용함수의 형태
</p>

위 그림에서 만약 알고리즘에서 random으로 배정된 포인트가 0근처에서 시작한다고 가정해보자. 그렇다면 이 비용함수가 감소하는 방향 즉, 양의 방향으로 움직일탠데, 그림에서 표시된 빨간점에 도달한다면 어떻게 인식할까? 

아마 저 지점에서 비용함수의 기울기가 0이 되기 때문에 저 지점을 전역최솟값(global minimum)으로 인식할 것이다. 하지만 저 지점은 이 비용함수의 지역 최솟값이기에 비용함수가 최솟값을 갖는 올바른 parameter을 찾지 못할 것이다. 

하지만 다행이 선형회귀를 위한 MSE(mean squared error)비용함수는 prameter $\theta$​에 대한 비용함수를 정의하기에 convex하다. 다시말해 전역최솟값을 갖는 볼록함수이다. 또한 연속된 함수이고 기울기가 갑자기 변하지 않는다. 이 두 사실로부터 경사하상법이 전역 최솟값에 가깝게 접근할 수 있다는 것을 보장한다.

## 특성의 스케일링

경사하강법을 통해 최적의 pramter을 찾기 위해 우선 해야하는 작업이 있다. 각 $\theta$​​에 대한 정규화(regularyzation)이 필요한데, 각 특성 스케일을 맞춰줘야 하는 것이다.

그 이유는 무엇일까? 

위 그림과 같이 특성 $\theta_1$과 $\theta_2$​의 범위가 다르다고 할 때, $\theta_1$은 scale이 $\theta_2$보다 크다. 따라서 비용함수를 최소화 하기 위해 Point는 $\theta_1$의 방향으로 더 가야한다. 쉽게말해 Point 가 전역최솟값을 향해 가야하는 거리가 $\theta_1$​의 방향으로 더 늘어져있는 것이다.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gradient_5.png?raw=true">
  <br>
  그림 4. 특성 스케일을 적용한 경사하강법(왼쪽)과 적용하지 않은 경사하강법(오른쪽)
</p>

여기서 왼쪽그림과 같이 두 $\theta$​의 범위를 같게 해준다면 그림에서 보듯 Point가 전역 최솟값을 향해 가야하는 거리가 오른쪽 그림보다 짧아진다. 즉, 범위를 맞춰 직사각형의 3차원 공간에서 정사각형의 3차원 공간으로 비용함수 공간을 재구성하는 것이다. 

## paramter는 두개 이상일 수 있다!

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gradient_6.png?raw=true">
  <br>
  그림 5. 특성 스케일을 적용한 MSE 비용함수의 3차원 비용함수 공간
</p>

앞의 그림 4는 모델 훈련이 비용함수를 최소화하는 모델 파라미터의 조합을 찾는 일임을 설명해준다. 이를 모델의 파라미터 공간에서 찾는다고 한다. 

위 그림 5는 특성 스케일을 적용한 MSE 비용함수의 3차원 공간을 시각화한 것이다. 우리는 지금 특성이 2개인 3차원 공간에서 생각했기 때문에 직관적으로 해석하는 게 가능했지만 실제 파라미터가 더 많아진다면 차원이 3개일 때보다 최솟값을 위한 prameter조합을 찾는 것은 훨씬 복잡할 것이다. 다행히 선형회귀의 경우 비용함수는 Convex하므로 고차원이더라도 그 조합은 맨 바닥에 있을것이다.

# 경사하강법의 수식

우리는 여태껏 경사하강법의 목적과 용이성, 기본 개념에대해 알아봤다. 이제 MSE를 토대로 하여 경사하강법의 수식을 유도해보자.

우선 비용함수의 기울기를 알기 위해 우리는 편도함수를 구해야 한다.

$\frac{\partial}{\partial\theta}MSE(\theta) =\frac{2}{m}\sum_{i=1}^m(\theta^Tx^{(i)}-y^{(i)})x_j^{(i)}$ <br>이 식은 파라미터  $\theta_j$​에 대한비용함수의 편도함수이다. 해석해보자면 관측값의 수는 $m$​이고 각 파라미터에 대해서 각 관측치의 error에 파라미터 $\theta_j$​번째의 관측값 $x_j^{(i)}$를 곱한 것이다.

