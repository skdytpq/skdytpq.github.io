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

위 그림에서 왼쪽 그림처럼 학습률이 작을 때 포인트의 이동은 느릴 것이다. 결과적으로 최솟값으로 수렴되는 시간이 오래 걸릴 것이다. 오른쪽 그림은 어떤가? 학습률이 매우 크다면 골짜기를 가로질러 반대편으로 건너뛰게 되어 더 큰값으로 발산할 수도 있다. 

우리는 지금까지 2차원 볼록함수의 비용함수 그래프만 보았는데, 모든 비용함수가 이렇게 매끈한 모양이면 좋겠지만, 안그런 경우도 많다. 

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gradient_4.png?raw=true">
  <br>
  그림 3. 다양한 비용함수의 형태
</p>

위 그림에서 만약 알고리즘에서 random으로 배정된 포인트가 $x=0$​ 근처에서 시작한다고 가정해보자. 그렇다면 이 비용함수가 감소하는 방향 즉,  오른쪽으로 움직일탠데, 그림에서 표시된 빨간점에 도달한다면 어떻게 될까?

아마 저 지점에서 비용함수의 기울기가 0이 되기 때문에 저 지점을 전역최솟값(global minimum)으로 인식할 것이다. 하지만 저 지점은 이 비용함수의 지역 최솟값이기에 비용함수가 최솟값을 갖는 올바른 parameter을 찾지 못할 것이다. 

하지만 다행이 선형회귀를 위한 MSE(mean squared error)비용함수는convex하다. 다시말해 전역최솟값을 갖는 볼록함수이다. 또한 연속된 함수이고 기울기가 갑자기 변하지 않는다. 이 두 사실로부터 선형회귀에서는 경사하강법이 전역 최솟값에 가깝게 접근할 수 있다는 것을 보장한다.

## 특성의 스케일링

경사하강법을 통해 최적의 pramter을 찾기 위해 우선 해야하는 작업이 있다. 각 $\theta$​​​​​​의 scale을 맞춰주는 작업이 필요한데, 각 특성의 범위를 맞춰주는 것이다.

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

$\frac{\partial}{\partial\theta_j}MSE(\theta) =\frac{2}{m}\sum_{i=1}^m(\theta^Tx^{(i)}-y^{(i)})x_j^{(i)}$​​​​​​ <br>이 식은 개별 파라미터  $\theta_j$​​​​​​​에 대한비용함수의 편도함수이다. 해석해보자면 관측값의 수는 $m$​​​​​​​이고 $\theta_j$​​​에 대한 미분이므로 다른 $\theta$​​​는 상수처리 되며 각 관측치($i$​​​)의 error 에 파라미터 $\theta_j$​​​번째의 특성 $x_j^{(i)}$​​​​​​​​를 곱한 것이다.

편도함수를 각각 계산하는 대신 $\nabla_\theta MSE(\theta) = \frac{2}{m}X^T(X\theta - y)$​로 MSE 비용함수의 모든 편도함수를 담고있는 gradient vector로 한꺼번에 계산할 수 있다.

다음으로 적절한 $\eta$​​​​​​를 $\frac{\partial}{\partial\theta_j}MSE(\theta)$​​​​​​에 곱해줘서 다음 step에서의 Point인 $x$​​​​​​​값을 이동시키는 것이다. <br> $\theta^{(i+1)} = \theta_i - \eta \nabla_{\theta}MES(\theta_i)$​​​​​로 최종적인 수식을 나타낼 수 있다. 여기서 $\eta$​는 learning rate로 우리가 적절한 학습률을 설정하여 적절한 모델을 찾을 수 있다. 

## 경사하강법 알고리즘 구현

간단한 알고리즘으로 구현해보자면

```python
X = 2*np.random.rand(100,1)
y = 4+3*X + np.random.randn(100,1)
X_b = np.c_[np.ones((100, 1)), X] #X에 intercept 추가 x[0,:]=1
eta = 0.1 #학습률
n_iteration = 1000 #반복횟수
m = 100 #관측값의 수

theta = np.random.randn(2,1) #무작위로 theta를 뽑는 것.

for iteration in range(n_iteration):
	gradients = 2/m*X_b.T.dot(X_b.dot(theta)- y)
	theta = theta - eta*gradients
```

 이렇게 구현 가능하다. 여기서 gradient가 0보다 크다면 다음 theta는 gradient가 더 작아지는 방향으로, 0보다 작다면 다음 theta는 gradient가 커지는 방향으로  $\eta$로 설정한 보폭만큼 이동할 것이다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gradient_7.png?raw=true">
  <br>
  그림 6. 여러 학습률에 대한 경사하강법
</p>

위 그림에서 볼 수 있듯, 반복횟수를 1000으로 설정한 후  $\eta$의 크기에따라서 경사하강법을 적용시킬 때 비용함수를 최소화 하는 적절한 회귀직선을 찾는 경우는 $\eta = 0.1$​인 경우밖에 없다.

왼쪽은 학습률이 너무 낮아 알고리즘이 최적점에 도달하기까지의 충분한 반복을 하지 않았고 오른쪽 같은 경우 학습률이 너무 높아 알고리즘이 이리저리 널뛰며 스텝마다 최적점에서 너무 멀어져 발산한다. 

모델의 적절한 학습률을 찾으려면 그리드 탐색을 수행하면 되는데, 그리드 탐색에서 수렴하는 데 너무 오래 걸리는 모델을 막기 위해 반복 횟수를 제한해야 한다.

그렇다면 적절한 반복 횟수는 어떻게 찾는가? 간단한 해결책은 반복 횟수를 매우 크게 지정한 후 Gradient vector의 절댓값이 아주 작아지면 경사 하강법이 최솟값에 도달한 것이므로 알고리즘을 중지하는 것이다. 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/graddient_10.gif?raw=true">
  <br>
    <a href="https://hackernoon.com/life-is-gradient-descent-880c60ac1be8"> 그림 출처 </a> <br>
  그림 7. 경사하강법의 시각화
</p>



# 확률적 경사하강법

앞서 구현한 알고리즘에서 우리는 각 스텝마다 전체 훈련 세트를 이용해 Gradient를 계산했다. 이러한 경사하강법을 배치 경사하강법이라 하는데, 계산 방식 때문에 훈련 세트가 커지면 자연스럽게 느려지게 될 것이다. 

확률적 경사하강법은 이와 반대로 매 스텝에서 한 개의 샘플을 무작위로 선택하고 그 하나의 샘플에 대한 gradient를 계산한다. 

매 반복에서 다뤄야 할 데이터가 적기 때문에 확실히 알고리즘은 매우 빠르고 메모리 사용량도 상대적으로 매우 적다. 하지만 이름에서 알 수 있듯 확률적 경사하강법이기 때문에 이 알고리즘은 전체 훈련 세트로 계산하는 배치 경사하강법보다는 확실히 불안정하다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gradient_8.png?raw=true">
  <br>
  그림 8. 배치 경사하강법과 확률적 경사하강법의 최적해를 찾는 과정
</p>


위 그림에서 보듯 확률적 경사하강법은 최솟값에 다다를 때까지 요동치며 평균적으로 감소한다. 시간이 지나며 최솟값에 매우 근접하긴 하겠지만 최솟값에 안착하진 못할 것이다. 따라서 알고리즘이 멈출 때 설명력이 꽤 있는 파라미터는 구해지겠지만 최적의 파라미터를 구하진 않는다.

하지만 그림 3과 같이 지역 최솟값(local minimum)이 있는 경우에는 기존 경사하강법보다 전역 최솟값을 찾을 가능성이 높다. 지역 최솟값에 갇혀있는 게 아닌 그 근처를 요동치기 때문에 이러한 불규칙성이 지역 최솟값을 건너뛰는 데 도움을 주는 것이다.

지역 최솟값을 탈출시켜주는 장점이 있지만 전역 최솟값에 도달하지는 못하는 단점이 있는 이 확률적 경사하강법의 딜레마를 해결시켜줄 수 있는 한가지 방법은 학습률을 점진적으로 감소시키는 것이다. 다시말해, 최적해에 가까울수록 널뛰는 폭이 줄어들게 하자는 것이다.

이렇게 매 반복에서 학습률을 결정하는 함수를 학습 스케줄이라고 부른다. 

## 확률적 경사하강법 알고리즘 구현

다음 코드는 학습 스케줄을 적용한 확률적 경사하강법의 간단한 구현이다.

```python
n_epochs = 50
t0,t1 = 5, 50 #하이퍼 파라미터

def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch*m + i)
        theta = theta - eta*gradeints
```

 위 코드에서 우리는 xi,yi 에 random한 index를 부여하고 그 곳에서 gradient를 계산한 후 theta값을 조정해준다. 이 과정은 데이터의 수 m번을 반복하게 되는데 그 과정마다 xi,yi의 값은 또 random하게 배정되며 theta 는 배정된 xi,yi의 gradient를 토대로 업데이트 된다.

일반적으로 한 반복에서 m번 반복되며 이때 각 반복을 에포크(epoch)라고 한다. 배치 경사하강법은 전체 훈련세트에 대해 1000번 반복한다고 한다면 이 코드에서는 훈련세트에서 50번만 반복하게 된다. (보통 데이터가 크다면 이 epoch는 1~10번 사이이다.)

‘’$\theta$가 계속 업데이트되는 과정에서 과연 저 $\theta$​​​​가 최적값을 갖기 위한 방향으로 평균적인 이동이 일어날까?” 라는 의문이 자연스럽게 들 수 있다. 

위 코드에서 보듯 하나의 샘플이 선택되면 우리는 거기서 theta에 대해 적절한 조정한 한다. 그 후  random하게 배정된 xi,yi 에 대해서도 `gradients = 2*xi.T.dot(xi.dot(theta)-yi)` 에서 생각할 수 있듯 이전 스텝을 통해 **조정된** theta를 통해 계산한다. 

즉, 전 스텝에서 비용함수를 적게하는 방향으로 조정된 theta가 다시 random하게 뽑은 xi, yi 를 통해 구한 비용함수도 적게하는 방향으로 조정되는 것이기 때문에 각 스텝에서는 요동칠 수 있지만 결과적으로 훈련샘플의 비용함수를 적게하는 방향으로 조정되는 것이기 때문에 평균적으로 비용함수가 감소하는 방향으로 움직이는 것이다.<br>**최적값이라는 것은 개별 데이터에서의 최적값이 아닌 전체 데이터에서의 모델의 비용함수에 대한 최적값이다.** 난 이 사실이 굉장히 중요하다고 생각한다.



```python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter = 1000,tol = 1e-3, penalty = None,
                      eta0 = 0.1)
sgd_reg.fit(X,y.ravel()) # tol은 손실. penalty 는 규제.
```

알고리즘으로 구현하는 방법 말고도 간단히 위의 코드처럼 사이킷런을 통해 SGD방식으로 선형 회귀를 사용할 수 있다.
