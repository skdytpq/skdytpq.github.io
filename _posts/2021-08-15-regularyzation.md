---
title: 정규화모델
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 기초통계
key: 20210815
tags: 
  -통계학
use_math: true

---

# 좋은 모델은 무엇일까?

* 현재 데이터(training data)를 잘 설명하는 모델 Explanatory modeling
* 미래 데이터(testing data)에 대한 예측 성능이 좋은 모델  Predictive modeling

좋은 모델은 현재 데이터를 잘 설명하며 미래 데이터에 대한 예측 성능이 좋아야 한다.<br>training 현재 데이터를 잘 설명하는 모델은 error를 최소화하는 모델이 좋은 모델이라고 할 수 있다. 

## 편향과 분산에 대한 수식

$MSE_{(training)} = (Y - \hat{Y})^2$ 간단한 수식으로 이렇게 나타낼 수 있는데, 이것에 대한 기댓값을 전개한다면 $Expected MSE = E[(Y-\hat{Y})^2|X]$ 의 형태가 된다. 우선 이식을 보기 전 유도를 위한 분산의 정의와 가정에 대해 살펴보자.
<br>


$Var\left(X\right)$​ 

$=E\left[\left(X-E\left[X\right]\right)^2\right]$​ $=E\left[X^2-2XE\left[X\right]+E\left[X\right]^2\right]$​​

$=E\left[X^2\right]-E\left[X\right]^2$​​​ <br>우선 위식과 같이 분산은 기댓값을 통해 정의될 수 있다.​ 또한 우리는 $Y = f(x)+\epsilon$을 따른다고 가정하는데 여기서 에러  $\epsilon$​ 는 $N(0,\sigma^2)$을 따른다고 가정한다.

따라서 <br>$E[(Y-\hat{Y})^2]$​ = $E[\hat{Y}^2 - 2Y\hat{Y} + Y^2] $​  

=  $E[\hat{Y}^2] - 2E[Y]E[\hat{Y}]+E[{Y^2}]$​​​  가 되며 여기서 sample과 에러 $\epsilon$​​ 의 분포는 독립이므로 

= $Var(\hat{Y}) +(E[\hat{Y}])^2 -2E[Y]E[\hat{Y}] +Var(Y)+E[Y]^2$ (위 분산식 정의에 의해)

= $Var(\hat{Y}) + (Y-E[\hat{Y}])^2 + Var(Y)$​ 	($E[Y] = Y$​​이므로)

= $Variance + Bias^2 + \sigma^2$ ($Y$ 는 $N(Y,\sigma^2)$​을 따른다는 가정에 의해)

위 식에서 알 수 있듯 에러에대한 기댓값은 줄일수 없는 오차와 분산, 편향의 수식으로 이루어져있다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/regular_1.png?raw=true">
  <br>
  그림 1. 분산과 편향
</p>

이 그림은 분산과 편향에 대한 유명한 그림인데, 이 그림에서 알 수 있듯이 가장 좋은 모델은 낮은 분산과 낮은 편향을 갖는 모델이라 할 수 있다. 하지만 안타깝게도 분산과 편향을 모두 잡는 모델은 존재하지 않는다.

편향의 식 $ (Y-E[\hat{Y}])^2$​를 쉽게 해석하기 위해 기존의 데이터 집합을 그림에서의 빨간 점이라고 생각해보자. 그렇다면 좌측 하단의 그림이 의미하는 것은 관측값 전체의 기댓값을 씌운 값이 빨간 점과 많이 벗어나있다는 것이다. 다시말해 관측값 전체를 대표하는 $E[\hat{Y}]$​가 기존 $Y$​​​에 많이 벗어낫다는 의미로 해석할 수 있다. 따라서 편향은 관측값 $\hat{Y}$​​의 분포가 $Y$를 기준으로 쏠려있는지에 대한 정보를 알려준다.

분산은 $E\left[\left(\hat{Y}-E\left[\hat{Y}\right]\right)^2\right]$​​의 식에서 보듯 각 $\hat{Y}$​​가 $\hat{Y}$​​​의 대푯값인 $E\left[\hat{Y}\right]$​​에 얼마나 떨어져있는지에 대한 정보를 알려준다. 따라서 분산이 클수록 우측 하단의 그림처럼 각 데이터가 평균에 많이 떨어져있고 데이터 전체에 대한 평균의 설명력이 떨어진다.​​

이러한 분산과 편향은 Trade-off 관계이기 때문에 두 값을 모두 0으로 만드는 모델은 만들 수 없다.

# 다항 회귀

분산과 편향의 trade-off 관계를 생각해보기 위해 다항 회귀 모델에 대해서 이야기해보자.

우리가 가지고있는 데이터가 단순한 직선보다 복잡한 형태라면 어떨까? 


<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/regular_2.png?raw=true">
  <br>
  그림 2. 데이터가 비선형이다!
</p>

우리는 적절한 치환을 통해 비선형 데이터를 학습하는 데 선형 모델을 사용할 수 있다.<br>$y = w_0 + w_1x + w_2x^2 + w_3x^3$​​의 모델을 생각해보자. 여기서 우리는 $x_d = x^d$라고 $x$​를 치환시켜 $y = w_0 + w_1x_1 + w_2x_2 + w_3x_3$ 와 같은 형태로 $y$에 대한 선형 결합식으로 간주하여 선형 모델을 사용할 수 있다. <br>여기서 선형 결합에 사용된 각 축 $x_d$는 $x^d$​​라고 해석되는 것이다. 

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/regulary_3.png?raw=true">
  <br>
  그림 3. 비선형 데이터에 선형모델 적용
</p>
위 그림은 비선형 데이터에 대해 선형 모델을 적용시킨 것이다. 위 모델은 훈련 데이터에 각 특성을 제곱한 새로운 특성을 추가하여 2개에 특성에 대한 최적값을 계산한 것이다. 즉, 데이터의 분포가 2차임을 가정한 후 데이터에 임의로 1개의 특성을 더 줘 2차원으로 만든 것이다.(사이킷런에서는 각 특성을 제곱하여 새로운 특성으로 추가한다.)

만약 이러한 2차함수꼴의 회귀선보다 더 고차 다항 회귀를 적용한다면 우리는 훈련 데이터에 대해서 더 잘 설명할 수 있을까?

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/scter.gif?raw=true">
  <br>
  애니메이션1. 고차 다항 회귀 적용
</p>
이 애니메이션은 각 300차 다항회귀, 2차 다항회귀, 1차 선형회귀 모델을 훈련데이터에 적용한 것이다. 여기서 보이는 첫번째 300차 다항회귀는 일반화가 잘 돼있다고 할 수 없다. 이 훈련 데이터에서 설명력이 좋다고는 할 수 있겠지만 외부에 들어온 데이터에 2차 다항회귀 보다 결코 예측력이 좋지 않을 것이다. 

다시말해 훈련 데이터에 대해 모델이 데이터에 과대적합 됐다고 이야기 할 수 있다. 이러한 과대적합 여부는 Cross-validation이나 학습 곡선을 살펴 확인할 수 있다.

비유해보자면 피아노를 칠 때 4/4 리듬의 곡만 연습한다면 우리는 다른 4/4 리듬의 곡, 심지어 2/2, 8/8 리듬의 곡도 낯설지만 어렵지 않게 연주할 수 있을 것이다. 하지만 3/4 리듬의 곡을 치라고 하면 우리는 매우 생소할 것이다. 왜냐면 우리는 4/4 리듬의 곡만 익숙하고 그 리듬에 과대적합 되있기 때문이다. 따라서 특화돼있는 패턴 외의 리듬에서 우리의 실력을 100퍼센트 발휘하기란 쉽지 않을 것이다.

# 편향을 희생하자

선형 회귀 모델에서 우리는 최소제곱법을 통해 $\hat{\beta}$​를 구했다. 이 $\hat{\beta}$​의 특징은 비편향 추정량 중 가장 작은 분산을 갖는 다는 것인데 다시말해 오차항이 독립이고 기댓값이 0 일때 최소제곱 추정량 $\hat{\beta}$​는 $Y_i$​의 선형함수로 주어지는 $\beta$​​​의 비 편향 추정량들 중에서 가장 작은 분산을 갖는다. 이 특성은 매우 좋은 특성이다.

 하지만 우리는 여기서 분산에 좀 더 집중하고 편향을 조금 희생해서 분산을 더 줄이는 model을 찾는다고 해보자. 그럼 어떻게 해야할까? 다시 말해 현재 훈련 데이터에 대한 설명력을 조금 희생하더라도 미래 데이터에 대한 예측력을 높이는 좀 더 일반화된 모델을 만들려면 어떻게 해야할까?

앞서 살펴본 과대적합된 모델에 대해서 다시 생각해보자. 과대적합은 왜 일어나는 것인가? 우리는 각 parameter에 아무런 제약을 주지 않았다. 즉 파라미터가 데이터가 주어졌을 때 어떤 값들도 가질 수 있기 때문에 특성이 많아질 때 각 파라미터들은 이리저리 튀는 값을 가질 수 있다. 

예를들어서 $\hat\beta_1 = 1 , \hat\beta_2 = 4, \hat\beta_3 = 20,  \hat\beta_4 = -7 \dots$​​​​​​등 만약 결정해야 하는 파라미터가 많아질수록(모델이 복잡해질수록) 모델을 주어진 데이터에 Fitting해야 하기 때문에 각 $\hat\beta $에 대한$E[\hat\beta]$​의 $\sigma$​(분산)은 점점 커질 것이다. 

이러한 분산을 줄이기 위한 여러 방법이 있을 수 있겠지만 우리는 간단하게 각 $\hat\beta_i$​ 값들이 가질 수 있는 값들의 범위를 미리 정해주는 방법을 생각할 수 있다. 다시말해 모델의 과대적합을 막고 좀 더 일반화 할 수 있는 모델을 만드는 것이다.

 각 $\hat\beta_i$​ 값들이 가질 수 있는 값들의 범위를 미리 정해준다는 것은 우리는 모델의 파라미터의 자유도를 줄인다고 이야기 할 수도 있을 것이다. 다시말해 모델을 규제한다고도 할 수 있는데 다항 회귀 모델을 규제하는 간단한 방법은 다항식의 차수를 제한하는 것이다. 

선형 회귀 모델에서는 보통 모델의 가중치를 제한하면서 규제를 가한다. 정규화(regular-ized) 선형회귀 방법은 선형회귀 계수(weight)에 대한 제약 조건을 추가함으로써 모형이 과도하게 최적화되는 현상, 즉 과최적화를 막는 방법이다. 이러한 가중치를 제한하는 방법에 따라 회귀 식의 이름이 다른데, 우리는 대표적으로 세가지를 살펴보자.

# 정규화 선형회귀

최소제곱법과 정규화의 관점의 차이를 간단하게 다시 이야기하자면 최소제곱법은 $\theta$​가 어떤 값이 되던 error을 최소화 하겠다는 것이고 정규화는 error을 최소화 하겠다는 목적도 갖지만 $\theta$​에 대한 분산도 같이 고려하겠다는 것이다.

## Ridge

ridge 회귀는 간단히 비용함수에 $\alpha\sum_i^n\theta_i^2$ 항이 추가된 선형 회귀이다. 여기서 $\alpha$ 는 하이퍼파라미터로 모델을 얼마나 규제할지에 대한 강도이다. 비용함수의 전체적인 형태를 보자면<br>$J(\theta) = MSE(\theta) + \alpha\frac{1}{2}\sum_i^n \theta_i^2$​ 인데 일반적인 선형 회귀 모델의 비용함수인 $MSE(\theta)$​ 는 오차 제곱 합이며 이 식에 각 $\theta_i$ 의 제곱합을 추가로 줘서 오차 제곱합과 동시에 각 $\theta_i$의 값도 고려하여 최적의 $\theta$​ 값을 찾는다.

여기서 $w$ 를 특성의 가중치 벡터$(\theta_i,\dots, \theta_i)$라고 정의하면 규제항은 $\frac{1}{2}(||w||)^2$라고 할 수 있다. 즉, $w$ 벡터의 $l_2$norm이다. 
이 것은 Ridge의 큰 특징인데, 이후 이야기하는 Lasso 와 비교하여 자세히 이야기 해보겠다.

 $\alpha$ 값은 이러한 규제의 정도인데, 만약 $\alpha$ 값이 매우 크다면 비용 함수를 최소화 하기 위해 회귀 모델의 $\theta$ 의 값들은 점점 작아져 $\alpha$가 계속해서 커진다면 회귀 직선은 일직선에 가까워질 것이다. 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/ridge_2.png?raw=true">
  <br>
  그림 4. alpha에 따른 회귀 직선의 변화
</p>
위 그림에서 보듯이 $\alpha$​​​가 100일 때 각 $\theta$​​​ 는 특정 값을 갖는 것이 제한될 수 있다. 논리적으로 허술할 수 있지만 쉽게 생각해본다면 정의된 비용함수 중 $MSE(\theta)$​​​​ 의 영향력보다 $\alpha\frac{1}{2}\sum_i^n\theta^2$​​의 영향력이 더 강해져서 error의 증가보다  각 $\theta$​​를 매우 작게 즉, 각 $\theta$​​​​가 0에 가깝게 만들어 비용함수를 줄이는 방향이 더 우선순위가 되는 것이라 생각할 수 있다. 그렇기 때문에 $\alpha$​가 100일 때와 $\alpha$​​가 1일 때 더 수평선 형태가 되는 것이다.

## Ridge의 기하학적 의미

만약 우리가 찾아야하는 회귀계수 벡터를  $\theta_0 ,\theta_1$ 라고 둔다면   <br>$MSE(\theta_0,\theta_1)$ = $\sum_i^n (y_i - \theta_0x_{i1}-\theta_1x_{i2})^2$​

<br>$=(\sum_i^nx^2_{i1})\theta_0^2 + (\sum_i^2x_{i2}^2)\theta_1^2 + 2(\sum_i^nx_{i1}x_{i2})\theta_0\theta_1 -2(\sum_i^ny_ix_{i1})\theta_0$

$-2(\sum_i^2y_ix_{i2})\theta_1 + \sum_i^ny_i^2$  로 풀어쓸 수 있는데 

$MSE(\theta_0,\theta_1)$ 은 $A\theta_0^2 + B\theta_0\theta_1 + C\theta_1^2 + D\theta_0 + E\theta_1 + F $​​​​​​ 형태의 **원추곡선**이 된다. <br> 원추곡선은 원뿔을 자르면 나오는 2차곡선의 단면형태라고 쉽게 말할 수 있는데 타원 쌍곡선, 원 포물선은 이러한 형태의 특별한 형태이다. <br> 판별식 $B^2 -4AC$​가 0보다 작다면 이 곡선의 형태는 타원이 되는데 위 식의 판별식을 계산해 본다면 코시-슈바르츠 부등식 조건에 의해 0 이하가 된다고 한다.

또한 Ridge의 규제에서 $\alpha$​​ 값을 조정하는 것은 $\sum_i^n\hat{\theta}^{ridge^2}\leq t^2$​ 의 식에서 t 값을 조정하는 것으로 다시 해석할 수 있는데, 이와 같이 해석하면 우리는   $\theta_0 ,\theta_1$​ 에 대해 $\theta_0^2 + \theta_1^2 = t^2$​​ 원 안의 영역으로 생각할 수 있다. 따라서 우리가 원하는 최적 파라미터는 $\theta_0,\theta_1$​의 파라미터 공간 안에서  $MSE$​​​의 타원과 원이 만나는 지점의 값이라고 생각 할 수 있다.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/manim/circle_ellipse.gif?raw=true">
  <br>
  애니메이션2. MSE와 l2 norm
</p>

위 애니메이션이 정확하진 않지만 간단하게 생각한다면 저렇게 MSE 타원의 영역이 점점 커지고 규제항인 l2 norm은 고정되어 있다. 

즉, Ridge 에서 우리가 엄격하게 생각하는 것은 $\theta$의 값이고 이 목적은 다시 말하자면 $\theta$값들의 범위를 규제하여 $\theta$의 분포의 분산의 증가를 막고 이러한 분산의 감소를 막음으로써 모델의 훈련데이터에 대한 과대적합을 막는 것이다.

선형 회귀와 마찬가지로 릿지 회귀도 정규방정식을 통해 계산할 수 있으며 또한 경사 하강법을 사용할 수 있다. 

경사하강법으로 간단히 계산한다면 아래 코드처럼 penalty 에 l2 norm 규제를 주면 된다.

```python
SGDRegressor(penalty = 'l2')
```

원리는 똑같이 비용함수 공간에서 $\theta$​값을 찾아나가는 것이라고 생각하면 된다. 

## Lasso

Lasso 회귀도 선형 회귀의 또 다른 규제 버전이다. 릿지 회귀와 다른점은 비용함수에 l1 norm을 사용한다는 것인데, 식으로 나타내면 $J(\theta) = MSE(\theta) + \alpha\sum_i^n|\theta_i|$이다. 
즉, 절대값을 씌운 것이다.

이 라쏘 회귀의 중요한 특징은 덜 중요한 특성의 가중치를 제거하려 한다는 것이다. 하지만 Lasso는 안타깝게도 $\theta_i = 0$​ 인 지점에서 미분이 불가능하기 때문에 최소값을 직접 찾는 것은 불가능 하기 때문에 수치 최적화를 통해 구해야 한다.​ 하지만 $\theta_i = 0$일 때 subgradient vector 을 사용하면 경사 하강법을 적용하는 데 문제가 되지 않는다.

$sign(\theta_i) = \begin{cases}-1 \;\; \theta_i<0 \newline 0 \;\;\theta_i=0\\ \newline 1 \;\;\theta_i >0  \end{cases}$ 의 식을 
$g(\theta,J) = \nabla_{\theta}MSE(\theta) + \alpha(sign(\theta_1),\dots,sign(\theta_n))$​​ 
의 식에 사용하면 모든 경우에서 함수값을 갖기 때문에 ​​무리 없이 경사 하강법을 적용하여 Lasso에서의 $\theta$를 찾을 수 있다.

## Lasso 의 기하학적 의미

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/manim/Lasso_ellipse.gif?raw=true">
  <br>
  애니메이션3. MSE와 l1 norm
</p>

위 애니메이션을 보면 Lasso 에서의 규제가 마름모 꼴의 형태가 됨을 확인할 수 있다. 이러한 마름모의 제약범위 내에서 $MSE$가 최소가 되는 접점이 $\theta_0 = 0$​ 이 되는 점으로 표현 했는데 이 점을 본다면 Lasso 의 특징인 변수의 선택 즉, $\theta_0$에 대응되는 독립변수 $x_0$​이 예측에 중요하지 않다는 말과 같다.  

다행이게도 이러한 변수 선택은 data가 달라질 때와 관계 없이 거의 비슷하게 이루어진다. Lasso 에서는 $\alpha$의 계수 조절로 변수 선택을 얼마나 까다롭게 할 지 정할 수 있다.

아래 그림에서 더 쉽게 확인할 수 있는데 l1,l2 규제를 적용시킨 비용함수에 경사하강법을 적용하여 최적의 파라미터를 찾아내는 것이다.  이 그림에서 볼 수 있듯이 l1 규제를 진행할 때 규제가 강할수록 즉, $\alpha$​가 늘수록 $\theta_2$​의 값은 0이 된다. 즉 선택되지 않는 것이다. 이 등고선은 각 비용함수를 나타내는데 빨간 사각형은 임의의 $\alpha$​에서 전역 최솟값에 도달할 때의 $\theta$​​이다. 

이 $\alpha$​값이 증가하면 선택할 수 있는 $\theta$의 영역은 좁아지기에 빨간 사각형은 왼쪽으로 점점 이동 할 것이다. 또한 앞서 살펴봤듯이 Lasso 는 $\theta_i = 0$일 때 미분 가능하지 않기에 $\theta_1 =0$ 으로 도달할 때 Ridge 에서와 달리 진동이 조금 있다. 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/lasso_ridge.png?raw=true">
  <br>
   그림 5. Ridge와 Lasso 의 경사하강법 적용 시 theta의 변화
    </p>

