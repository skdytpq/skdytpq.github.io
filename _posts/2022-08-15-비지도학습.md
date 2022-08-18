---
title: 비지도학습
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 머신러닝
key: 20220815
tags: 
  - 머신러닝


---
# 비지도 학습

### DBSCAN

DBSCAN 알고리즘은 K-means 와 다르게 밀집된 연속적인 지역을 클러스터로 정의하게 된다.

이 DBSCAN의 경우에는 K-means 와 다르게 군집을 나누어 각 군집 간의 거리를 계산하는 알고리즘이라기보다 밀도가 높은 부분을 클러스터링 하는 방식이다.

예를 들어 인구가 밀집된 지역을 하나의 군집으로 보아 그 지역을 도시, 그 곳에 포함 안되어있는 지역을 비 도시라고 정의하는 것과 비슷한 맥락으로 알고리즘 상 특정 경계안에 N개의 데이터 포인트가 존재한다면 그 데이터 포인트가 존재하는 곳을 하나의 군집으로 정의하는 것이다.

먼저 DBSCAN의 경우에는 가정을 하는데, 점 P가 있고 그 점으로 부터 $\epsilon$ 만큼 떨어진 거리에 M개 만큼의 데이터 샘플이 있다면 하나의 군집으로 인식하게 된다. 
<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/dbsacan1.png?raw=true">
  <br>
  그림 DBSCAN1
</p>

위 그림에서 원의 반경을 $\epsilon$ 이라고 할 때 우리가 사전의 정의한 M 이 4 라면 점 P은 해당 조건을 만족하였으므로 해당 점은 우리는 core point 라고 부르게 된다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/dbscan2.png?raw=true">
  <br>
  그림 DBSCAN2
</p>



하지만 위 그림에서의 점 P2는 M 이4인 조건을 만족하지 못하였기 때문에 군집의 core point 가 되지는 못한다. 하지만 해당 점의 $\epsilon$ 반경 안에 core point 인 점 P가 속하기 때문에 해당 Sample은 border point (경계점) 이라고 부르게 된다.
<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/dbscan3.png?raw=true">
  <br>
  그림 DBSCAN3
</p>



위 그림에서 보면 알 수 있듯이, Corepoint는 한 군집 안에서 여러 점이 될 수 있다. DBSCAN에서의 한 군집은 위와 같은 방식으로 점진적으로 영역을 넓혀가게 된다.
<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/dbscan4.png?raw=true">
  <br>
  그림 DBSCAN4
</p>


하지만 위 그림처럼 어떠한 군집에 속하지 않고 동떨어진 점이 존재할 수 있는데, 해당 점은 DBSCAN에서는 Noise point 라고 부르게 된다. 

DBSCAN의 경우 Noisepoint를 둠으로써  특이값인 point를 어떤 군집에도 속하지 않게 배제함으로써 전반적인 데이터 분포의 양상을 보다 잘 나타내게 된다.

예를 들어 어떤 데이터 분포가 특정 기하적인 이미지를 띈다고 가정했을 때 (ex : Swiss roll) DBSCAN은 과감하게 애매한 point 는 Noise 로 분류하기 때문에 보다 강건하게 해당 기하적 분포를 잘 나눌 수 있다는 장점이 존재한다.

또한 K-means 와 같이 클러스터의 수를 따로 나누지 않아도 되기 때문에 군집화를 하기에 비교적 수월하다고 할 수 있으며, 한가지 더 특징적인 것은 DBSCAN 자체적으로 새로운 샘플이 들어왔을 때 어느 군집에 속할지 확률을 반환할 수 없다. 

그 이유는 곰곰히 생각해보면 자연스러울 수 있는데, 클러스터 간의 거리를 계산하는 것이 아니라 각 샘플에 대해서 점진적으로 클러스터를 찾는 방식이기 때문에 새로운 클러스터가 들어가게 된다면 전반적인 클러스터링이 달라질 수 있기 때문이다.

따라서 DBSCAN 같은 경우에는 KNN 알고리즘을 따로 훈련시켜 간단하게 해당 데이터 포인트의 확률 값을 확인할 수 있다.\

### GMM

가우시안 혼합 모델 (Gaussian mixture model) 은 샘플이 파라미터가 알려지지 않은 여러개의 혼합된 가우시안 분포에서 생성되었다고 가정한다. 

즉, 2차원 평면에 데이터가 펼쳐져있다고 가정할 때 모든 데이터 샘플은 어떤 가우시안 분포에서 나온 것이며 이 가우시안 분포는 2차원 평면 안에 여러 개가 존재한다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/gmm1.png?raw=true">
  <br>
  그림 GMM의 분포
</p>


위 그림은 1차원에서의 가우시안 분포를 나타낸 것인데, 각각의 분포를 a,b,c 라고할 때 각 데이터 포인트가 어떤 분포에서 나올 확률이 가장 큰지를 추정하는 것이라고 할 수 있으며, 데이터 샘플을 통해 위 분포를 좀 더 고도화 하는 모델이라고 생각하면 된다.

가장 간단한 가우시안 혼합 모델의 가정 방식은 다음과 같다.

- 샘플마다 k개의 클러스터에서 랜덤하게 한 클러스터가 선택된다. j번째 클러스터를 선택할 확률은 클러스터의 가중치 $\phi^{(j)}$ 로 정의되며 i번째 샘플을 위해 선택한 클러스터 인덱스는 $z^{(i)}$ 로 표시한다.
- $z^{(i)} = j$ 이면 즉, i번째 샘플이 j 번째 클러스터에 할당되었다면 이 샘플의 위치 $x^{(i)}$ 는 평균이 $\mu ^{(j)}$ 이고 공분산 행렬이 $\Sigma ^{(j)}$ 인 가우시안 분포에서 랜덤하게 샘플링 된다. 

위에서 풀어 쓴 GMM의 샘플링 방식은 수식으로 정의할 수 있다.

$p(x) = \sum_{(k=1)}^{K} \pi_k 	N ( x\vert u_k , \sum_k)$  

우선 전체 데이터는 K 개의 가우시안 분포가 혼합되어있는 상태에서 샘플링 되었다고 가정하며 각 $\pi $ 는 $\phi$라고 생각하면 되는데, 각 가우시안 분포가 나타날 확률이다. 

해당 수식은 너무 많은 데이터에 샘플 $x$에 대해서 각각 정의하게 되는데 모든 데이터 $x$에 대해서 적절한 $\pi, \mu, \sum$ 을 추정하는 방식이다.

그렇다면 당연하게도 어떤 샘플이 어떤 클러스터에 속할지에 대한 확률은 매우 구하기 쉬울 것이다. 간단하게 베이즈 정리를 이용하여 구할 수 있는데, $r = p(z_{(nk)} = 1 \vert x_n)$ 즉 n 번째 데이터 포인트가 들어왔을 때 클러스터 1에  할당 될 확률을 우리가 구하고 싶다면 샘플이 각 클러스터에 들어갈 확률을 각각 구한 뒤 그 값으로 클러스터 1일 때의 확률을 나누어 주면 된다.

간단한 수식은 아래와 같이 나타낼 수 있다.
<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/gmm2.png?raw=true">
  <br>
  그림 GMM의 베이즈 정리
</p>

가우시안 혼합 모델에서 각 파라미터 $\pi, \mu, \sum$ 을 최대화 하기위해 모델은 EM 알고리즘을 통해 훈련하게 되는데, 해당 과정을 간단하게 이야기하자면, 각 분포의 파라미터를 랜덤하게 초기화 한 후 E 스탭과 M 스탭을 수렴할 때까지 반복하게 된다.

E-step의 경우에는 각 샘플 $x_n$ 을 모든 클러스터에 할당하며 M-step 의 경우엔 데이터 라벨을 활용하여 모든 샘플을 사용해 해당 클러스터의 분포 파라미터를 업데이트 하게 된다.

하나씩 뜯어보자면, 우선 최대값을 구하기 위해 우도 함수를 정의하게 된다.

$L(X;\theta) = \ln p(X\vert \pi,\mu,\Sigma) = ln \{\Pi_{n=1}^N p(x_n \vert \pi, \mu, \Sigma)\}  = \sum_{n=1}^N ln \{ \sum_{k=1}^K \pi_k N(x_n \vert \mu_k , \sum_k) \}$  로 수식이 정의가 되는데,  직관적으로는 데이터 $X$ 가 주어졌을 때  해당 $X$ 를 가장 잘 표현하는 GMM 을 구성하는 것과 동일한 의미이다.

이제 이 식을 각각 편미분 하여 구한다면 아래의 그림과 같은 식으로 도출될 수 있으며, $\pi$ 와 같은 경우에는 0과 1 사이라는 제약이 있기 때문에 라그랑주 승수법을 통해 최대가 되게하는 값을 구하게 된다.
<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/gmmg3.png?raw=true">
  <br>




<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/gmm4.png?raw=true">
  <br>




<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/gmm5.png?raw=true">
  <br>

</p>

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/gmm/gmm6.png?raw=true">
  <br>


</p>


이렇게 위 수식을 최대화 하는 과정을 반복을 통해 구하게 되면 데이터 분포를 결과적으로 가장 잘 설명할 수 있는 가우시안 혼합 모델을 만들 수 있게 된다.


위와 같은 가우시안 모형의 활용은 물론 데이터의 군집화도 가능하겠지만, 해당 모델을 통해 간단하게 이상치를 탐지할 수 있다.

간단하게 해당 모델에서 각 밀도확률이 얼마나 되는지를 구하여 해당 확률이 특정 임계값보다 낮다면 데이터 분포에서 해당 샘플은 이상치로 판단할 수 있는 것이다.

이 과정이 가능한 이유는 단순하게 모델이 군집 간 거리를 나타낸 것이 아닌 어떤 분포에서 나왔음을 가정하고 해당 분포를 샘플링하여 구했기 때문에 각 분포의 확률모형을 앎으로써 진행이 되는 것이다.

이 과정에서 우리가 구한 것은 확률 모형이기 때문에 물론 해당 확률 모형을 통해 데이터 생성도 가능하다. 또한 가장 적절한 클러스터(분포)의 개수를 판단하기 위해 BIC, AIC와 같은 기준을 활용할 수 있다. 

- BIC $ = log(m)p - 2log(\hat{L})$ 
- AIC $ = 2p - 2log(\hat{L})$ 

위 식은 이론적 정보 기준을 통해 정의된 BIC 와 AIC 각 클러스터 개수 마다 EM 알고리즘을 통해 우도함수의 최댓값을 구한 후 도출한다.

