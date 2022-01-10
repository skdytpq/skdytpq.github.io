---
title: 퍼셉트론
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 딥러닝
key: 20210106
tags: 
  - 딥러닝
use_math: true

---


# 들어가기 앞서,

내용이 조금 길 수 있다.

우리는 머신러닝에서 분류에 적용 가능한 회귀식 중 로지스틱 회귀라는 것을 알아봤다. 로지스틱 회귀에서 우리가 초점을 맞춘 것은  $\pi_i = \beta_0 + \beta_1X_i$​ 에서 확률로 0과 1 사이의 범위 값을 갖는 $\pi_i$를 출력하는 것이었다. 

여기서 우리는 회귀식의 범위를 알맞게 조정하기 위해 Odds(승산)을 살펴 보았고 범위를 조정하여 output을 확률로 바라보게 되었다. 

우리는 Odds에 log를 취한 후 역함수를 구해 input 식의 범위를 조정하였는데, 여기서 우리는 $\beta_0 + \beta_1X_i$ 에 변형을 가한 함수를 sigmoid 함수라고 이야기하며 이 변형에는 다양한 함수를 사용 할 수 있다고 하였다.

 딥 러닝에서 이렇게선형 결합의 식을 sigmoid 함수와 같이 비 선형의 식으로 바꾸는 함수를 Activation-function이라 칭한다. 

차후 살펴보겠지만 이후 나오는 신경망에서 	

$\pi(X=x) = \frac{1}{1+e^(-(\beta_0 + \beta_1X))}$  와 같은  sigmoid 함수 꼴과 비슷한 형식은 자주 나오게된다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/logistic_4.png?raw=true">
  <br>
  그림 1.로지스틱함수에서 sigmoid 함수 사용
</p>


# 퍼셉트론(Perceptron)

퍼셉트론(Perceptron)은 Frank Rosenblatt 가 1957년 제안한 초기 형태의 인공 신경망이다. 

다수의 입력(input)을 넣어 하나의 결과 값을 출력하는 알고리즘인데, 실제 우리 몸에서 작동하는 뉴런과 같이 신호가 역치 수준에 다다르면 활성화가 되는 원리에 따라 신호가 일정 크기 이상이 되면 
출력을 하는 것과 유사하다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/%EA%B7%B8%EB%A6%BC2.png?raw=true">
  <br>
  그림2. 퍼셉트론 구조
</p>


위 그림과 같이 단층 퍼셉트론은 여러개의 input 값을 받으면 각 $w_i$​ 는 가중치가 된다. 

이 구조는 우리가 익숙하게 알던 기존 회귀식과 다를 바가 없다. 우리는 이러한 단층 퍼셉트론에서 

$argmin_w \sum_i L(Y,f(x_i,w_i))$​​ 과 같은 Cost function을 갖고 가중치 $w_i$​ 를 조정하며 수치 예측을 

시도할 수 있고 마찬가지로 분류에서는 $Cross \ entropy$​를 사용하여 $w_i$​를 조정할 수 있다.

여기서 우리는 각 input 값과 가중치를 단순히 합한 뒤 이 합한 선형결합 식을 함수 $f$ 에 넣어 원하는 값을 출력한다.

앞서 이야기한 sigmoid 함수 혹은 ReLU 함수 같은 것들이 여기서 적용되는 것이고 이 $f$​ 를 우리는 **활성화 함수** 라고 이야기 하는 것이며,

 **여기서 가장 중요한 사실은 $f$​ 라는 함수의 input은 선형결합의 형태이고 이것의 output은 비 선형 형태인 것이다!** 

이 사실은 내 개인적인 생각으로 신경망을 이해하는 과정에서 가장 핵심이 되는 것이라 생각한다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/stepfuntion.png?raw=true">
  <br>
  그림3. step function
</p>


위 그림은 단층 퍼셉트론에 적용시키는 활성화 함수 중 매우 단순한 step funtion을 적용시킨 것인데 식으로 표현하면 다음과 같다.

$$ \begin{cases} 1 \ \ \ \ if \sum w_ix_i >\theta \\ 0 \ \ \ \ otherwise \end{cases} $$​ 

(여기서 $\theta$는 0으로 표현 돼 있지만 임의로 설정 가능하다)​

여기서 중요한 사실은 이러한 단층 퍼셉트론은 **선형 분류**만 가능하다는 사실인데 나는 이러한 사실이 잘 이해가 안돼 조금 더 자세하게 설명 해 보려 한다.

## 왜 선형분류만 가능한가?

<p align = "center">
  <img width = "700" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/106.jpg?raw=true">
  <br>
  그림4. AND OR XOR
   출처 https://thebook.io/080228/part03/ch06/03-01/
</p>


우리는 단층 퍼셉트론이 위와 같이 선형 분류는 가능하지만 XOR 문제에 대한 분류는 불가능하기 때문에 한계가 있다는 사실에 대해서는 자주 들어봤다.

하지만 이게 과연 진짜 무슨 말인지에 대해 나 스스로도 이해가 부족하기 때문에 이 것의 의미를 파악하기 위해 좀 더 추가적인 설명을 해보려 한다.

가끔 우리는 실제 데이터 벡터 $y$ 와 $\hat{y} = w_1x_1 + w_2x_2 + bias$​ 에서의 $\hat{y}$ 의 의미를 혼돈하는 경향이 있다. 

그렇기 때문에 sigmoid 함수와 같은 활성화 함수 즉, 비선형 꼴로 함수 값을 출력하는데 왜 선형 분류밖에 안되는지와 같이 각각의 함수와 수식이 무엇을 의미하는지 헷갈려하며 혼돈에 빠진다.

그렇기 때문에 우리는 우리가 예측하고자 하는 것이 무엇인지, 선을 긋는다는 게 무슨 의미인지에 대해 정확히 알아야한다.

우리가 앞서 지정한 $\hat{y}$ 는 input 데이터가 들어왔을 때 퍼셉트론이 내놓는 **예측값**이다. 

다시 말해서 우리가 선형결합 꼴의 $w_1x_1 + w_2x_2 + bias$​​ 식을 퍼셉트론에 넘겨주게 된다면 활성화 함수를 거쳐 특정 임계값을 넘는다면 1로 출력을 하는 과정을 거친다. 

우리가 Binary Classificaion 을 한다고 할 때  활성화 함수를 $f$ 라고 두면 $f(z) = P(y=1|z)$ 라고 표현할 수 있다. (여기서 $z = w_1x_1 + w_2x_2 + bias$ )

단층 퍼셉트론은 $z$​ 의 값을 조정하여 이렇게 표현된 $f(z)$ 의 영역 즉, Decision boundary 를 만드는 것이다. 

여기서 $z$ 값에 따라서 $f$ 의 출력이 바뀌는데 만약 $P(y=1|z) > 0.5$ 일 때 $f$ 의 출력 값이 1이 된다면 우리는 특정 경계를 찾을 수 있다. 
<br>

```python
import matplotlib.pyplot as plt
import random
w1 = random.random()
w2 = random.random()
bias = 1
x1 = [w1*i + bias for i in range(1,100)]
x2 = [w2*j for j in range(1,100)]
plt.plot(x1,x2)
```

위 식과 같이 코드를 작성하여 $x_1,x_2$ 에 대한 plot을 찍어보면 직선의 꼴을 보이는 것을 알 수 있다. 여기서 이 직선은 f 의 경계값이라고 생각할 수 있다. 

3차원으로 생각하면 보다 직관적으로 나타낼 수 있다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/%EA%B7%B8%EB%A6%BC5.png?raw=true">
  <br>
  그림5 3차원에서 단층 퍼셉트론의 경계
</p>


위 그림에서 보듯이 $x_1 , x_2$​ 로 표현된 아래 평면에서 절벽이 급히 하강하는 절벽이 보인다. 저 절벽이 우리가 plot을 찍을 때 보이는 경계라고 생각할 수 있다. 

우리는 직관적으로 이해하기 위해 두가지 변수만 사용했지만 실제 3개 이상의 변수를 사용한다면 더욱 고차원에서 위와같은 상황이 펼쳐지지만 변하지 않는 것은 경계가 선형이라는 것이다. 

간혹 우리는 3차원 혹은 그 이상의 차원에서 벌어지는 일을 2차원으로 압축시켜 표현하며 다르게 이해하는 경향이 있는데 나 같은 경우 이렇게 이 것을 좀 더 직관적으로 풀어서 이해하는 것이 더 정확하게 이해할 수 있는 것 같다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/%EA%B7%B8%EB%A6%BC6.png?raw=true">
  <br>
  그림6 다른 시점에서의 경계
</p>


## 시점의 변환

세상에는 여러가지 문제들이 있다. 이러한 문제를 풀 때 우리는 AND, OR 만을 이용해서 문제를 풀기엔 세상은 매우 복잡해서 XOR 과 같은 논리 문제도 해결을 해야한다. 

하지만 잎서 살펴봤듯이 단층 퍼셉트론은 이러한 XOR 문제를 풀 방법이 없다. 그 이유는 단층 퍼셉트론이 분류하는 경계는 선형이기 때문이다.

그렇다면 우리는 어떻게 XOR 문제를 풀 수 있을까?

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/%EA%B7%B8%EB%A6%BC8.png?raw=true">
  <br>
  그림7 negation AND 
  <br>
   from Pascal Vincent's slides
</p>


<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/%EA%B7%B8%EB%A6%BC7.png?raw=true">
  <br>
  그림8 XOR
  <br>
   from Pascal Vincent's slides
</p>


위 그림은 $AND(\bar{x_1},x_2)$ , $AND(x_1,\bar{x_2})$ 에 대한 표현과  XOR 문제에서의 축을  $AND(\bar{x_1},x_2)$ , $AND(x_1,\bar{x_2})$ 로 바꾸어 나타낸 XOR 문제이다.

여기서 우리가 발견할 수 있는 매우 흥미로운 점은 각 축을 $AND(\bar{x_1},x_2)$​ , $AND(x_1,\bar{x_2})$​​ 로 바꾸니 XOR 문제가 선형분류로 풀 수 있는 형태처럼 보인다는 것이다.

**만약 우리가 각 축을 잘 바꾸거나 input을 적절하게 잘 표현하면 XOR을 풀 수 있지 않을까?  하는 가정에서 출발해보자**

$AND(\bar{x_1},x_2)$​ 에 대해서 간단히 설명하자면 $AND(\bar{x_1},x_2)$​ 에서 $\bar{x_1}$​ 과 $x_2$​​가 1이라면 1 아니면 0 을 출력하는 함수이다. (negation of AND 라고 표현하겠다.  $\bar{x_1}$ 은 $x_1$ 의 부정)





| $AND(\bar{x_1},x_2)$ | $AND(x_1,\bar{x_2})$ | $x_1$ | $x_2$ |
| :------------------: | :------------------: | :---: | :---: |
|          0           |          0           |   0   |   0   |
|          0           |          0           |   1   |   1   |
|          1           |          0           |   0   |   1   |
|          0           |          1           |   1   |   0   |

<br>
  											표 1. $x_1,x_2$​ 와 negation of AND 의 관계

위 표는 $x_1,x_2$​​를 각 negation of AND 에 넣을 때 나오는 값을 나타낸다.

그림 8에서 (0,0), (1,1) 은 위 표를 거친다면 $AND(\bar{x_1},x_2)$ , $AND(x_1,\bar{x_2})$ 로 표현된 좌표계 안에서 (0,0) 으로 표현된다. (0,1) 과 (1,0) 은 서로 위치가 바뀐다. 

그렇다면 우리는 XOR 문제에 대해 negation of AND를 각 축으로 표현하면 충분히 선형 분류가 가능할 것 같다. 

이 말인 즉슨  $AND(\bar{x_1},x_2)$ ,  $AND(x_1,\bar{x_2})$ 를 각 $f$ , $g$ 라고 표현한다면 $x_1 , x_2$를 각 함수 통에 집어 넣은 뒤 그 함수통을 XOR 의 함수통 $z$ 에 집어넣는다면 XOR 문제를 해결 할 수 있다는 것이다. 

$z(f(x_1,x_2) , g(x_1,x_2))$ 라고 표현할 수 있는데 이러한 표현으로 우리는 XOR 문제를 선형분류 문제의 관점으로 보아 해결할 수 있다는 것이다. 

그렇다면 우리가 해야 할 일은 무엇인가? 퍼셉트론의 층을 몇 개 더 쌓는 것이다. 위 예제와 같은 경우에는 퍼셉트론 층이 하나 더 있다고 생각하면 될 것이다.

# 다층 퍼셉트론

층을 하나 더 쌓는다는 이야기는 무엇일까?

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-01-06%20165516.png?raw=true">
  <br>
  그림9 다층 퍼셉트론
</p>


다층 퍼셉트론의 구조는 다음과 같이 입력층과 은닉층이 있는데 이러한 은닉층은  여러 개가 될 수 있다. 각 층마다 결과값을 다음 층에 전달하고 마지막 출력층에서 결과 값을 출력하는 것이다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/%EA%B7%B8%EB%A6%BC%209.png?raw=true">
  <br>
  그림10 비선형 나누기
</p>


XOR 문제를 풀 가능성을 제시할 때 우리는 위 그림을 확인할 수 있다. 하지만 여기서 오해하기 쉬운 것은 다층 퍼셉트론을 만든다면 은닉층의 퍼셉트론이 저렇게 비 선형으로 자동으로 나누는 것인가? 하는 오해를 할 수 있다는 것이다.

하지만 저기서 $x_1,x_2$​ 는 입력층에 들어간 input 벡터이다. 다층 퍼셉트론도 결국 여러 개의 퍼셉트론이 연결 된 구조이기 때문에 각 층마다 돌아가는 방식은 동일하다.

**우리가 유념해야 할 사실은 각 층마다 다음 층에 전달하는 과정은 모두 선형이라는 것이다.**

 각 층은 단순히 각 가중치를 더해 선형결합 형태의 값을 받고 활성화 함수를 통해 비 선형 결과 값을 내놓는데, 또 이 값은 다음 층에서 가중치들의 합으로 선형 결합되어 비선형 값을 출력하는 것을 반복하는 것이다.

이 과정에서 처음 input 벡터들은 각 층마다 변환되고 결과에 이르러서는 비선형 형태로 변형되는 것이다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/perceptron1/%EA%B7%B8%EB%A6%BC10.png?raw=true">
  <br>
  그림11 은닉층이 1개일 때
</p>


위 그림은 보다 직관적으로 해석 가능한 그림이다. 

위 그림은 입력층 노드가 2개 은닉층 노드가 4개 출력층 노드가 1개 일 때를 표현한 것이다. 

여기서 우리는 이변량 정규분포 꼴의 출력층 함수를 표현하기 위해 각 은닉층에서 선형으로 표현된 경계값 $x_{1i}$​​​​​  들의 가중치와 편향을 조정하는 것이다.

이렇게 신경망 input 에서 Weight 와 Hidden 을 거쳐 Output을 내보내는 과정을 ‘Feed -Forward’라고 한다.

이러한 가중치의 조절을 통해 우리는 이제 XOR 과 같은 비선형 분류 문제도 해결할 수 있다. 

이러한 은닉층이 충분히 많다면 최적값을 꽤 잘 찾을 수 있다. 하지만 여기서 우리가 봉착하는 문제는 과연 이러한 가중치 조정을 어떻게 할 수 있냐는 것이다. 

이 문제에 대해 우리는  ‘Gradient Descent’ 와 ‘Back Propagation’ 라는 강력한 Idea를 통해 가중치 조정을 할 수 있다.

이 개념에 대해서는 다음 게시글에 좀 더 면밀하게 살펴보겠다.

