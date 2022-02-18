# Attention

## 들어가기 앞서

우리는 앞서 Seq2Seq 모델에 대해서 살펴보았다. 

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/seq2seq/seq2seq1.png?raw=true">
  <br>
  그림 1. Seq2Seq 구조
</p>

다시 복기하자면, Seq2Seq 모델은 위와 같이 Encoder 와 Decoder 의 구조를 띄고 있다.

기존 RNN 셀을 이어붙인 모델들처럼 하나의 입력값에서 하나의 출력값이 나오는 것과 달리 Encoder 부분에서 Hidden score 만 출력하고 Decoder 부분에서는 한번에 들어온 Context vector 의 정보를 토대로 예측을 수행한다.

하지만 이러한 RNN 기반 Seq2Seq 모델에서는 두 가지의 문제가 있다. 

첫째로는 하나의 고정된 크기의 Context Vector 가 들어오기 때문에 정보의 손실이 발생 할 수 있다. 여기서 Context Vector 란 Encoder 마지막 셀에서 나오는 Hidden state 값인데, 이 값이 Encoder 의 모든 문맥 정보를 담고 있다고 기대하는 것은 무리가 있기 때문이다.

두번째로는 Gradient Vanishing 문제이다. 본질적으로 RNN 셀을 이어 붙이는 구조를 갖기 때문에 BBTT 를 거치더라도 문장이 길어진다면 근본적인 Gradient Vanishing 문제에서 자유로울 수 없다.

이러한 결과는 기계 번역 분야에서 성능의 저하를 야기한다. 이를 위해 등장 한 것이 Attention(어텐션)이다.

## Attention?

### 어텐션이란

어텐션의 핵심 아이디어는 이름 그대로 모델이 **중요한 부분에 ‘주목’** 한다는 것이다. 

Seq2Seq에서 Context vector 를 한 번 참고하여 (물론 ‘Peeky’도 있지만 ) 그 결과를 토대로 Decoder 는 쭉 학습한다. 

Attention은 Decoder 에서 출력 단어를 예측하는 매 시점(time-stamp)마다 인코더에서의 전체 문장을 참고하는 것이다. 하지만 여기서 모든 문장 정보에 대해 동일하게 참고하는 것이 아니라 각 시점마다 Encoder 에서 어떠한 단어의 정보가 연관이 있는지 계산하여 가중치를 더해 진행하게 된다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Attention/attention.png?raw=true">
  <br>
  그림 2. Attention
</p>

위 그림은 Attention 메커니즘을 적용한 간단한 그림이다. Decoder 셀에서 a 에 대한 단어를 예측 할 때 어텐션 메커니즘을 적용한 모습인데, 그 과정을 알아보자.

위 그림에서 볼 수 있듯이 Decoder는 a 라는 값의 예측 값을 내놓기 위해 Encoder 부분의 모든 Cell 의 정보를 받는다.

여기서 Softmax 함수에 대해서 살펴보면, Softmax 함수에서의 결과값들은 각각의 Encoder 의 단어들이 해당 출력값을 예측 할 때 얼마나 도움이 됐는지 수치화 한 값이다.

위 그림에서는 Cat 이라는 단어가 해당 셀에서 예측을 수행 할 때 가장 도움이 된 단어라고 할 수 있다.

Encoder 의 각 단어가 수치화 된 정도가 측정되면, 이 정보를 Decoder 로 전달하게 된다. 위 그림에서 Pass to Decoder 로 표현된 화살표가 그 값을 의미한다.

이 수치를 어떻게 생성하는지 자세히 알아보자.

### Attention score

기존 Decoder 의 현재 시점 $t$​​ 에서 필요한 입력값은 시점 $t-1$​​​ 의 셀에서 나온 hidden state 값($s_{t-1}$​)과 $t-1$​ 시점에서의 출력 단어이다. 

하지만 어텐션 메커니즘에서는 추가적으로 $t$ 시점에서 단어를 예측하기 위해 Attention Value 를 필요로 하는데 이 값을 $a$ 라고 불러보자.

이러한 Attention Value 가 어떻게 반영되는지에 대해 알아보기 전 밑걸음이 되는 Attention Score 를 구해보자. 

이 스코어는 인코더의 모든 은닉 상태 각 값이 $t$ 시점의 디코더와 얼마나 유사한지 판단하는 값이다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Attention/attention3.png?raw=true">
  <br>
  그림 3. Dot-Attention
</p>

위 그림은 어텐션 스코어 계산 과정 중 Dot-attention 의 계산과정인데, 위 스코어는 시점 $t-1$​​​ 에서의 hidden state 값과 각각의 Encoder 의 hidden state 값을 내적 계산을 수행한다.

 $e_{tj} = s_{t-1}^Th_j$​​​​​ 

위 수식에서 $e_{tj}$​​ 는 디코더가 $t$​번째 단어를 예측할 때 직전 step 에서의 hidden state Vector 인 $s_{t-1}$​ 이 인코더의 열벡터 $h_j$​​와 얼마나 유사한지를 나타내는 값이다. 

이 Score 의 형태는 Vector 의 Dot-product로 진행되기 때문에 스칼라 값이다.  다시말하자면  $j$​​ 번 째 hidden state 인 열 벡터와 유사한 정도를 수치화 한 것이다.

여기서 $e_t = [s_{t-1}^Th_1 , ...,s_{t-1}^Th_N]$​ 의 형태가 되는데, $s_{t-1}$​과 Encoder 의 모든 은닉 상태의 값을 내적한 결과를 담아 낸 Attention score 의 모음 값이다.

### Attention Distribution

우리는 이제 $e_t$​​ 라는 통에 Encoder 의 모든 hidden state 와 $s_{t-1}$의 유사도를 구했다. 

각각의 Encoder 의 단어 별로 유사한 정도를 수치화 한 것이다. 이제 각 내적값들에 대해서 Softmax 함수를 적용하여 각 가중치를 0부터 1 사의 값으로 비율로 환산해보자. 

$a_{tj} = \frac{exp(e_{tj})}{\sum_{k=1}^{T_x}exp(e_{tk})}$​​​​   이러한 식으로 $a_{tj}$ 각각의 값이 구해지는데,  이 값들이 각 Encoder 의 hidden state 값과 Decoder 의 $t-1$​ 시점의 hidden state 값이 얼마나 유사한지에 대한 값을 비율로 나타낸 것이라고 할 수 있다.

각각의 $a_{tj}$ 값은 일종의 분포라고 생각 할 수 있다.

### Attention Value

마지막으로 $a_t = \sum_{j=1}^{T_x}s_{tj}h_j$​​​ 의 값으로써 가중치 합을 구한다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Attention/attention4.png?raw=true">
  <br>
  그림 4. Attention Value
</p>

위 그림과 같이 우리는 Attention Value 인 $a_t$ 값을 구했다. 

이 Attention Value 가 구해지면 어텐션 메커니즘은 $t$ 시점에서의 Decoder 의 hidden state 값인 $s_t$

와  $a_t$를 결합(concatenate)하여 하나의 벡터로 받는다.

이 Vector 를 위 그림처럼 $v_t$​ 라고 할 때 이 $v_t$​ 를 $y$ 의 예측 값인 $\hat{y}$​​ 연산의 입력으로 사용하므로 결과적으로 Encoder 의 모든 셀의 정보에서 유사한 정도와 이전 hidden state 값을 참고하여 예측을 수행하게 된다.

이것이 Attention 메커니즘의 핵심이라고 할 수 있다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Attention/attention5.png?raw=true">
  <br>
  그림 5. Attention Value 연산
</p>

이후 출력층의 연산이 이루어지기 전 $v_t$ 를 바로 출력하는 것이 아닌 학습 가능한 가중치 행렬과의 연산을 한번 더 진행하여 출력층 연산을 위한 새로운 벡터 $\hat{s}$ 를 만든다.

수식으로 표현하자면 $\hat{s} = tanh(W_c[a_t;s_t]+b_c)$​ 위와 같은 구조를 띄는데 여기서 $b_c$ 는 편향을 나타낸다. 

이후 이 $\hat{s}$ 를 입력으로 사용하여 $\hat{y}_t = Softmax(W_y\hat{s}_t + b_y)$ 의 값으로 예측한 값을 얻게 된다.

### 많은 Attention 기법

우리는 Seq2Seq 모델과 Attention 기법을 적용한 아키텍쳐에 대해서 알아보았다.

Attention score 를 구하기 위해 우리는 Vector 의 내적값을 사용하였는데, 내적 말고도 여러가지 방법을 사용할 수 있다. 

다시말해 각 Vector 의 유사도를 구하기 위한 방법으로 여러 지표들이 사용되는 것이다. 

이러한 Attention 아이디어는 매우 강력하다. 

그렇기에 이후 등장하는 모델들은 아예 RNN 기법을 사용하지 않고 Attention만 사용한 모델들이 나오기도 한다. 

다음 게시글에서는 Attention is all you need 라는 논문을 살펴보며 Attention 만 사용하는 아키텍쳐에 대해 알아보겠다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Attention/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-02-10%20231507.png?raw=true">
  <br>
  그림 6. Attention 수식 요약
</p>
