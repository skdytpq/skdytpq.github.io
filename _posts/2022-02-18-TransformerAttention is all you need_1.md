# Transformer:Attention is all you need

게시글 그림에 대한 출처는 [1) 트랜스포머(Transformer) - 딥 러닝을 이용한 자연어 처리 입문 (wikidocs.net)](https://wikidocs.net/31379) 에서 참고하였습니다.

## 들어가기 앞서



우리는 앞서 Seq2Seq 기법에 Attention 메커니즘을 적용한 예시에 대해서 살펴보았다. 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Attention/attention.png?raw=true">
  <br>
  그림 1
</p>

복기하자면 위와 같은 구조로 이루어진 Seq2Seq + Attention 구조는 Context Vector 를 참고하면서도 Encoder 의 각 노드에서의 hidden state 값을 입력받아 참고하여 사용 한 것이다.

이렇게 진행 한 이유는 말했듯 고정된 길이의 Context Vector 가 입력 시퀀스의 모든 문맥 정보를 담지 못 할 것이라는 생각에서였다.

또한 RNN Cell 을 통과 할 때 일어 날 수 있는 Gradient Vanishing 문제도 Attention 을 사용한 추가적인 이유이다.

그렇다면 RNN 모델을 이용하지 않고 온전히 Attention 기법으로 모델을 만들게 되면 어떨까?

이러한 생각에서 나온 것이 Transformer 인데, 이 Transformer(트랜스포머)는 RNN 을 전혀 사용하지 않는 모델이다.

Transformer 의 논문인 Attention is all you need 라는 논문을 토대로 어떻게 진행되는지 살펴보자.

## Transformer

### word embedding

우선 모델이 학습하기 전 단어들을 수치적으로 정의해야 한다.

트랜스 포머를 번역 모델로 사용한다고 생각해보자.

단어 자체를 컴퓨터가 해석하기에는 무리가 있기 때문에 우리는 입력 언어와 출력 언어의 사전을 우선 정의해준다.

이렇게 사전을 통해 단어를 정의 할 때 기본적으로 one-hot encoding 을 사용하는데, 이는 단어의 개수가 n 이라면 n차원의 벡터 공간을 만들어 모든 단어를 하나의 기저에 위치시킨다.

예를 들어 각 사전에 1만개의 단어가 들어있다고 한다면 각 단어는 1만 차원의 벡터 공간에서 각 1차원의 공간을 차지하고 있다.

강아지 = [0,0,...,1,0,...,0]  

강아지라는 단어가 우리 언어 사전의 7000번 째의 위치에 존재한다고 한다면, 강아지라는 언어에 대한 Vector 표현은 9999개의 0과 단 1개의 1로 구성 돼 있다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B82.jpeg?raw=true">
  <br>
  그림 2
</p>

위 그림을 보면 보다 직관적으로 이해가 가능할탠데, 이러한 표현의 문제점은 단어 사전의 크기가 크면 너무나 많은 공간적 낭비가 발생하고 벡터끼리의 연산이 어렵다는 것이다.

그렇기 때문에 보통 우리는 Word embedding 을 통해 단어의 차원을 줄이는데, 각 단어의 유사도를 고려하여 1만개로 표현된 벡터의 차원을 임베딩 차원에 응집하여 표현 할 수 있는 것이다.

이렇게 된다면 보다 효율적으로 단어를 표현 할 수 있기 때문에 거의 모든 모델은 이러한 임베딩 과정을 거친다. 위 논문에서는 이러한 임베딩 차원을 512 차원으로 지정하였다.

우선 Word embedding 에 의도에 대해 간략히 이해하고 이러한 Word embedding 에 대해서는 다른 게시글에서 자세히 써보겠다.

### Embedding & Positional encoding

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B83.jpg?raw=true">
  <br>
  그림 3
</p>

트랜스 포머의 경우 위와 같이 Input 과 Output 데이터들에 대해 우선적으로 임베딩을 진행한다.

 위 논문에서는 이러한 임베딩 차원을 512 차원으로 지정하였는데, 이후 위 그림에서와 같이 Positional encoding 이라는 것을 거친다.

기존 RNN 모델을 통해 Encoder 와 Decoder 를 구성 할 때 우리는 각 단어의 위치 정보를 고려하지 않아도 됐다.

그 이유는 임베딩 된 각각의 단어들은 순차적으로 RNN 셀을 거치며 연산되기 때문에 우리가 굳이 단어 배열에 대한 위치 정보를 주지 않아도 알아서 그러한 정보를 가질 수 있기 때문이다.

하지만 트랜스포머의 경우 RNN 셀을 전혀 사용하지 않기 때문에 임베딩만 거친다고 하여 단어의 위치 정보를 줄 수 없다.

따라서 단어의 위치 정보를 주기 위해 각 단어의 임베딩 벡터를 더해서 Encoder 와 Decoder의 입력으로 사용한다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B84.png?raw=true">
  <br>
  그림 4
</p>

embedding vector 와 positional encoding 은 간단하게 위 그림과 같은 방식으로 진행되는데, 어떠한 과정을 거치는지 자세히 살펴보자.

트랜스 포머는 위치 정보를 가진 값을 만들기 위해 아래 두 개의 함수를 사용한다.

$PE(pos,2i) = sin(pos/10000^{2i/d_{model}})$

$PE(pos,2i+1) = cos(pos/10000^{2i/d_{model}})$

위 식은 직관적으로 어떻게 해석하기가 어렵다. 그렇다면 위 식은 어떠한 이유로 유도되었을까?

#### Positional encoding 수식의 의미

우선 논문에서는 입력 시퀀스의 위치에 대한 정보가 없기에 위 수식과 같은 Encoding 을 하여야 한다고 이야기한다.

하지만 input 문장 시퀀스의 길이는 고정적이지 않는데 어떻게 가변적인 문장의 길이에서 일정하게 위치 정보를 부여 할 수 있을까?

이상적으로 Encoding 을 수행하기 위해 필요 한 것은 무엇인지 부터 살펴보자.

1) 각 time-step 마다 하나의 유일한 encoding 값을 가져야 한다.
2) 서로 다른 길이의 문장에 있어 두 time-step 간 거리는 일정해야 한다.
3) 모델에 대한 일반화가  가능해야한다. 즉, 가변적인 길이의 문장에 모두 적용이 가능해야 한다.
4) Encoding 결과값이 모두 동일해야 한다.

예를 들어 label encoding 처럼 각 time- step 마다 1부터 숫자를 할당하게 된다면 시퀀스가 길어 질 때 숫자가 매우 커질 수 있어 학습이 매우 불안정 할 수 있다.

따라서 우리는 각각의 인코딩 정보가 상대적으로 고유한 값을 갖기 위한 방법을 찾아야 한다.

우선 각 행렬에서 $d_{model}$​의 차원은 같을 수 밖에 없다. 모든 단어가 이 차원 안에서 수치화 되어 표현되기 때문이다.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B85.png?raw=true">
  <br>
  그림 5
</p>

우리는 $d_{model}$​​ 차원으로 축소가 된 각 단어들에 대해서 위치를 부여하기 위해 위 그림처럼 시퀀스 길이가 7이고 Embedding 차원이 $d_{model}$​인 행렬을 생각해보자.

지금 보이는 위의 그림은 이진수 표현으로 각 길이를 표현 한 것이다. 

하지만 위 그림처럼 이진수로 이산화하여 표현하기 보다 각 Vector를 연속적인 값으로 나타내서 함수의 대한 정보와 대표값으로 표현되는 각 위치에 대해서 표현하면 더 좋을 것 같다.

보다 본질적인 이산화 과정의 문제점이 있다. 

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B87.png?raw=true">
  <br>
  그림 6
</p>

위 그림에서 검정색 선은 이산함수의 이진수 출력인데 ((0,0), (0,1), (1,1), (1,0))으로 되어있다.

파란 직선은 그냥 연속 함수에서 (0, 0.33, 0.66, 0.99) 이다.

파랑 선은 점 사이 간격이 일정하지만 검은 점들 사이의 거리는  1과 $2^{1/2}$ 로 일정하지 않다.

따라서 각 위치간의 거리를 이산화하여 표현하게 된다면 모델은 실제 거리 계산을 다르게 할 것이다.

따라서 연속적이고 부드러운 함수를 찾아 0과 1 사이를 부드럽게 움직이며 positional encoding 이 가능한 것을 찾아야 한다. 

이렇게 표현하기 가장 적합한 방식이 삼각함수를 이용한 방식 이라고 할 수 있다.

$sin, cos$ 의 표현은 [-1,1] 안에서 표현이 가능하여 일종의 정규화 효과가 있으며 각 인덱스, 다시말해 앞서 식의 표현에서 $i$​​ 마다 주기를 다르게 하여 각 타임스텝 간의 거리가 일정하게 유지되도록 한다. 

이러한 표현에 대해서 나의 해석이 확실하지 않기 때문에 좀 더 공부해보고 추가 내용 작성을 하겠다.

아무튼 $PE$​​ 라는 함수를 이용하여 positional encoding 은 [시퀀스 길이, 임베딩 차원]의 형태를 갖고 이 값을 시퀀스 정보를 담은 임베딩 행렬과 더한다.

### Attention

자 우선 이제 위치정보까지 담은 임베딩 행렬이 생성되었다고 가정 한 후 각 Encoder Decoder 에서 어떻게 학습이 진행되는지 알아보자.

#### Encoder

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B88.png?raw=true">
  <br>
  그림 7
</p>

위 그림은 트랜스포머의 인코더 구조이다. 

트랜스포머는 하이퍼 파라미터인 num_layers로 인코더 층을 쌓는데, 논문에서는 총 6개의 인코더 층을 사용하였다. 

하나의 인코더 층에는 총 2개의 sub layer 로 나뉘어진다. 각각 Self- Attention 과 Position-wise 피드 포워드 신경망이다. 

Multi-head 의 경우 Self -attention 을 병렬적으로 사용하겠다는 것이다.

우선 Self attention 에 대해서 알아보겠다.

#### self attention

앞선 게시글에서 Attention 메커니즘에 대해 꽤 알아보았다. 여기서 각 Query, Keys, Values 의 의미를 정의해보자.

* Query(Q) : t 시점의 디코더 셀에서의 은닉상태
* Keys(K) : 모든 시점의 인코더 셀에서의 은닉상태
* Values(V): 모든 시점의 인코더 셀에서의 은닉상태

이 과정에서 최종적으로 Values를 이용하여 상대적 비율인 Attention Value 를 얻는다.

하지만 Self - Attention 의 경우 Q,K,V 가 모두 입력 문장의 모든 단어 벡터들로 의미가 같다. 

그렇다면 어떠한 메커니즘으로 Attention value 를 구하는가?

우선 i am a student 라는 문장에 대해서 Self attention 이 어떻게 작동하는지 살펴보자.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B89.png?raw=true">
  <br>
  그림 8
</p>

위 그림에서 볼 수 있듯 Student 라는 단어 벡터에 대해 생각해보자.

여기서 벡터의 길이는 $d_{model}$ 로 적용한 512 이고, 각 가중치 행렬의 사이즈는 $d_{model} \times (d_{model} / num \ head)$ 가 된다.  num_head 를 8로 설정한다면, 가중치 행렬의 사이즈는 512 X 64 의 사이즈를 갖게 되며 각 나오는 Q , K ,V  Vector 는 1 X 64 의 사이즈르 갖게 된다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B810.png?raw=true">
  <br>
  그림 9
</p>

이후 나온 Q K 벡터에 대하여 기존의 어텐션 메커니즘과 같이 내적을 수행하여 유사도를 구한다.

Score 를 산출하는 함수는 여러가지가 있을 수 있는데, Self Attention 의 경우 $q\centerdot k/ n^{1/2}$  로 score 를 구하는데, 이러한 과정은 내적값을 Scailing 해준다고 하여 Scaled dot- product attention 이라 한다.

여기서 스케일링을 수행하는 $n$ 의 값은 $d_k$ 의 값인 64의 루트를 씌운 8이 된다. 

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B811.png?raw=true">
  <br>
  그림 9
</p>

이후 위의 그림처럼 소프트맥스 함수를 사용하여 Attention 분포를 구한 후 해당 가중치를 V 벡터에 곱해주어 최종적인 Attention Value 를 구한다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-02-18%20152624.png?raw=true">
  <br>
  그림 10
</p>

사실 위의 그림처럼 한 단어씩 하기 보다 각 문장에 대해 통째로 행렬 연산을 수행 할 수 있다.

위 그림이 시사하는 바는 한번의 Attention 프로세스에서는 각 단어 별 Weight 가 동일하다는 것이다. (일종의 weight sharing)

문장의 길이를 $len_s$ 라고 할 때 문장 행렬의 사이즈는 $(len_s \times d_{model})$ 이 되고 각각의 가중치 행렬은 $(d_{models} \times d_{models}/num head)$  가 되기 때문에 최종 산출되는 output 행렬의 사이즈는 $(len_s \times d_{model})$ 이 된다.

 그렇다면 여기서 Q 행렬과 K 행렬의 전치행렬을 내적 해 준다면 앞서 하나의 스칼라 값으로 표현된 각 단어의 score 값이 모두 저장된 행렬이 나올 수 있게 되는 것이다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B814.png?raw=true">
  <br>
  그림 11
</p>

위 그림에서 산출물 행렬에서 “I” 문장을 예로 살펴볼 때, 1행의 있는 모든 값들은 $Q$ 에서 ‘I’ 행과 $K^T$​ 에서의  ‘i’ ‘am’ ‘a’ ‘ student’ 인 모든 값과의 내적을 포함하고 있다.

이와 마찬가지로 산출물의 2번 째 행에 대해서도 ‘am’ 과 다른 단어들의 내적값이 모두 들어가 있다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B815.png?raw=true">
  <br>
  그림 12
</p>

이후 이 산출물을 각 행에 대해 Softmax 함수를 거친 뒤 $V$ 벡터와의 내적을 진행하여 Attention Value 행렬을 계산한다.

위 그림에서 각 V 에는 각각 나온 Attention distribution 값과 각 단어의 임베딩 값들을 내적하여 최종적으로 나온 Attention Value 행렬 식 안에 각 단어 별 Attention value 와 V 행렬의 가중치 값들이 내적되어 들어가게 된다.

따라서 $Attention(Q,K,V) = softmax(\frac{QK^T}{d_k^{1/2}})V$ 가 되는 것이다.

인코더에서는 이러한 Self-Attention 과정을 병렬적으로 num_heads 수 만큼 거친다. 

이러한 어텐션 과정은 각 어텐션 과정이 문장의 정보와 문맥을 각기 다른 시각에서 보고 있는 것이라고 생각할 수 있다.

Self - attention 을 통해 우리는 모든 입력층 단어에 대해서 어떤 단어와 유사한지, 문맥에서 어떤 단어와 연관이 있는지 주의를 기울여 살펴 볼 수 있게 된다. 

따라서 어텐션을 거친 각 행렬은 각각의 단어와 얼마만큼의 유사도를 가지는지에 대한 정보도 들어가 있는 것이다.

##### mask

Encoder 에서 수행하는 또 다른 것으로 masking 이라는 것이있다. 

문장을 배치사이즈로 묶어서 학습을 시킬 때 각 문장의 길이가 같다면 좋겠지만 그러한 경우는 흔치 않기 때문에 우리는 문장의 길이를 맞춰주기 위해 padding 토큰을 사용한다.

하지만 이러한 패팅은 아무런 의미가 없는 것이기 때문에 만약 문장에 패딩이 있다면 그 값에 매우 작은 값을 곱해주어($10^{-9}$) 어텐션 스코어에 영향을 주지 않게 한다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B816.png?raw=true">
  <br>
  그림 13
</p>

위 그림은 Soft max 함수를 거쳐 나온 Attention score 행렬인데, 여기서 pad 값은 매우 작은 값이기 때문에 Soft max 함수를 거쳐도 0에 수렴하는 값이 나와 어떤 유의미한 값도 갖지 않을 것이다.

##### FFNN & Residual connection & Layer Normaliazation

FNN 은 Encoder Decoder 모두 갖고 있는 sub layer 이다. 

우선적으로 Multi-head Self Attention 이 끝난 뒤 나온 $x$ 값은 $(seq \ len , d_{model})$ 의 행렬이다. 

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B817.png?raw=true">
  <br>
  그림 14
</p>

이후 Fully connected layer 과 같이 $W_1$​ 이라는 행렬과 곱을 한 뒤 활성화 함수를 거친 후 또 다른 $W_2$ 행렬을 곱한 뒤 최종 산출물 $F_3$ 이 출력된다.

여기서 $W_1$ 의 size 는 $(d_{model}, d_{ff})$ 인데, $d_{ff}$ 는 하이퍼 파라미터이며 논문에서 이 값은 2048 의 크기를 가진다. 

이후 $W_2$ 는 $(d_{ff} ,d_{model})$  의 size 를 갖는데,  이렇게 된 뒤 최종 출력 $F_3$ 은 $(seq\ len , d_{model})$ 의 사이즈를 갖는다.

여기서 하나의 Encoder 층 안에는 동일한 행렬 $W_1 \ , W_2$ 를 갖게 된다. 

Self Attention 전 Positional encoding 을 거친 단어 행렬의 크기도 최종 출력된 행렬 크기와 같기 때문에 이러한 Encoder 층을 직렬적으로 여러 개를 쌓을 수 있다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B818.png?raw=true">
  <br>
  그림 15
</p>

전체적인 Encoder 구조는 위 그림과 같은데, 여기서 Add&Norm 이란 잔차 연결과 정규화 과정이다.

ResNet 에서 Residual Block 을 추가한 것을 복기하면, 잔차 연결은 이 과정과 동일하게 진행된다.

<p align = "center">
  <img width = "200" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B819.png?raw=true">
  <br>
  그림 16
</p>

다시말해 위 그림과 같이 입력 input $x$ 와 출력인 $F(x)$​ 를 더해주는 것이다. 이렇게 진행하여 information flow 를 유지한다. 이 과정이 Add&Norm 에서 Add 에 해당한다.

이후 Norm 이라는 것은 층을 정규화 하는 것인데, Batch normalization 과정과 비슷하다.

다만 차이점은 정규화 과정을 각 행에 대해서 시킨다는 것이다.  다시말해 각 행의 정보는 각 단어 별 Attention Value 를 내포하고 있는데, 이러한 정보를 정규화 시키는 것이다.

이러한 전체적인 Encoding 과정을 num_layers 만큼 반복하게 된다. (가능한 이유는 입력 출력 행렬 크기가 같기때문!)

### Decoder

이제 디코더에 대해 알아보자. 

<p align = "center">
  <img width = "200" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B820.png?raw=true">
  <br>
  그림 17
</p>

위 그림과 같이 디코더도 동일하게 임베딩과 positional encoding 을 거친다.

디코더는 이렇게 문장 행렬을 입력받아 예측하는 Task 를 수행하는데, 이 과정에서 Seq2Seq 와 마찬가지로 교사 강요를 사용한다.

하지만 위 모델은 Seq2Seq 와 달리 입력 단어를 매 시점마다 하나씩 받는 것이 아니라 한번에 모든 입력 단어를 받게 된다.

예를 들어 ‘나는 사과가 좋다.’ 라는 문장에서 ‘사과’ 라는 단어를 예측 할 때 기존 Seq2Seq 는 ‘SOS’ , ‘나는’ 이라는 단어로 예측을 진행하는데, 트랜스포머의 경우 SOS 를 포함한 모든 문장 정보를 받게 된다.

따라서 트랜스포머 내부에서는 현재 시점의 예측에서 미래 단어를 참고하지 못하도록 look-ahead mask를 진행한다.

이 과정은 복잡한 것이 아니라, 기존대로 Attention score 를 구하는데, 여기서 자신보다 미래의 단어에 대해서 Masking 을 하는 것이다.

<p align = "center">
  <img width = "200" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B821.png?raw=true">
  <br>
  그림 18
</p>

위 그림에서 보듯이 Attention score 행렬의 각 행의 정보는 해당 단어에 대한 전체 Attention score 의 값인데, 마스킹을 해줘 삼각 행렬을 만들었기 때문에 이후 정보에 대한 Attention score 의 값은 참고하지 못하게 돼 있다.

이렇게 출력층의 각 단어에 대한 연관도와 문맥 정보를 구한 후 디코더의 두번째 층으로 향하게 된다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/Transformer/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B822.png?raw=true">
  <br>
  그림 19
</p>

위와같이 디코더의 두번째 층에서 연산은 똑같이 QKV 과정을 거치는데,  Query 행렬은 디코더에서 가져 온 것이고 K,V 행렬은 인코더의 행렬이다.

여기서 K 와 V 는 Encoder 에서 입력 단어에 대한 위치 정보 뿐만 아니라 각 단어가 서로 얼마나 유사하고 연관이 있는지, 각 단어는 어떤 성격이 있는지에 대한 정보도 담고있다.

따라서 결과값인 행렬에 대해서는 해당 번역 대상이 되는 단어와 Encoder 의 모든 단어에 대한 Attention 값이 들어가게 되는 것이다.

여기서 Q 는 앞선 레이어를 거쳤기 때문에 각 단어별로 유사도를 구한 척도가 되는 것이다.

이 외의 과정은 Encoder 와 동일하다.

이러한 디코더도 전체적으로 출력 행렬의 크기가 입력 행렬의 크기와 동일하기 때문에 num_layer 만큼 쌓을 수 있다.
