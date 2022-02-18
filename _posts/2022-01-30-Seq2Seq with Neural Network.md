---
title: Seq2Seq
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 딥러닝
key: 20220130
tags: 
  -
use_math: true

---
# Seq2Seq: Sequence to Sequence Learning with Neural Networks

## 들어가기 앞서

Convolutional Neural-Network 구조에서의 핵심은 이미지를 잘 학습 시키기 위해 기존 이미지에서 얻을 수 있는 공간적 특성을 Feature map 에 입력 시키는 것이었다.

CNN 에서 파생된 여러 Architector 를 보면 기본적인 구조는 Feature map 을 생성하고 pooling을 거치는 작업을 어떻게 효율적으로 진행하며 모델이 깊어 질 수록 기울기 문제를 어떻게 해결 할 지에 대한 것이었다.

RNN 에서 집중해서 둔 작업은 데이터에 포함되어 있는 연속성에 대한 정보이다. 

동영상, 음성, 자연어 데이터 각 하나 하나는 독립적인 개체로 볼 수 없고 CNN 에서의 각 픽셀 값과 마찬가지로 어떠한 요인을 받는다. 

그 요인 중 하나가 시간에 대한 정보이다. 단적인 예로, ‘나는 배가 아파서 약을 X.’ 에서 X 에 들어가야 하는 정보는 앞선 단어들의 결합에서의 의미에 영향을 받는다. 

여기서 이미지 데이터와 다른 것은 자연어의 경우 컴퓨터의 언어로 만들어지지 않았기 때문에 여러 단어의 이름, 관계를 수치형으로 표현하기 위해 Word Embedding 작업이 필요하다.

뒤에 나오는 단어에 대해 예측을 할 때 앞선 단어 혹은 앞서 일어난 event 에 대한 정보가 주요하기 때문에 우리는 RNN이라는 순환 신경망 모델을 사용하여 반복적으로 이전 단계에서 얻은 정보가 지속되도록 한다.

하지만  기본적인 Vanilla RNN 에서의 문제점은 이 모델에 들어가는 한 연속적인 정보의 길이가 길어지거나 여러번 순환되면 처음 들어간 정보가 희미해 질 수 있다는 것이다.

문장의 경우 바로 이전 단어가 이후 단어에 영향을 많이 주는 경우도 있지만 반대로 더 많은 문맥을 필요로 하는 경우가 있다. 

RNN 은 이렇게 문맥의 전반적인 정보를 골고루 학습하기 어렵기 때문에 RNN 의 셀 구조를 개선해서 forget gate, input gate 등을 거쳐 Cell 의 정보를 컨베이어 벨트처럼 운반하는 LSTM 모델, GRU와 같은 다양한 모델들이 파생되었다.

하지만 이러한 단순 RNN, LSTM 모델들은 입력의 크기와 출력의 크기가 고정되어 있기 때문에 실제 단어 예측, 번역 Task 에서 뚜렷하게 좋은 성능을 내지는 못한다.

이러한 RNN, LSTM 에 대해서는 다른 게시글에서 자세히 이야기 하겠다.

* 토큰화
* 수치형 임베딩
* 모델 학습

## Seq2Seq

Seq2Seq는 번역기에서 대표적으로 사용되는 모델이다. CNN 에서 파생된 여러 아키텍쳐도 결국 핵심적인 layer를 어떻게 조합하는지에 따라 이름이 달라졌듯이 이 Seq2Seq 모델도 단지 RNN 모델을 어떻게 조립했느냐에 따른 것이다.

Seq2Seq 는 2014년 처음 등장했는데, 이전엔 통계 기반의 기계번역들이 많이 사용 되었고 이 Seq2Seq 이후 딥러닝 활용 기법들이 많이 등장했다.(통계 기반 기계번역은 n-gram 같은 것들이 있다.)

여기서 시퀀스(Sequence)란 여러 토큰이 모인 것을 의미한다. 

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/seq2seq/seq2seq1.png?raw=true">
  <br>
  그림 1. Seq2Seq 구조
</p>

Seq2Seq 모델은 위와 같이 인코더, Context vector, 디코더로 구성되어있다.

하나의 시퀀스가 모델에 들어갔을 때 Encoder 는 하나의 벡터를 뱉고 Decoder 는 번역 대상의 언어로 Decoding을 진행하는 구조이다.

여기서 가장 중요한 점은 그림 1 중간에 위치한 Context Vector 이다. 여기서 Context Vector 는 고정된 크기를 갖고 있기 때문에 Encoder 의 정보는 이러한 고정된 크기의 Vector 에 응집된다.

입력 정보를 context Vector 가 모두 담아야 하기 때문에 일종의 Bottle neck 형태로 생각 할 수 있다. 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/seq2seq/seq2seq2.png?raw=true">
  <br>
  그림 2. Seq2Seq 구조2
</p>

기존의 RNN 에서는 각 층마다 행렬 $W_{tt}$​ 는 weight sharing 으로 동일하게 적용되고 $x_t, h_t$​ 라는 입력이 변화하게 된다.  여기서 hidden state 는 전체 문맥정보를 포함하고 있다. 

이러한 방법은 입력과 출력에 포함되어 있는 단어의 개수가 같다고 가정한다.

하지만 번역과 음성인식과 같은 경우 입력과 출력의 데이터 길이가 바뀔 필요가 있다.

그렇기 때문에 Seq2Seq 경우에는 RNN 에서 입력과 출력의 데이터 길이를 고정시키지 않고 유동적으로 가져갔다.

Seq2Seq 모델을 펼쳐서 살펴보면, 두 개의 RNN 아키텍쳐로 만들어 진 것을 확인 할 수 있다. 여기서 보면 알 수 있듯이, Seq2Seq 논문에서는 RNN 보다 LSTM 으로 모델을 구성 할 때 더 좋은 성능을 보였다고 이야기 하기에 각 인코더 디코더의 셀은 LSTM을 사용한다.

그 이유는 뭘까?

이전의 모델들은 이러한 시퀀스에 대해 잘 처리하지 못했다. 그 이유는 입력 데이터와 출력 데이터의 길이를 고정시켰기 때문이다.

즉, 다시말해 한 토큰에 대해서 학습하고 출력하는 것이 목적이라면 아무리 긴 문장이더라도 한 단어를 출력하여 전체 문장 문맥에 대해 학습 한 후 출력하는 것이 아닌 당장에 그 이전 정보를 갖고 출력하는 것이다.

이러한 문제점으로 생기는 성능 저하 문제를 해소하기 위해 Seq2Seq는 한 단어마다 반복하는 것이 아니라 하나의 Context 에서 벡터를 뽑은 후에 Decoder를 이용한다. 

또한 호흡이 긴 문장은 LSTM 이 효율 적이기에 LSTM 에 기반한다.

### Encoder

Seq2Seq 는 주로 번역 Task 에서 사용된다고 했으므로 번역에 대해서 살펴보자.

번역을 하기 위해 일단 주어진 시퀀스를 이해하는 작업이 필요하다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/seq2seq/seq2seq3.png?raw=true">
  <br>
  그림 3. Seq2Seq Encoder
</p>

위 그림과 같이 encoder RNN 은 번역해야 할 문장을 한 단어씩 읽는다. 그 후 기존 RNN 모델과 같이 hidden state 가 갱신되며 순차적으로 update 가 된다.

그 후 모든 단어를 다 처리하면 encoder RNN 에서 최종적인 hidden state 가 생기는데, 이 값은 Word embedding 처럼 Vector 표현이 된다.

Seq2Seq 논문에서는 최종 hidden state 를 거친 이 Vector 가 전체 문장에 대한 정보를 다 담고있을 것으로 판단하여 이 Vector 자체를 Context Vector 로 사용한다.

### Decoder

Decoder 가 하는 일은 간단한데, 우선 시작 토큰인 $<sos>$​ 가 들어오면, 다음에 등장 할 확률이 높은 단어를 예측한다. 

또한 첫 단에서 내놓은 예측을 토대로 반복적으로 끝날 때 까지 셀의 입력과 출력을 수행한다. 

디코더 같은 경우에는 출력을 할 때 FC 를 거쳐 출력을 하게 되는데, 여기서 기존 모델이 알고있는 vocab 에서 가장 확률이 높은 단어를 Soft max 함수를 거쳐 출력하게 된다.

기존 RNNLM같은 경우 Encoder 와 Decoder 없이 문장을 생성하였는데, Seq2Seq 모델의 경우 단순히 입력, 출력 계층을 나누어 Context Vector 라는 Bottle neck 을 거치게 설계 했다는 점이 차이가 있다.

여기서 Encoder Decoder 의 내용은 테스트 과정을 이야기 한 것이다. 하지만 모델의 학습 과정은 이러한 테스트 과정과 사뭇 다르다.

### 모델의 학습

모델의 학습 과정은 기존 RNNLM 에서의 학습 방법과 비슷하다. 기존 정답 Lable 과 Softmax-layer 가 생성하는 값과의 차이를 Cross Entropy를 이용하여 계산하는 과정이다.

<p align = "center">
  <img width = "500" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/seq2seq/seq2seq4.png?raw=true">
  <br>
  그림 4. Seq2Seq 학습
</p>

위 그림과 같이 학습을 진행하는데, 여기서 Decoder 만 학습이 될 수 있는 문제가 있기 때문에 BBTT 알고리즘을 사용하여 해결한다. 

이 BBTT 과정을 거쳐 encoder 에서 잘못된 부분을 개선해 나간다. 

즉 다시말해 Decoder 에서 모델 학습 시 Context Vector 까지 역전파 흐름이 흘러가 전반적인 Encoder 에서 hidden state 값들도 개선이 된다는 것이다.

또한 Decoder 앞 단어가 예측을 실수했을 때 뒤 단어를 아무리 잘 예측해도 번역 자체가 틀려지는 경우가 생길 수 있다. 

이 문제를 해결하기 위해 Top-K Beam Search라는 방법이 도입되었는데, Decoder 에서 출력을 할 때 가장 확률이 높은 단어를 고르는 것이 아닌 k 개의 후보군을 추출하여 가설을 설정하게 된다.

하지만 이렇게 된다면 가설의 개수는 문장이 길어질 수록 k의 급수 형태로 늘어나게 될 것이다. 따라서 이 k 는 hyper-parameter 이지만 3과 같은 작은 수를 채택한다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/seq2seq/seq2seq5.png?raw=true">
  <br>
  그림 5. Seq2Seq Embedding
</p>

또한 학습에서 단어를 입력 받을 때 위 그림과 같이 embedding 을 하게 된다. 

Seq2Seq 논문의 경우 이러한 워드 임베딩 차원을 1000차원으로 설정하고 16만개의 Vocab을 사용하였는데, 번역 Task 이기 때문에 각 나라의 언어마다 Vocab 혹은 Embedding 의 차원이 다르기 때문에 Encoder, Decoder 의 Embedding layer 차원은 바뀔 수 있다.

### 여러 Skill

Seq2Seq 논문의 저자는 입력 데이터를 그대로 읽는 것이 아니라 아예 순서를 뒤바꿔 입력을 하니 성능이 개선되었다고 이야기 한다.

예를 들어 ‘나는 밥을 먹는다.’ 이 문장을 ‘.먹는다 밥을 나는’ 라고 역전시켜 입력을 하면 학습이 더 빠랄지고 성능이 개선된다는 것이다.

그 이유는 무엇일까?  그 이유는 필연적으로 RNN 모델을 거치며 첫 단어의 정보의 영향력이 낮아지게 되는데, 시계열 데이터의 경우 앞쪽 데이터에 대한 정확한 예측이 더욱 중요하기 때문이라고 생각 할 수 있다.

또 다른 Skill 로는 Peeky 라는 ‘엿보기 구멍’이 있다. 앞서 살펴보았듯이 Decoder 의 경우 입력되는 정보는 오로지 Encoder 에서 산출된 Context Vector 밖에 없다.

이 정보는 최초 Decoder RNN Cell 에서만 온전히 얻게 되는데 이 정보를 이후 RNN Cell 에 대해서도 전달하는 것이다. 

하지만 이 경우 Parameter 의 수가 많아지기 때문에 계산량이 늘어난다는 단점이 있다.

### Code

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self. hid_dim = hid_dim #원-핫 벡터의 사이즈 
        self.n_layers = n_layers #RNN의 레이어 개수
        
        self.embedding = nn.Embedding(input_dim, emb_dim)#Dense vector 로 변환
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        # RNN 모델을 LSTM 으로 사용 후 Drop out 설정
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src len, batch_size]
        # 토큰화 돼서 나온 src는 3차원 텐서이다.
        # src 는 출발 문장 즉, 해석 할 문자. 학습을 효율적으로 하기 위해 배치 설정
        embedded = self.dropout(self.embedding(src))
        
        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # outputs = [src len, batch size, hid dim * n directions]
        #현재 단어의 출력 정보
        # hidden = [n layers * n directions, batch size, hid dim]
        # 현재까지 모든 단어의 정보
        # cell = [n layers * n directions, batch size, hid dim]
        #현재까지의 모든 단어의 정보
        
        return hidden, cell # 문맥 벡터로 사용
        
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim #output data의 voacab size 여기선 16만
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        # hidden layer 에서 output_dim 을 뽑는다.
        self.dropout = nn.Dropout(dropout_
        
    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # Decoder에서 항상 n directions = 1
        # 따라서 hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        
        # input = [1, batch size]
        input = input.unsqueeze(0)
        
        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # 현재 단어와 지금까지의 단어에 대한 정보를 같이 넣어서 매번 output 을 뽑아냄
        # output = [seq len, batch size, hid dim * n directions]
        #마지막 time-stamp/마지막 레이어의 은닉상태만 
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # 각 time-stamp/각 레이어들의 은닉상태와 cell state들의 리스트
        
        # Decoder에서 항상 seq len = n directions = 1 
        # 한 번에 한 토큰씩만 디코딩하므로 seq len = 1
        # 따라서 output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        
        # prediction = [batch size, output dim]
        prediction = self.fc_out(output.squeeze(0))
        # 이전까지의 정보가 담긴 output 을 뽑아낼 수 있도록 한다.
        
        return prediction, hidden, cell
        # 현재 출력 단어, 현재까지의 모든 단어의 정보, 현재까지의 모든 단어의 정보
        # 단어를 하나씩 얻는다.
```



위 코드에서는 앞서 그림에서의 Seq2Seq 구조에서 각 RNN 셀의 layer 를 n_layer 만큼 쌓는다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/seq2seq/seq2seq6.png?raw=true">
  <br>
  그림 6. Seq2Seq 2-layer
</p>

위 그림과 같이 Layer 의 개수가 2 개 라면 Layer 1의 경우 직전 time-stamp 로 부터 은닉상태(s) 와 cell state 를 받고, 이들과 embedded token 인 $y_t$​ 를 입력받아 새로운 은닉 상태와 cell state 를 만든다. (논문에서는 이 Layer 를 4개를 사용한다.)

Layer 2의 경우 Layer 1의 은닉상태(s)와 직전 time-stamp의 은닉 상태(s)와 cell state를 입력으로 받아 새로운 은닉 상태와 cell state를 만들어낸다.

 $(s_t^l,c_t^l) = DecoderLSTM^1(d(y_t),(s_{t-1}^l , c_{t-1}^l))$

 $(s_t^2,c_t^2) = DecoderLSTM^2(s_t^1,(s_{t-1}^2 , c_{t-1}^2))$​​

이후 여기서 나온 $f(s_t^2)$ 의 값을 예측값으로 내놓아 실제 값과의 차이를 계산하는 것이다.

```python
class Seq2Seq(nn.Module):
   def __init__(self, encoder, decoder, device):
       super().__init__()
       
       self.encoder = encoder
       self.decoder = decoder
       self.device = device
       
       # Encoder와 Decoder의 hidden dim이 같아야 함
       assert encoder.hid_dim == decoder.hid_dim
       # Encoder와 Decoder의 layer 개수가 같아야 함
       assert encoder.n_layers == decoder.n_layers
       
   def forward(self, src, trg, teacher_forcing_ratio=0.5):
       # src = [src len, batch size]
        # 2차원 리스트로 들어온다: 각 단어의 인덱스 정보가 들어온다.
        #0.5의 비율로 teacher_forcing 을 사용한다.
       # trg = [trg len, batch size]
       
       trg_len = trg.shape[0]
       batch_size = trg.shape[1]
       trg_vocab_size = self.decoder.ouput_dim
       
       # decoder 결과를 저장할 텐서
       outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
       
       # Encoder의 마지막 은닉 상태가 Decoder의 초기 은닉상태로 쓰임
       hidden, cell = self.encoder(src)
       
       # Decoder에 들어갈 첫 input은 <sos> 토큰
       input = trg[0, :]
       # 디코더의 모든 정보를 담기 위해 하나의 텐서 객체를 초기화 시킨다. 
       # target length만큼 반복
       # range(0,trg_len)이 아니라 range(1,trg_len)인 이유 : 0번째 trg는 항상 <sos>라서 그에 대한 output도 항상 0 
       for t in range(1, trg_len):
           output, hidden, cell = self.decoder(input, hidden, cell)
           outputs[t] = output
           
           # random.random() : [0,1] 사이 랜덤한 숫자 
           # 랜덤 숫자가 teacher_forcing_ratio보다 작으면 True니까 teacher_force=1
           teacher_force = random.random() < teacher_forcing_ratio
           
           # 확률 가장 높게 예측한 토큰
           top1 = output.argmax(1) 
           
           # techer_force = 1 = True이면 trg[t]를 아니면 top1을 input으로 사용
           input = trg[t] if teacher_force else top1
       
       return output
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uiform_(param.data, -0.08, 0.08)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

# <pad> 토큰의 index를 넘겨 받으면 오차 계산하지 않고 ignore하기
# <pad> = padding
trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss=0
    
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        
        # loss 함수는 2d input으로만 계산 가능 
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        # trg = [(trg len-1) * batch size]
        # output = [(trg len-1) * batch size, output dim)]
        loss = criterion(output, trg)
        
        loss.backward()
        
        # 기울기 폭발 막기 위해 clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss+=loss.item()
        
    return epoch_loss/len(iterator)
       
```

이후 가중치 초기화, 최적화 함수와 Loss를 적용시켜 모델을 학습 시키면 된다.

또한 Seq2Seq 아키텍쳐의 특징은 Teacher forcing 을 사용하는데, 디코더가 초반 단어의 예측에 실패하면 후에 뒤따르는 모든 단어에 대한 예측이 어긋 날 수 있다.

모델이 내뱉은 정답과 실제 정답간의 차이를 통해 모델이 업데이트 되는 과정에서  디코더의 예측을 다음 입력으로 사용하지 않고, 실제 정답 값을 다음 입력으로 사용한다.

