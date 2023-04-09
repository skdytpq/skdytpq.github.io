---
title: Relation-Based Associative Joint Location for Human Pose Estimation in Videos
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 머신러닝
key: 20230111
tags: 
  -머신러닝
use_math: true

---

# Temporal Distance Matrices for Squat Classification(Waseda squat dataset)

## Introduction

해당 연구는 비디오를 활용하여 스쿼트 자세에서의 올바른 자세가 무엇인지에 대해 탐지한다. 특정 환경인 Squat 자세에서의 탐지를 위해 본 연구진은 다양한 밝기와 각도, 환경 등을 다양하게 세팅하여 데이터를 생성하였다. 

이러한 다양한 세팅은 detection 의 난이도를 어렵게 하며, 가장 간단한 방법은 데이터 셋의 크기를 매우 방대하게 늘리는 것이다.

따라서 훈련 데이터와 다른 Input 에 대해 일반화 성능이 뛰어난 알고리즘 개발이 필요하며 본 논문의 저자는 temporal distance matrix 를 도입하여 해당 알고리즘을 개발하였다.

본 연구 방법은 다음과 같이 진행된다

- Extract 3D human pose from video (각 개별 Pixel의 Information 표현)
- Normalize limb length(참조하는 프레임의 특징에 대해 예민하기 때문에 Subject 별 특징에 덜 민감해지도록)
- compute distance matrix(each different joint)
  - 본 논문의 저자는 이러한 distance matrix가 장면정보, 개별 골격 길이 밑 reference frame 과 크게 독립적인 정보를 제공한다고 표현한다.
- CNN 1D convoltuion (본 논문이 저자는 CNN 통과 시 ResNet이 성능이 좋다고 얘기한다.)

### Contribution

1. A dataset for classification of good and various bad form of squats.

2. A method to assess the workout form from video by a feature extraction approach based on temporal distance matrices, which is robust to differences in scene, subject, and global translation and rotation. 
3. An experimental validation of our method, in which it outperforms existing video classification approaches.

해당 논문은 Assessment FeedBack 을 제공하는 것이 아닌 기존에 상정한 자세(Inward knee, Round back Warped back, Upwards head, Shallowness, Frontal Knee, Good Squat) 에 대한 Classification 을 진행한다.

## Dataset

한 비디오는 약 300 Frame, 10 초의 영상으로 이루어져있으며, 한 동영상에 약 3~5회의 스쿼트가 진행된다. 

## Method

본 연구는 Pose 의 information 을 이용함으로 진행된다. 이러한 Pose infromation 을 사용하는 경우 좀 더 Robustness하고 General 한 feature 를 생성할 수 있게 된다.

이러한 방식은 Background 가 매우 다양한 환경에서 유의미하게 작용할 수 있기 때문에 해당 논문의 저자는 Pose information을 사용하였으며, Video 전체의 infromation을 사용하는 것이 아니라, 3D pose information 만 사용하게 된다.

이러한 3D pose 를 translation, rotation 을 하여 Invariant 한 representation을 distance matrix로 추출하여 1D Convolution 으로 처리하게 한다.

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/waseda1.png?raw=true">
  <br>
  그림 1. 
</p>

전반적인 Process 는 다음 그림과 같다. 

우선적으로 3D pose 를 추출한 후 모든 Joint Pair 에 대한 Uclidiean Distance 를 계산한 후 삼각행렬을 만들어 Flatten 시킨다.

이후 추출된 vector를 시간 순으로 정렬한 후 1D Convolution 기반의 CNN 모델을 통과한 이후 Classification 이 진행된다.

### 3D pose estimation

Single person 의 단일 카메라 영상에서 19개의 Keypoint 를 출력하여 3D 좌표를 구한 후 SMPL 모델의 파라미터로 정규화를 진행한다.

SMPL을 사용하여 Hidden joint(가려진 관절 좌표) 탐지도 가능하게 한다.

### Normalization

Pose information 을 사용하기 위해 Subject 별 특징(Limb length)같은 것의 영향을 최소화 해야한다. 이러한 결과는 데이터의 Individual이 적을 때 generalize가 잘 안되는 현상을 야기한다.

따라서 이러한 문제를 해결하기 위해 본 논문의 저자는 여러 방식의 Normalization 방식을 적용하였다. Normaliztion 의 결과는 각 Limb 를 unit length 로 converting 함으로써 진행하게 된다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/waseda2.png?raw=true">
  <br>
  그림 2. 
</p>

해당 그림과 같이 진행하는데, Head 의 경우 Head 의 각 Joint 를 Unit length 로 변환한 것이며, Full 의 경우 모든 Joint 의 limb length 를 Unit length 로 변환 한 것이다

### Distance Matries

Normalized 됐다 하더라도 Pose가 Fully invariant는 아니라고 논문의 저자는 주장한다.

예를 들어 전체 3D pose 도 global reference frame에 영향을 받는다. 이러한 이유 때문에 본 연구는 3D pose 를 distance matrix로 convert 하여 gloabl translation, rotation 에 invariant 한 성질을 띄게 한다.

또한 해당 Representaiton은 각 Pose 에 대한 Unique representation 을 지니고 있다. (Angle 의 각도, 거리 등등)

$d_{i,j} = \sqrt{(x_i - x_j)^2 + (y_j - y_i)^2 +(z_i - z_j)^2}$ 

총 19개의 Joint 로 이루어져있으며, Matrix는 171개의 elements로 이루어져있으며 해당 Vector를 Flatten시켜 classification 모델에 투입하게 된다.

### Classification model

이후 Temporal 축에 대해 representation Matrix를 이어 붙인다. 그렇게 되면 y-axis 는 시간 축이 되는데, 연구진은 이 Matrix를 1D convolution 을 각 temporal 축에 진행하여 정보를 압축시킨다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/waseda3.png?raw=true">
  <br>
  그림 3. 
</p>

### Data argumentation

본 연구는 일반화 성능을 높이기 위해 선형 보간법을 통해 영상마다 고정된 Frame 을 추출하였다. (이 과정은 비디오의 속도를 조정하였으며 이 결과는 각 개인의 운동 수행 시의 시간 차이가 있는 와중에 성능 향상의 결과를 기대할 수 있다.)

또한 training video 에서 squat 를 수행할 때의 Canoncial pose 를 처음 프레임에 넣었는데, 이 과정을 샘플링 하기 위해 발목과 엉덩이의 각도 차이가 $150^{\degree}$ 를 넘는다면 Canoncial pose라고 간주하였다. (일종의 Standing position)

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/waseda6.png?raw=true">
  <br>
  그림 4. 
</p>



# Relation-Based Associative Joint Location for Human Pose Estimation in Videos

## Introduction

최근의 human pose estimation 은 크게 두 가지 Category 에서 어려움을 겪는다.

- Regression method : Image 에서 Body joint 를 직접적으로 regressing 하여 heatmap representation 으로 표현된 Guidence와 함께 예측한 좌표값을 추출

해당 Joint 를 추출하는 대부분의 모델은 Convolution Layer를 통해 내재적으로 Joint 간의 관계를 포착한다. 즉, 유동적이고 다양한 Joint 간의 관계를 잘 포착하지 못한다.

Pose에 대한 구조적인 정보는 Body 의 Topology 덕분에 관절 좌표를 잘 찾을 수 있게 해준다.  사람들이 보통 Motion을 수행할 때 신체의 각 Joint는 어떠한 관계를 갖고있으며 이러한 관계는 Human body에 대한 구조적 정보를 제공한다. 

예를들어 어떠한 관절 좌표가 가려져 있는 Image의 경우 Model 은 구조적 정보를 활용하여 Occluled 된 Joint 를 추론하게 된다. 따라서 이러한 구조적 정보를 학습하는 것은 HPE 에서 중요한 Task라고 할 수 있다. 

기존의 연구에선 Tree 혹은 Graph 기반의 모델이 Joint 간의 관계를 학습하기 위해 제언되었지만, 해당 모델들은 사전에 Hand-crafted 된 Structure 이 존재해야 하며 Motion의 다양성은 이렇게 사전에 표현해야 하는 Hand-Crafted 된 표현의 생성을 어렵게 한다.

따라서 Automatic하게 Joint relationship을 배울 수 있는 Scheme가 필요하다. 따라서 본 연구는 JRE(Joint relation extractor)를 추가하여 명시적으로 각 Joint 간의 관계를 학습할 수 있는 모듈을 추가하여 개발하였다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/JRE1.png?raw=true">
  <br>
  그림 5. 
</p>

또한 인접 프레임 간의 pose semantic feature를 활용하여 현재의 Frame 의 invisible joint 를 포착하게 했다.

동작 Motion 이 급작스럽게 바뀌거나 매우 동적인 Motion 을 취할 때 아무리 인접한 Frame이더라도 Joint 간의 관계가 불분명해질 수 있다. 

기존 선행 연구의 Pose kernel distillator(PKD)는 인접 프레임 간의 Joint를 포착하는 representation feature 를 생성하였지만, 해당 모델은 본 연구에서 제시하는 JRE 의 propagate과정을 무시한다. 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/JRE2.png?raw=true">
  <br>
  그림 6. 
</p>

Motion 이 크게 바뀌는 경우 Joint 의 변화가 급속적이어서 Joint 간의 관계가 불분명해 지는 현상이 발생할 수 있지만  Explicit 하게 나타내지 않은 모듈을 사용하고 Joint 를 Frame 단에서 계산하는 과정만을 취했을 때는 구조적인 정보를 담을 수 없기 때문에 유의하지 않을 수 있다.

따라서 JRPSP(joint relation guided pose semantics propagator) 모듈을 추가하여 JRE를 사용하여 Joint 의 구조적 정보를 담음과 동시에 Temporal 한 Pose의 정보를 포착하게 한다.

본 연구는 이러한 배경에서 나온  JRE, JRPSP 모델을 통틀어 Relation-based Pose Semantics Transfer Network(RPSTN)이라 칭하는 모델을 개발하게 된다.



### RPSTN

- Generate Heatmap for all body joint 
- JRE : Pose 의 Spatial 한 정보를 Joint relation 으로 파악하기 위해 사용
  - Pseudo heatmap을 받으며 이 Heatpmap에서 Joint 간의 관계 측정
- JRPSP : Pose Sequence 의 temporal dynamics 를 capture하기 위해 사용
  - 인접 프레임 간의 Relation based feauture 를 생성한다.

위 두 가지 모듈을 거쳐 RPSTN은 Invisible Joint의 추론을 가능하게 한다.

## Method

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/JRE3.png?raw=true">
  <br>
  그림 7. 
</p>

- Initial pose estimation : pose estimation 을 위한 heatmap 시리즈를 만들기 위해 사용
  - 기존 psedu heatmap $I_t$ 생성($I_t \in \R^{H\times W \times 3}$)
  - $P(I_t)$ (estimator)  : Frame 마다의 Joint relation에 대한 Heatmap생성 
    - 코드 상에서 해당 부분은 Resnet 을 통과하여  Joint chennel 축으로 맞춰진다.
  - $M'_t = BN(R(P(I_t))), t=1$ : First frame의 heatmap

- Historical pose semantic learning :  각 Frame 당 Feature를 생성하는 과정, Pose의 전체적인 정보 Feature
  - $X_t^a = Conv_a (f_t \oplus M'(t))$: feature 와 joint 정보를 concat 시킴 $1 \times 1\times C$ Convolution 진행
  - 따라서 $X_t^a$ 는 appearance 뿐만 아니라 Joint 에 대한 정보도 포함한다 판단 가능
- Pose semantic propagation and global matching : $t $ 시점의 전체 representation feature 는 $t+1$ 시점의 frame으로 들어가게 된다.
  - Global Matching mechanism : 인접 프레임 간의 지역 유사성을 찾아 pose information 으로 transfer 하는 과정
  - JRPSP : $X_t^a$ 를 Input 으로 받아 전체 Pose에 대한 Infromation 을 받은 후 JRE 의 Input을 만듬($S$ : several convolution)
  -  $M_{t+1} = Conv_d (S(X_t^a) \otimes F(I_{t+1}))$ : pseudo heatmaps 가 생성되는 방식 $\otimes$ : dynamic convolution 
  - $Conv_d$ : 각 Joint 에 대한 Convolution 진행 채널 축 $K $ 가 joint 의 개수가 됨(이전 Frame 의 Feature map 을 이후 Frame 에 바로 활용하는 것이 아닌 JRE 모듈을 통해 간접적으로 정보를 뽑음)
- Relation modeling in the current frame : $M'_{t+1} = BN(R(M_{t+1}))$ $M_{t+1}$ 을 받아 최종적 Joint heat map $M'_{t+1}$ 을 생성 
  - Loss 는 $M_t , \hat{M_t}$ 와의 L2 norm 을 구한 후 계산된다.

### JRE Module

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/JRE4.png?raw=true">
  <br>
  그림 8. 
</p>

JRE Module의 결과 $M$ 은 joint heatmap 인 $F(i_{t+1})$ 과 semantic feature $S(X_t^a)$ 정보 모두 포함하게 된다.

우선 $M , M^T$ 행렬(K개의 Joint 별 관계를 나타낸 Heatmap) 을 dot product 한 후 soft max 함수를 통과하여 $K\times K$ Matrix를 만든다.

만들어진 행렬 ($W_r$)은 각 Joint 의 Correlation 을 담고있다. 이 정보는 기존 원본 정보 $M$ 과 Dot product 형식으로 계산되게 된다.

이후 기존 $M$ 을 $1 \times 1$ Convolution 을 추가적으로 진행해 모든 Joint 에 대한 전체 정보를 담는다.(K 축으로 진행하면 Joint 마다의 특정 위치에서의 Receptive field 생성) $G : H' \times W' \times K$

$W_r$ 은 2개의  joint 에 대한 각 상관관계를 나타내며 $G$는 각 Joint 에 대해 Regional information은 포함한다. 즉, $W_r$ 을 G 행렬에 important joint 를 강조할 수 있는 보조 장치로 사용할 수 있다.(어떤 Joint 가 다른 Joint B 와 상관관계가 높다면 값이 크게 나오며 반대는 더 작게 나올 것) 

위의 과정은 Spatial 한 정보(각 Joint에 대한)와 Joint 간의 관계를 모두 포함한 Feature 가 형성 됨 $\bar{G}$ 생성,

해당 JRE Modeling 은 positional information 의 정보 학습이 끝난 후, 계속 갱신되는 것은 오히려 모델의 약점이 될 수 있다. 

그래서 JRE Model 의 학습이 끝나면 해당 모델을 None-Local-Block 으로 대체하게 되는데, 해당 모델은 Joint level 에서 두 관절 간의 관계를 매핑하는 것이 아니라, 각 픽셀간의 정보의 유사도를 구하게 된다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/JRE5.png?raw=true">
  <br>
  그림 9. 
</p>
또한 Propagation 을 위해 DKD 모듈을 가져와 사용하였다.

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202023-01-12%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.11.30.png?raw=true">
  <br>
  그림 10. 
</p>

## Experiment 

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/JRE_tabe.png?raw=true">
  <br>
  그림 11. 
</p>


Sota 달성



## 결론

- Waseda Dataset : 3d pose estimation 이 선행되어 assesment를 진행하며, Initial point 에 대해 특정 각도를 넘는 임계점을 설정한다. , Joint 정보를 Distance Matrix를 활용하여 Sub-information으로 사용
- JRE : Hidden Joint 를 잘 Inference 하기 위해 Joint relationship 을 활용한 Structed information을 사용한다. 

위 두가지 Contribution의 활용

- 우선적으로 2D pose-estimation을 진행한 후 3D skeleton generate를 위해 Gernerative 모델 활용.
  - 3D Labeling 된 Fitness dataset 이 없기때문에 생성 모델을 Unsupervised Learning을 통해 학습
  - 해당 학습 과정에서 Skeleton의 구조 정보를 파악하기 위해 JRE 모듈 사용 
  - 3D skeleton 생성 모델 설계 후 Assesment를 위한 Video sampling 과정에서 운동 시작 Degree 기반으로 Initial point 추론
  - 이후 Classification 을 통해 올바른 자세 분류(FitnessAQA,waseda squat)를 위해 Distance Matrix를 활용한 Classification 모듈 활용(OR graph based model) 
  - 올바른 자세와 올바르지 않은 자세에 대한 Skeleton 각도의 범위 Hueristic 하게 분석(?)및 Assesment 수행

 