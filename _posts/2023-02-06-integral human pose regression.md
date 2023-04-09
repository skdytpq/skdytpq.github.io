---
title: Integral Human Pose Regression
sidebar:
  nav: docs-ko
toc: true
toc_sticky: true
categories:
  - 딥러닝
key: 20230206
tags: 
  - 딥러닝
use_math: true

---

# Integral Human Pose Regression

## Integral Pose error

$H_k$ 라는 히트맵이 $k$번째 joint 에서 나왔다고 할 때 히트맵의 각 원소는 해당 위치에서 joint 가 위치할 확률 분포를 포함하게 된다.

최종적으로 Heatmap을 통해 나오는 K번째 원소의  Joint 좌표는 

$J_k = argmax_p H_k(p)$ 의 식에 따라 나올 수 있다.

그러나 이러한 방식은 크게 두가지 단점이 있다.

- Non-Differentiable : 히트맵에서 각 joint 를 추출하는 방식은 단순 인덱싱이기 때문에 미분 불가능하며 이후에 joint 를 통해 학습을 하는 과정을 불가능하게 하며 end-to-end 모델이 가능하지 않다. 따라서 지도학습은 Heatmap 단 까지 밖에 진행되지 않는다.

- Quantization error : 히트맵의 resolution 은 input image 의 resolution 보다 매우 낮다. 지금 연구하고 있는 unsupervised 3d pose estimation 의 경우 input 이미지는 대략 480 X 480 의 resolution 이지만 output 으로 뽑아내는 Heatmap 의 resolution 은 64 X 64 이다.

  이러한 문제는 Down sampling 과정에서 발생하게 된다. 

  따라서 Joint 예측의 정확성은 한정된 64X64 원소들에 의해 결정되며 이 부분은 모델의 정확성을 해칠 수 있다.

  해당 부분은 내가 연구하는 문제에서 매우 중요한 Issue 가 될 수 있다. 기본적으로 Heatmap Regression 이 기반이 되는 JRE 모듈을 통해 2D Heatmap을 생성하며 해당 joint 를 GT로 설정하여 3D keypoint를 추출하게 되는데 Heatmap to Joint 과정에서 accuracy 가 엉망진창이라면 GT 에 영향을 많이 받는 이후 3D keypoint 모델의 학습은 엉망이 될 수 있다.

  물론 size 가 큰 Heatmap 을 설정할 수 있지만 이 경우 computing 자원의 문제가 생기게 된다. (급수적으로 증가)

Regression methods 는 Heatmap based method에 비해 두가지 장점이 존재한다. 

직접적인 Joint regression 을 통해 End-to-End 모델 학습이 가능하다. 

두번째로 output이 quantization 을 통해 나온 discreate한 값이 아닌 수치값이기 때문에 quantization을 통해 joint 를 구하는 heatmap based 방식에서 발생하는 accuracy 문제를 해결할 수 있다.

본 논문은 Heatmap based model 의 output 을 joint coordinates로 transform하는 방식을 소개하며 heatmap, regression based 모델 간의 gap을 줄인다. 이러한 방식은 실용적이고 원칙적인 이점을 제공한다.

논문에서 제시한 식은 다음과 같이 Expectation 을 구하는 적분 식이다.
$J_k = \int_{p \in \O} p \cdot \tilde{H_k}$

해당 식에서 $\tilde{H_k}$는 Normalized 된 heatmap이며 $\O$ 가 정규화를 수행한 domain이다.

정규화의 결과로 $\tilde{H_k}$ 안에 있는 모든 원소는 non-negative 상태가 되며 합계는 1이 된다. 

이렇게 정규화를 진행한 $\tilde{H_k}$의 원소는 Softmax 함수 취급이 가능하며 해당 식은

$\tilde{H_k} = \frac{e^{\tilde{H_k}(p)}}{\int _{q \in \O}e^{H_k}(q)}$  로 표현될 수 있다. 

앞서 제시한 $J_k$  (연속 확률 분포의 pdf)식을 Discreate하게 표현하면

$J_k = \sum_{p_z = 1}^D \sum _{p_y = 1} ^ H \sum_{p_x = 1}^W p \cdot \tilde{H_k}(p)$ 형태로 표현할 수 있으며 D 의 경우 1로 두면 2D heatmap 이 된다.

이런 방법을 통해 Heatmap based 모델의 output 이 joint 형태로 바뀔 수 있게 된다. 

본 연구는 해당 방법론을 **integral pose regression**이라 칭한다. 

해당 방법론은 heatmap based approach와 regression based approach의 이점을 결합하였으며 미분 가능하고 end-to-end 학습이 가능하다는 장점을 가지고있다.

또한 해당 방법은 non parametric하기 때문에 inference 과정에서 시간적 소모가 있지 않으며 output자체가 continous하기 때문에 quantization problem의 문제를 갖지 않고 원본 이미지 size 로 곧바로 resize시키면 원본 이미지에 맞는 joint 를 표현할 수 있게 된다. 

## 평가지표

- PCK : 특정 threshold 보다 detected-true 간의 차이가 작다면 correct 로 간주하는 평가 지표이다.
  기본적으로 PCK @ 0.2 는 threshold 가 0.2 * torso diameter 로써, 여기서 torso 는 사람의 몸통(팔다리를 제외한 몸 부분)이다. 
  - PCKh : PCK 의 임계값은 인물의 머리 크기에 따라 다르며 PCKh@0.5의 경우 머리 크기의 0.5 를 임계값으로 설정한다.