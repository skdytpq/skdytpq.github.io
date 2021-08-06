---
layout: post
title:  "핸즈온 머신러닝 2챕터 실습 코드"
---

I hope you like it!
```python
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D # 3차원 시각화
import seaborn as sns

import arrow
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
```


```python
import sklearn.linear_model
import os
import tarfile
import urllib
```


```python
DOWNLOAD_ROOT='https://raw.githubusercontent.com/ageron/handson-ml2/master/'
HOUSING_PATH = os.path.join('datasets','housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

def fetch_housing_data(housing_url=HOUSING_URL, housing_paht = HOUSING_PATH):
    os.makedirs(housing_path,exis_ok=True)
    tgz_path = os.path.join(housing_path,'housing.tgz')
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```


```python
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)



housing = load_housing_data()
```


```python
housing.isna().sum()
```




    longitude               0
    latitude                0
    housing_median_age      0
    total_rooms             0
    total_bedrooms        207
    population              0
    households              0
    median_income           0
    median_house_value      0
    ocean_proximity         0
    dtype: int64




```python
housing.hist(bins=50,figsize=(20,10))
plt.show()
```


    
![png](output_5_0.png)
    



```python
def split_test_data(data,test_ratio):
    np.random.seed(42)
    shuffled_indicies=np.random.permutation(len(data))# 데이터의 길이에서 랜덤하게 셔플된 arrange를 반환시킴 integer을 array로 만듦
    test_set_size=int(len(data)*test_ratio)
    test_indicies=shuffled_indicies[:test_set_size]
    train_indicies=shuffled_indicies[test_set_size:]
    return data.iloc[train_indicies],data.iloc[test_indicies]
```


```python
train_set,test_set=split_test_data(housing,0.2)
```


```python
len(train_set)
#len(test_set)
```




    16512



테스트 세트와 훈련 세트가 시작할 때 마다 섞이지 않게 조정해야 한다. \
각 샘플마다 고유한 해시값을 계산하고 최댓값의 20% 이하인 샘플만 테스트 세트로 보내기. \
데이터셋이 갱신되더라도 테스트 세트가 동일하게 유지.\
이렇게 된다면 새 샘플의 경우 해시값이 기존 최댓값 이상이더라도 이전 훈련세트에 존재하던 샘플은 새로운 테스트 세트에 포함되지
않을 것





```python
from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier))& 0xffffffff < test_ratio*2**32
def split_train_test_by_id(data,test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]
```


```python
train_set, test_set = split_train_test_by_id(housing.reset_index(),0.2,'index')
#인덱스를 고유식별자로 한다면 데이터는 merge가 아닌 concat으로 붙여야함 so 고유하고 변하지 않는 특성을 인덱스로 설정하여
#ID 를 만들어 테스트 셋,훈련 셋을 구분할 수 있다.
```


```python
housing_with_id=housing.reset_index()
housing_with_id['id'] = housing.longitude*1000+housing['latitude']
train_set,test_set= split_train_test_by_id(housing_with_id,0.2,'id')
```


```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing , test_size = 0.2 , random_state = 42)
#특정 열을 기준으로 나누고싶다면 startift=True 인자 사용
```


```python
housing['income_cat']=pd.cut(housing.median_income,bins=[0.,1.5,3.0,4.5,6,np.inf]\
                            ,labels=list(range(1,6)))#qcut은 동일비율로 5개 집단을 나누는 것
housing['income_cat'].hist()# 소득분위별 계층 샘플링을 진행할 준비
```




    <AxesSubplot:>




    
![png](output_14_1.png)
    



```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1 ,test_size = 0.2 , random_state = 42)#시행될 때마다 데이터셋이 섞이지 않도록 난수 지정
# 각 층의 비율을 고려하여 무작위로 훈련 테스트 세트 분할해주는 indices(지수) 반환
#n_split: 한개의 세트만 분할한다. 
#여기서의 split 은 일종의 함수 역할. 층화 분리를 하겠다는 선언
for train_index , test_index in split.split(housing, housing['income_cat']):#income_cat열을 기준으로 분리해줌 random
# housing 안에서 income_cat이라는 labels 을 바탕으로 계층의 비율을 고려하여 무작위로 추출하고 그 indices를 반환하여 loc으로 묶음
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    a=train_index
strat_train_set.income_cat.value_counts()/len(strat_train_set)
split.split(housing,housing.income_cat)
```




    <generator object BaseShuffleSplit.split at 0x0000019D3595FE40>




```python
split.split(housing,housing.income_cat) # 기존데이터에서 incom_cat 의 비율을 그대로 가져와서 샘플링을 한 것
```




    <generator object BaseShuffleSplit.split at 0x0000019D35937970>




```python
for set_ in (strat_train_set , strat_test_set):
    set_.drop('income_cat' , axis = 1 ,inplace = True)
```


```python
strat_train_set
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.00</td>
      <td>1568.00</td>
      <td>351.00</td>
      <td>710.00</td>
      <td>339.00</td>
      <td>2.70</td>
      <td>286600.00</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.00</td>
      <td>679.00</td>
      <td>108.00</td>
      <td>306.00</td>
      <td>113.00</td>
      <td>6.42</td>
      <td>340600.00</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.00</td>
      <td>1952.00</td>
      <td>471.00</td>
      <td>936.00</td>
      <td>462.00</td>
      <td>2.86</td>
      <td>196900.00</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.00</td>
      <td>1847.00</td>
      <td>371.00</td>
      <td>1460.00</td>
      <td>353.00</td>
      <td>1.88</td>
      <td>46300.00</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.00</td>
      <td>6592.00</td>
      <td>1525.00</td>
      <td>4459.00</td>
      <td>1463.00</td>
      <td>3.03</td>
      <td>254500.00</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6563</th>
      <td>-118.13</td>
      <td>34.20</td>
      <td>46.00</td>
      <td>1271.00</td>
      <td>236.00</td>
      <td>573.00</td>
      <td>210.00</td>
      <td>4.93</td>
      <td>240200.00</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>12053</th>
      <td>-117.56</td>
      <td>33.88</td>
      <td>40.00</td>
      <td>1196.00</td>
      <td>294.00</td>
      <td>1052.00</td>
      <td>258.00</td>
      <td>2.07</td>
      <td>113000.00</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>13908</th>
      <td>-116.40</td>
      <td>34.09</td>
      <td>9.00</td>
      <td>4855.00</td>
      <td>872.00</td>
      <td>2098.00</td>
      <td>765.00</td>
      <td>3.27</td>
      <td>97800.00</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>-118.01</td>
      <td>33.82</td>
      <td>31.00</td>
      <td>1960.00</td>
      <td>380.00</td>
      <td>1356.00</td>
      <td>356.00</td>
      <td>4.06</td>
      <td>225900.00</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>15775</th>
      <td>-122.45</td>
      <td>37.77</td>
      <td>52.00</td>
      <td>3095.00</td>
      <td>682.00</td>
      <td>1269.00</td>
      <td>639.00</td>
      <td>3.58</td>
      <td>500001.00</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 10 columns</p>
</div>




```python
housing=strat_train_set.copy()
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,
            s=housing['population']/100,label='인구',c='median_house_value',cmap=plt.get_cmap('jet')
            ,colorbar=True, sharex=False,figsize=(10,7))
plt.ylabel('위도')
plt.xlabel('경도')
plt.show()
#s로 원의 반지름, c 인 color 의 분류는 중간 주택 가격이며 cmap으로 jet의 팔레트를 불러옴
#주택가격은 지역과 인구밀도에 관련이 매우 크다. 
```


    
![png](output_19_0.png)
    



```python
corr_matrix = housing.corr() # 상관관계 조사
corr_matrix['median_house_value'].sort_values(ascending=False)
```




    median_house_value    1.00
    median_income         0.69
    total_rooms           0.14
    housing_median_age    0.11
    households            0.06
    total_bedrooms        0.05
    population           -0.03
    longitude            -0.05
    latitude             -0.14
    Name: median_house_value, dtype: float64




```python
sns.pairplot(housing)
```




    <seaborn.axisgrid.PairGrid at 0x1f952d97dc0>




    
![png](output_21_1.png)
    



```python
#중간소득과 중간주택가격과의 관계는?
housing.plot(kind='scatter', x ='median_income', y ='median_house_value',alpha=0.1)
# 점이 위로 향하는 경향이 있음을 알 수 있고 500000의 수평선을 가격 제한값이 잘 보인다. 
```




    <AxesSubplot:xlabel='median_income', ylabel='median_house_value'>




    
![png](output_22_1.png)
    



```python
housing['room_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']
```


```python
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
```




    median_house_value          1.00
    median_income               0.69
    room_per_household          0.15
    total_rooms                 0.14
    housing_median_age          0.11
    households                  0.06
    total_bedrooms              0.05
    population_per_household   -0.02
    population                 -0.03
    longitude                  -0.05
    latitude                   -0.14
    bedrooms_per_room          -0.26
    Name: median_house_value, dtype: float64




```python
housing=strat_train_set.drop('median_house_value',axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
#중간 주택 가격으로 레이블을 나눴기 때문에 훈련 세트에서 label을 제외한 나머지에 대해 정제를 수행하기 위해 drop 으로 중간 주택
#가격을 제외시킴. 그 후 housing_labels로  원하는 y 값을 받음 데이터의 순서는 변하지 않기에 housing과 labels는 대응된다.
```


```python
# 데이터 정제를 손쉽게 할 수 있게 도와주는 imputer객체
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
housing_num = housing.drop('ocean_proximity',axis=1)
imputer.fit(housing_num)#imputer 객체에 fit을 사용 하여 훈련 데이터에 적용 가능
imputer.statistics_#imputer 객체를 훈련시켜서 median 값 지정 해줌
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns= housing_num.columns , index = housing_num.index)
housing_tr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.00</td>
      <td>1568.00</td>
      <td>351.00</td>
      <td>710.00</td>
      <td>339.00</td>
      <td>2.70</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.00</td>
      <td>679.00</td>
      <td>108.00</td>
      <td>306.00</td>
      <td>113.00</td>
      <td>6.42</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.00</td>
      <td>1952.00</td>
      <td>471.00</td>
      <td>936.00</td>
      <td>462.00</td>
      <td>2.86</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.00</td>
      <td>1847.00</td>
      <td>371.00</td>
      <td>1460.00</td>
      <td>353.00</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.00</td>
      <td>6592.00</td>
      <td>1525.00</td>
      <td>4459.00</td>
      <td>1463.00</td>
      <td>3.03</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6563</th>
      <td>-118.13</td>
      <td>34.20</td>
      <td>46.00</td>
      <td>1271.00</td>
      <td>236.00</td>
      <td>573.00</td>
      <td>210.00</td>
      <td>4.93</td>
    </tr>
    <tr>
      <th>12053</th>
      <td>-117.56</td>
      <td>33.88</td>
      <td>40.00</td>
      <td>1196.00</td>
      <td>294.00</td>
      <td>1052.00</td>
      <td>258.00</td>
      <td>2.07</td>
    </tr>
    <tr>
      <th>13908</th>
      <td>-116.40</td>
      <td>34.09</td>
      <td>9.00</td>
      <td>4855.00</td>
      <td>872.00</td>
      <td>2098.00</td>
      <td>765.00</td>
      <td>3.27</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>-118.01</td>
      <td>33.82</td>
      <td>31.00</td>
      <td>1960.00</td>
      <td>380.00</td>
      <td>1356.00</td>
      <td>356.00</td>
      <td>4.06</td>
    </tr>
    <tr>
      <th>15775</th>
      <td>-122.45</td>
      <td>37.77</td>
      <td>52.00</td>
      <td>3095.00</td>
      <td>682.00</td>
      <td>1269.00</td>
      <td>639.00</td>
      <td>3.58</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 8 columns</p>
</div>




```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat = housing[['ocean_proximity']]
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat) #등간척도로 범주형 변수 숫자로 변환.
ordinal_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]




```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot #출력이 사이파이 희소행렬이다. 0이 많기 때문에 메모리 낭비가 심할 수 있다.
#one hot 인코딩으로 나머지 특성에 대해서 0으로 보는 것
#따라서 희소행렬로 본 후에 열 번호가 같다면 같은 범주형 데이터라고 볼 수 있다.
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>




```python
cat_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]




```python
from sklearn.base import BaseEstimator , TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X,y=None):
        return self
    def transform(self,X):
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        population_per_household = X[:,population_ix]/X[: , households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household, population_per_household,bedrooms_per_room]
        
        else:
            return np.c_[X,rooms_per_household,population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)
#컬럼에서 클래스와 메소드를 만들어서 직접 생성
#class 에서 add_bedromms라는 하이퍼 파라미터를 갖는데, 전체적으로 데이터의 값에서 transform을 한다면 열을 새로 추가시켜 주는 것
#새로운 특성을 만드는 간단한 클래스
#하이퍼파라미터의 추가로 이 것이 도움이 되는 특성인지 빠르게 확인 가능 함
```


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([ ('imputer', SimpleImputer(strategy='median')),
                        ('attribs_adder',CombinedAttributesAdder()),
                        ('std_scaler',StandardScaler()),
                        ])
housing_num_tr=num_pipeline.fit_transform(housing_num)
# 수치형 컬럼에 대한 임퓨트, 열 추가, 정제 진행 후 housing_num 에 저장
```


```python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num',num_pipeline, num_attribs),
    ('cat',OneHotEncoder(),cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
# 전체적인 파이프라인을 통해 카테고리를 원 핫 인코더로 트렌스폼 함
# 한번에 num인 것과 cat인 것의 파이프라인을 진행한다.
```




    array([[-1.15604281,  0.77194962,  0.74333089, ...,  0.        ,
             0.        ,  0.        ],
           [-1.17602483,  0.6596948 , -1.1653172 , ...,  0.        ,
             0.        ,  0.        ],
           [ 1.18684903, -1.34218285,  0.18664186, ...,  0.        ,
             0.        ,  1.        ],
           ...,
           [ 1.58648943, -0.72478134, -1.56295222, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.78221312, -0.85106801,  0.18664186, ...,  0.        ,
             0.        ,  0.        ],
           [-1.43579109,  0.99645926,  1.85670895, ...,  0.        ,
             1.        ,  0.        ]])




```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
```




    LinearRegression()




```python
some_data = housing.iloc[:5,]
some_labels = housing_labels.iloc[:5,]
some_data_prepared = full_pipeline.transform(some_data)
print(lin_reg.predict(some_data_prepared))
print(list(some_labels))
```

    [210644.60459286 317768.80697211 210956.43331178  59218.98886849
     189747.55849879]
    [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]
    


```python
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```




    68628.19819848923




```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
```




    DecisionTreeRegressor()




```python
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels , housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
#housing set에 대해서 좀 과대적합 된 것 같다. 
```




    0.0




```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg , housing_prepared , housing_labels,
                        scoring='neg_mean_squared_error',cv=10)
tree_rmse_scores = np.sqrt(-scores)
#k-fold-cross-validation
# cv 만큼 휸련 세트를 서브셋으로 무작위 분할한 후 결정트리 모델을 10번 훈련하고 평가한다. 9개 폴드가 훈련, 1개가 평가에 사용
```


```python
def display_scores(scores):
    print('점수 :', scores)
    print('평균 :',scores.mean())
    print('표준편차 :',scores.std())
    
display_scores(tree_rmse_scores)
```

    점수 : [67669.73816437 67362.61970456 70229.34328631 68299.62160307
     71291.36213287 75577.59634532 72668.53981039 70226.23940832
     77010.86953522 69438.29976359]
    평균 : 70977.42297540254
    표준편차 : 3085.3445235048844
    


```python
lin_scores = cross_val_score(lin_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
#선형회귀에서의 점수 측정
```

    점수 : [66782.73843989 66960.118071   70347.95244419 74739.57052552
     68031.13388938 71193.84183426 64969.63056405 68281.61137997
     71552.91566558 67665.10082067]
    평균 : 69052.46136345083
    표준편차 : 2731.6740017983493
    


```python

```
