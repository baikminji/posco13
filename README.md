# 생활코딩 - Tensorflow

## 1. 머신러닝

- 기계를 학습시켜서 인간의 판단능력을 기계에게 위임하는 것

<br>

## 2. 텐서플로우로 해결하려는 문제는 지도학습 영역의 회귀/분류
- 회귀 : 숫자로 된 결과를 예측
- 분류 : 범주형 문제를 예측

<br>

## 3. 머신러닝의 문제를 해결하는 알고리즘

- Decision Tree/Random Forest/KNN/SVM/Neural Network 등이 있음
- 인공신경망을 깊게 쌓으면 딥러닝

<br>

##  4. 딥러닝/머신러닝/인공지능은 서로 다른 개념

<br>

## 5. 지도학습의 과정

1. 과거의 데이터 필요(원인은 독립변수, 결과는 종속변수)
2. 모델의 구조를 만듦(원인은 input, 결과는 output)
3. 데이터로 모델을 학습(fit, 모델이 데이터를 가지고 학습)
4. 모델을 이용

<br>

## 6. 위의 순서대로 코드 적용

1) 독립변수/종속변수 나누기

```python
import pandas as pd
import tensorflow as tf

lemonadefile = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(lemonadefile)

indelemon = lemonade[['온도']]
depelemon = lemonade[['판매량']]
print(indelemon.shape, depelemon.shape)
```

2) 모델의 구조 만들기

```python
X = tf.keras.layers.Input(shape=[1])
Y= tf.keras.layers.Dense(1)(X)
model=tf.keras.models.Model(X,Y)
model.compile(loss='mse')

X = tf.keras.layers.Input(shape=[1])
# lemonade 데이터에서 독립변수는 '온도'라는 칼럼 1개라서 shape=[1], 1이라고 적음

Y= tf.keras.layers.Dense(1)(X)
# lemonade 데이터에서 종속변수는 '판매량'칼럼 1개라 Dense(1), 1이라고 적음
```

3) 데이터로 모델 학습하기

```python
model.fit(indelemon, depelemon, epochs=1000)
```

4) 모델을 이용해 값을 예측하기

```python
print("Predictions:", model.predict(([15]))
```



## 7. 보스턴 집값 예측

### 라이브러리 사용

```python
import tensorflow as tf
import pandas as pd
```

<br>

#### 1. 과거의 데이터를 준비합니다.

```python
파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
보스턴 = pd.read_csv(파일경로)
print(보스턴.columns)
보스턴.head()

# boston 데이터에서는 1~13번째 열이 14번째 열에 영향을 줌
# 따라서 원인에 해당하는 독립변수는 1~13번째 열이고 결과에 해당하는 종속변수는 14번째 열임.

# 독립변수, 종속변수 분리 
독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
            'ptratio', 'b', 'lstat']]
종속 = 보스턴[['medv']]
print(독립.shape, 종속.shape)
```

<br>

#### 2. 모델의 구조를 만듭니다

```python
X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')
```

<br>

#### 3.데이터로 모델을 학습(FIT)합니다.

```python
model.fit(독립, 종속, epochs=1000, verbose=0)
model.fit(독립, 종속, epochs=10)
```

<br>

#### 4. 모델을 이용합니다

```python
print(model.predict(독립[5:10]))

# 종속변수 확인
print(종속[5:10])

# 모델의 수식 확인
print(model.get_weights())
```

---

보스턴 데이터를 가지고 13개의 입력을 받아 1개의 출력을 만듦
이를 수식으로 표현하면 `y=w1x1+w2x2+.......+w13x13+b` 이렇게 나타냄

`Y=tf.keras.layers.Dense(1)(X)`
dense layer에서 위와 같은 수식을 만드는 것이고 모델은 수식에서 들어갈 w와 b의 값을 찾는 과정을 거침


뉴런은 두뇌안에 있는 세포의 이름이고 인공신경망에서 뉴런 역할을 하는게 모형과 수식
이 모형에는 퍼셉트론(Perceptron)이라는 이름이 있음. 수식에 등장하는 w는 가중치(weight), b는 편향(bias)임

만약 종속변수 12개에 독립변수 2개인 경우면 입력과 출력은 12개, 2개임. 이럴 땐 수식이 2개가 필요한데, 퍼셉트론이 두개가 병렬로 연결되어 있는 것. 이런 경우 w가 12개 *2(한 퍼셉트론 당 w가 12개, 퍼셉트론이 2개이므로 2배) + b(bias)가 1개*2 해서 총 26개의 답을 찾아야 함
