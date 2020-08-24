# TIL / ML-DL
Machine Learning and Deep Learning
<br>

> 머신러닝을 구현 가능한 프로그래밍 언어 (Python or R 실습)  
> 기초 수학 과목: 미적분학, 선형대수학, 확률 및 통계학

프로그래밍언어 중 Python, R같은 경우 속도가 C, Java보다 빠르진 않지만, 기존 머신러닝에 쓰이던 Matlab 보다는 빠르며, 코드로 구현하기 쉽다는 장점이 있기 때문에 최근 많이 사용되고 있다.

---

## Chapter1. Linear Regression



## Chapter2. Linear Classification Model



## Chapter3. non-Linear Classification Model



## Chapter4. Clustering



## Chapter5. Deep Learning & Multi-layer Perceptron



## Chapter6. Convolutional Neural Networks



## Chapter7. Recurrent Neural Networks



## Chapter8. Autoencoders and Generative Models







## PJT -  Heart Failure Prediction using ML/DL and NN

This project will focus on predicting Heart Failure(disease) using ML/DL and NN

the reason  why i focus on heart disease is it has the highest rank of domestic death rates except `Tumor Neoplasia`  such like cancer, choronicle disease.

- this project is utilize a dataset of 303 patients (in kaggle dataset - `출처남길것`)

- i'll be using some common Python libraries, such as `pandas`, `numpy`, and `matplotlib`.

  - for the machine learning side of this project, i'll be using	
    NN , ML, DL-  `sklearn`, `keras`

  - for heatmap visualization - `seaborn` 



### Steps

- Import data and modules
- Train/Test-split in scikit-learn
- Normalization
- Adding Dropout
- Adding Weight Regularization



### Result



---

### Process

1. **Importing the Dataset**

   1) import modules

   ```python
   # common library for DataFrame
   import numpy as np
   import pandas as pd
   import warnings # suppress matplotlib warning
   warnings.filterwarning('ignore') 
   
   # Visualization & plotting
   from sklearn.metrics import plot_precision_recall_curve
   import seaborn as sns
   import matplotlib.pyplot as plt
   
   # for data split and training
   from sklearn.model_selection import train_test_split
   
   # for the modelling
   from sklearn.svm import SVC
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
   from sklearn.neighboros import NearestNegibors # Knn
   
   # for building and training Neural Networks
   from keras.models import Sequential
   from keras.layers import Dense
   from keras.optimizers import Adam
   from keras.layers import Dropout # normalization
   from keras import regularizers
   
   # for evaluation
   from sklearn.metrics import precision_recall_curve
   from sklearn.metrics import accuracy_score, f1_score
   from skelarn.metrics import confusion_matrix, roc_curve, auc
   ```

   

   2) load data

   ```python
   df = pd.read_csv('./dataset/heart.csv')
   df.head()
   ```

   ![heart_df_head](.\Dataset\images\heart_df_head.jpg)

   ```python
   print(df.shape) # (303, 14)
   ```

   

   >## Data Description
   >
   >1. age : age in years
   >
   >2. sex : (1 = male; 0 = female)
   >
   >3. cp : chest pain type
   >
   >    (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
   >
   >4. trestbps : resting blood pressure (in mm Hg on admission to the hospital)
   >
   >5. chol : serum cholestorla in mg/dl
   >
   >6. fbs : (fasting blood sugar & gt; 120mg/dl) (1 = true; 0 = false)
   >
   >7. restecg : resting electrocardiographic results
   >
   >8. thalach : maximum heart rate achieved
   >
   >9. exang : exercise induced angina (1 = yes; 0 = no)
   >
   >10. oldpeak :ST depression induced by exercise relative to rest
   >
   >11. slope : the slope of the peak exercise ST segment
   >
   >12. ca : number of major vessels (0-3) colored by flourosopy
   >
   >13. thal : 3 = normal; 6 = fixed defect; 7 = reversable defect
   >
   >14. target : 1 or 0

   

   3) data clearing

   ```python
   # remove NaN or missing data(indicated with a '?')
   data = df[~df.isin(['?'])]
   data = data.dropna(axis=0)
   ```

   

2. Create Training and Testing Dataset

   1) target == y; (1:having heart disease)

   ```python
   # `y` is result, `x` is factors of y (in views of logistic function)
   Y = df.target.values
   X_data = df.drop(['target'], axis=1)
   ```

   

   2) Feature Scaling -  Normalization & Standardization

   some ML alorithms are sensitive to feature scaling while others are virtually invariant to it.

    - **Gradient Descent Based Algorithms** 

      `linear regression, logistic regression, neural network, etc.`

      > Having features on a similar scale can help the gradient descent **converge more quickly towards the minima**

   - **Distance-Based Algorithms**

     `KNN, K-means, SVM`

     > they are using distances between data points to determine their similarity. Therefore, we scale our data before employing a distance based algorithm so that all **features contribute equally to the result**

   - **Tree-Based Algorithms**

     `decision tree, random forest, boosting `

     > Trees are only splitting node based on a single feature.  So there is **virturally no effect**

     

   ```python
   # MINMAX-Normalization(Scale a variable to have a values between 0 and 1)
   X_changed = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))
   
   # Standardization(Transforms data to have a mean of zero and std of 1)
   x_mean = X_data.mean(axis=0)
   X = X_data - x_mean
   
   ```

   

   ```python
   # train_test data split
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)
   ```

   

3. Building and Training the Neural Networks

   

4. Improving Results - A Binary Classification Problem

   

5. Results and Metrics