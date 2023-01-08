# Machine-Learning-Final-Project

## Introduction
- This is a solution of Kaggle compitition [Tabular Playground Series - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022), the model I provide can reach 0.59133 at private score. 
![accuracy](https://user-images.githubusercontent.com/71249961/211188668-13a0cd46-c33e-4d4e-83fa-9871ef196fec.jpg)

## How to run
### Enviornment
- Python 3.8.16
- It should work directly if you run the code on colab
- To setup the environment yourself, run `pip install -r requirements.txt`

### Training
- To train the model yourself, run the `109550094_Final_inference.ipynb` directly, be sure both `test.csv` and `train.csv` are at the same directory. Since we analysis the data for test and train at the same time, so both two files need to be load even though for training.
- After running the code, it will generate the `model.pkl`, which is the training model.

### Pre-trained model
- The pre-trained model can be found on the above code region, i.e. the `model.pkl` file, or reference to this [link](https://drive.google.com/file/d/1PgHxevL1BsgPgTm0EZv_ufP-HBAgP3Da/view?usp=share_link). 

### Testing
- To test the model, run the `109550094_Final_inference.ipynb` directly, but 4 files are needed to be provided, first, both `test.csv` and `train.csv` are needed to be at the same directory, due to the data preprocessing need both two files. Second, you need to prepare a trained model named `model.pkl`, you can train by yourself or use the proveided pretrained model. Last, `sample_submission.csv` is needed as the template of output format.
- After running the code, it will generate a file named `submission.csv`, which is the prediction of the testing data.

## Details of implementation
### Data analysis
- We can find that
  1. there are many null values in the data
  2. the result of data is imbalancing
- So we need to solve the above problems in data preprocessing

### Data preprocssing
  1. To solve the first problem, I choose the idea[<sup>1</sup>](#reference) of using correlated columns to predict the null value, the prediction model is HaberRegressor. Since the correlated columns may also be null, so it may fail to predict. If it fail to predict, we use KNN instead.
  2. To solve the second problem, I use `imblearn.over_sampling.SMOTE` to generate some data for the class that has less data.
  3. Then, we choose some of the features that is more helpful to train the model instead of the whole features 
  3. Finally, we preform standardize on the input data before sending into the model
  
### Model
- I use Linear Support Vector Regression as the main model of training, the hyperparemeter I use is `epsilon=0, C=0.0001`, which perfomed the best
- On training, I use KFold first to adjust the hyperparemeter, and then use the whole training set to train the model

## Result Analysis
### Hyperparemeter testing
- Model Compare

|Model             |Accuracy|	Model                |Accuracy|
|------------------|--------|----------------------|--------|
|LinearSVR         |**0.59092**|SGDRegressor       |	0.59  |
|HugerRegression	 |0.5909  |RadiusNeighbors	     |0.5883  |
|PLSRegression	   |0.5909  |MLPRegressor          |0.58299 |
|LogisticRegression|0.59089 |	SVR                  |0.57898 |
|LinearRegression  |0.59087 |AdaboostRegression	   |0.57062 |
|Ridge          	 |0.59087 |RANSACRegression      |0.56859 |
|BayesianRidge     |0.59087 |RandomForestClassifier|0.53718 |
|TweedieRegression |0.59074 |DecisionTreeRegression|0.51337 |
|TheilSenRegressor |0.59067 |KNeighbors            |0.52154 |

- Used Feature
  - First we use whole features to train the model, and we can get the importance of each feature, as the figure show below.
  ![featureImportance](https://user-images.githubusercontent.com/71249961/211187814-a00a6243-6baa-4f73-91fc-84ca3b7b9232.png)
  - Then we can choose only parts of features to train the model
 
  |Used feature|all	   |top 10 |top 6|top 5	 |top 4  |top 3  |top 2  |
  |------------|-------|-------|-----|-------|-------|-------|-------|
  |Accuracy    |0.58941|0.59061|0.591|0.59117|0.59126|0.59114|0.59116|
  
- Null Filled Regressor Hyperparemeter
  - The hyperparemeter of the null value predict model, HugerRegressor

  |epsilon     |1.7	   |1.8    |1.9    |1.95	 |2          |2.05   |2.05   |
  |------------|-------|-------|-------|-------|-----------|-------|-------|
  |Accuracy    |0.59122|0.59121|0.59126|0.59131|**0.59133**|0.59128|0.59125|
  
-	Null filled policy
  - The hyperparemeters used to filled the null columns, we can get the best result if we only use the most correlated features, I think this is because the more features we use, the higher probability the correlated columns have null value in it and can't be use, and fall back to use the KNN to predict, which is less accurate(can reference to the [Ablation Study](#ablation-study) at below)
  
  |k of KNN|correlated columns count|Accuracy|
  |--------|------------------------|--------|
  |5	     |1	                      |0.5909  |
  |4	     |1	                      |0.59097 |
  |1	     |1	                      |0.58992 |
  |2	     |1                      	|0.5908  |
  |3	     |1                    |**0.59126**|
  |3	     |2                     	|0.59065 |
  |3	     |3	                      |0.59069 |
  |3	     |4	                      |0.59075 |
  
### Ablation Study
- To verify the model we design, we test some case without specific parts of our model

|Model                                   |Accuracy|
|----------------------------------------|--------|
|Current Model                        |**0.59126**|
|w/o balancing data                      |0.59087 |
|w/o filled null using correlated columns|0.58929 |
|w/o feature selection                   |0.58941 |




## Reference
1. <https://www.kaggle.com/code/vishnu123/tps-aug-22-top-2-logistic-regression-cv-fe> [(↑)](#data-preprocssing)

