# Machine-Learning-Final-Project

## Introduction
- This is a solution of Kaggle compitition [Tabular Playground Series - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022), the model I provide can reach 0.59131 at private score. 
![accuracy](https://user-images.githubusercontent.com/71249961/211186231-b701036c-7303-4ebf-82e6-e2afa0fa11ba.jpg)


## How to run
### Enviornment
- Python 3.8.16
- It should work directly if you run the code on colab
- To setup the environment yourself, run `pip install -r requirements.txt`

### Training
- To train the model yourself, run the `109550094_Final_inference.ipynb` directly, be sure both `test.csv` and `train.csv` are at the same directory. Since we analysis the data for test and train at the same time, so both two files need to be load even though for training.
- After running the code, it will generate the `model.pkl`, which is the training model.

### Testing
- To test the model, run the `109550094_Final_inference.ipynb` directly, but 4 files are needed to be provided, first, both `test.csv` and `train.csv` are needed to be at the same directory, due to the data preprocessing need both two files. Second, you need to prepare a trained model named `model.pkl`, you can train by yourself or the above also provide a trained model. Last, `sample_submission.csv` is needed as the template of output format.
- After running the code, it will generate a file named `submission.csv`, which is the prediction of the testing data.

## Details of implementation
### Data analysis
- We can find that
  1. there are many null values in the data
  2. the result of data is imbalancing
- So we need to solve the above problems in data preprocessing

### Data preprocssing
  1. To solve the first problem, I choose the idea[<sup>1</sup>](#reference) of using correlated columns to predict the null value, the prediction model is HaberRegressor. Since the correlated columns may also be null, so it may fail to predict. If it fail to predict, we use KNN instead.
  2. To solve the second problem, I use `imblearn.over_sampling import SMOTE` to generate some data for the class that has less data.
  3. Then, we choose some of the features that is more helpful to train the model instead of the whole features 
  3. Finally, we preform standardize on the input data before sending into the model
  
### Model
- I use Linear Support Vector Regression as the main model of training, the hyperparemeter I use is `epsilon=0, C=0.0001`, which perfomed the best
- On training, I use KFold first to adjust the hyperparemeter, and then use the whole training set to train the model


## Reference
1. <https://www.kaggle.com/code/vishnu123/tps-aug-22-top-2-logistic-regression-cv-fe> [(↑)](#data-preprocssing)

