import datetime
import numpy as np
import pandas as pd
from numpy import array
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def convert_to_learner_preds(x_train, learner1, learner2):
    learner_preds = []
    tree_pred = learner1.predict(x_train)
    mlp_pred = learner2.predict(x_train)
    learner_preds.append(tree_pred)
    learner_preds.append(mlp_pred)
    learner_preds = np.asarray(learner_preds).transpose()
    return learner_preds


def eraseUnwanted(data):
    for c in data.columns:
        if data[c].isnull().sum() > data.shape[0]*0.3:
            data.drop(columns=c, inplace=True)
    return data


def getValidData(data, label):
    indX = np.arange(1, y.shape[0] + 1)
    data = pd.DataFrame(data=data, index=indX)
    validXY = []
    for Ti in label.columns:
        X_temp = data[~label[Ti].isnull()]
        y_temp = label[~label[Ti].isnull()][Ti]
        validXY.append(pd.concat((X_temp, y_temp), axis=1))
    return validXY


def unitePredictions(X1, X2, X3, X4, X5, X6):
    return np.concatenate((X1, X2, X3, X4, X5, X6), axis=1)


def trainForests(X1, X2, X3, X4, X5, X6, y1, y2, y3, y4, y5, y6):
    tree1 = RandomForestClassifier().fit(X1, y1)
    tree2 = RandomForestClassifier().fit(X2, y2)
    tree3 = RandomForestClassifier().fit(X3, y3)
    tree4 = RandomForestClassifier().fit(X4, y4)
    tree5 = RandomForestClassifier().fit(X5, y5)
    tree6 = RandomForestClassifier().fit(X6, y6)
    return tree1, tree2, tree3, tree4, tree5, tree6


def trainKnns(X1, X2, X3, X4, X5, X6, y1, y2, y3, y4, y5, y6):
    tree1 = KNeighborsClassifier().fit(X1, y1)
    tree2 = KNeighborsClassifier().fit(X2, y2)
    tree3 = KNeighborsClassifier().fit(X3, y3)
    tree4 = KNeighborsClassifier().fit(X4, y4)
    tree5 = KNeighborsClassifier().fit(X5, y5)
    tree6 = KNeighborsClassifier().fit(X6, y6)
    return tree1, tree2, tree3, tree4, tree5, tree6


def getPredictions(learner1, X1, learner2, X2, learner3, X3, learner4, X4, learner5, X5, learner6, X6):
    preds1 = learner1.predict_proba(X1)[:, 1]
    preds2 = learner2.predict_proba(X2)[:, 1]
    preds3 = learner3.predict_proba(X3)[:, 1]
    preds4 = learner4.predict_proba(X4)[:, 1]
    preds5 = learner5.predict_proba(X5)[:, 1]
    preds6 = learner6.predict_proba(X6)[:, 1]
    return preds1, preds2, preds3, preds4, preds5, preds6


def trainLR(X1, X2, X3, X4, X5, X6, y1, y2, y3, y4, y5, y6):
    logReg1 = LogisticRegression().fit(X1, y1)
    logReg2 = LogisticRegression().fit(X2, y2)
    logReg3 = LogisticRegression().fit(X3, y3)
    logReg4 = LogisticRegression().fit(X4, y4)
    logReg5 = LogisticRegression().fit(X5, y5)
    logReg6 = LogisticRegression().fit(X6, y6)
    return logReg1, logReg2, logReg3, logReg4, logReg5, logReg6


def getScore(learner1, X1, y1, learner2, X2, y2, learner3, X3, y3,
            learner4, X4, y4, learner5, X5, y5, learner6, X6, y6):
    preds1 = learner1.score(X1, y1)
    preds2 = learner2.score(X2, y2)
    preds3 = learner3.score(X3, y3)
    preds4 = learner4.score(X4, y4)
    preds5 = learner5.score(X5, y5)
    preds6 = learner6.score(X6, y6)
    return preds1, preds2, preds3, preds4, preds5, preds6


def printScoresRandF(learner, X1, y1):
    print("Random Forest Score: ", learner.score(X1, y1))
    print(confusion_matrix(y1, learner.predict(X1)))
    print('AUROC: ', roc_auc_score(y1, learner.predict_proba(X1)[:, 1]))


def printScoresLR(learner, X1, y1):
    print("Logistic Regression Score: ", learner.score(X1, y1))
    print(confusion_matrix(y1, learner.predict(X1)))
    print('AUROC: ', roc_auc_score(y1, learner.predict_proba(X1)[:, 1]))


def printScoresKnn(learner, X1, y1):
    print("K-NNs Score: ", learner.score(X1, y1))
    print(confusion_matrix(y1, learner.predict(X1)))
    print('AUROC: ', roc_auc_score(y1, learner.predict_proba(X1)[:, 1]))



def extractX(data, num):
    Xi = data[num]
    return Xi.values[:, :640]


def extractY(data, num):
    Xi = data[num]
    return Xi.values[:, 640:]


def handleStrRows(data):
    strCols = data.select_dtypes('object')
    numCols = data.select_dtypes('int64', 'float64')
    imputerS = SimpleImputer(strategy="most_frequent").fit(strCols)
    imputer = SimpleImputer(strategy="mean").fit(numCols)
    strCols = imputerS.transform(strCols)
    numCols = imputer.transform(numCols)
    ohe = OneHotEncoder()
    enc = ohe.fit(strCols)
    strColsOhed = enc.transform(strCols).toarray()
    res = np.concatenate((numCols, strColsOhed), axis=1)
    return res

def printTargets():
    print('X1: ', X1.shape)
    print('X2: ', X2.shape)
    print('X3: ', X3.shape)
    print('X4: ', X4.shape)
    print('X5: ', X5.shape)
    print('X6: ', X6.shape)

if __name__ == '__main__':
    #   Import Data
    X = pd.read_csv('hw08_training_data.csv', index_col='ID')
    y = pd.read_csv('hw08_training_label.csv', index_col='ID')
    X_t = pd.read_csv('hw08_test_data.csv', index_col='ID')
    print(X.shape)
    print(X_t.shape)
    print(y.shape)
    train_num = X.shape[0]
    X = pd.concat((X, X_t), axis=0)

    # Data Preprocess
    eraseUnwanted(X)
    X = handleStrRows(X)
    X_t = X[train_num:]
    X = X[:train_num]
    print(X.shape)
    print(X_t.shape)
    print(y.shape)
    Xy = getValidData(X, y)
    X1, X2, X3, X4, X5, X6 = extractX(Xy, 0), extractX(Xy, 1), extractX(Xy, 2),\
        extractX(Xy, 3), extractX(Xy, 4), extractX(Xy, 5)
    y1, y2, y3, y4, y5, y6 = extractY(Xy, 0), extractY(Xy, 1), extractY(Xy, 2), \
        extractY(Xy, 3), extractY(Xy, 4), extractY(Xy, 5)
    printTargets()
    print(y1.shape)
    print(y5.shape)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2)
    X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.2)
    X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, test_size=0.2)
    X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, test_size=0.2)

    # Random Forest for learner
    for1, for2, for3, for4, for5, for6 = trainForests(X_train1, X_train2, X_train3, X_train4, X_train5, X_train6,
                                                    y_train1, y_train2, y_train3, y_train4, y_train5, y_train6)

    print('Target 1: ')
    printScoresRandF(for1, X_test1, y_test1)
    print('Target 2: ')
    printScoresRandF(for2, X_test2, y_test2)
    print('Target 3: ')
    printScoresRandF(for3, X_test3, y_test3)
    print('Target 4: ')
    printScoresRandF(for4, X_test4, y_test4)
    print('Target 5: ')
    printScoresRandF(for5, X_test5, y_test5)
    print('Target 6: ')
    printScoresRandF(for6, X_test6, y_test6)


    # K-NN for learner

    knn1, knn2, knn3, knn4, knn5, knn6 = trainKnns(X_train1, X_train2, X_train3, X_train4, X_train5, X_train6,
                                                      y_train1, y_train2, y_train3, y_train4, y_train5, y_train6)
    print('Target 1: ')
    printScoresKnn(knn1, X_test1, y_test1)
    print('Target 2: ')
    printScoresKnn(knn2, X_test2, y_test2)
    print('Target 3: ')
    printScoresKnn(knn3, X_test3, y_test3)
    print('Target 4: ')
    printScoresKnn(knn4, X_test4, y_test4)
    print('Target 5: ')
    printScoresKnn(knn5, X_test5, y_test5)
    print('Target 6: ')
    printScoresKnn(knn6, X_test6, y_test6)





    # Logistic Regression for ensemble
    lr1, lr2, lr3, lr4, lr5, lr6 = trainLR(convert_to_learner_preds(X_train1, knn1, for1),
                                           convert_to_learner_preds(X_train2, knn2, for2),
                                           convert_to_learner_preds(X_train3, knn3, for3),
                                           convert_to_learner_preds(X_train4, knn4, for4),
                                           convert_to_learner_preds(X_train5, knn5, for5),
                                           convert_to_learner_preds(X_train6, knn6, for6),
                                           y_train1, y_train2, y_train3, y_train4, y_train5, y_train6)
    print('Target 1: ')
    printScoresLR(lr1, convert_to_learner_preds(X_test1, knn1, for1), y_test1)
    print('Target 2: ')
    printScoresLR(lr2, convert_to_learner_preds(X_test2, knn2, for2), y_test2)
    print('Target 3: ')
    printScoresLR(lr3, convert_to_learner_preds(X_test3, knn3, for3), y_test3)
    print('Target 4: ')
    printScoresLR(lr4, convert_to_learner_preds(X_test4, knn4, for4), y_test4)
    print('Target 5: ')
    printScoresLR(lr5, convert_to_learner_preds(X_test5, knn5, for5), y_test5)
    print('Target 6: ')
    printScoresLR(lr6, convert_to_learner_preds(X_test6, knn6, for6), y_test6)

    # Writing Test Results
    p1, p2, p3, p4, p5, p6 = getPredictions(for1, X_t, for2, X_t, for3, X_t, for4, X_t, for5, X_t, for6, X_t)
    test_preds = unitePredictions(p1[:, np.newaxis], p2[:, np.newaxis],
                        p3[:, np.newaxis], p4[:, np.newaxis], p5[:, np.newaxis], p6[:, np.newaxis])
    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df.to_csv(header=False, index=False, path_or_buf='hw08_test_predictions.csv')

    # Creating Pseudo Labels
