
from random import shuffle

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def LIWC_Representation(data_type):
    f_out = open('./' + data_type + '/LIWCFeats.txt', 'w+')
    with open('LIWC2015_'+data_type+'_fake.txt') as f_fake:
        for line in f_fake:
            line = line.strip()
            all_data = line.split('\t')
            if all_data[0]=='Filename':
                continue
            ID = all_data[0]
            feats = all_data[2:]
            f_out.write(ID+'\t')
            f_out.write('\t'.join(f for f in feats))
            f_out.write('\n')

    with open('LIWC2015_'+data_type+'_real.txt') as f_fake:
        for line in f_fake:
            line = line.strip()
            all_data = line.split('\t')
            if all_data[0]=='Filename':
                continue
            ID = all_data[0]
            feats = all_data[2:]
            f_out.write(ID + '\t')
            f_out.write('\t'.join(f for f in feats))
            f_out.write('\n')
    f_out.close()
    print

def LIWC_Prediction(data_type):
    X_real = []
    y_real = []
    X_fake = []
    y_fake = []
    X=[]
    y=[]
    with open('./'+data_type+'/LIWCFeats.txt') as f_rst:
        for line in f_rst:
            line = line.strip()
            line_str = line.split('\t')
            ID = line_str[0]
            feats = [float(x) for x in line_str[1:]]
            if 'Real' in ID:
                X_real.append(feats)
                y_real.append(0)
            else:
                X_fake.append(feats)
                y_fake.append(1)
    ## Balance fake and true news
    num = len(y_fake)
    X_real = X_real[:num]
    y_real = y_real[:num]
    for i in range(num):
        X.append(X_real[i])
        X.append(X_fake[i])
        y.append(y_real[i])
        y.append(y_fake[i])

    X = np.array(X)
    y = np.array(y)
    # # shuffle the rows
    arry = range(X.shape[0])
    shuffle(arry)
    X = X[arry, :]
    y = y[arry]
    # clf = SVC(kernel='linear', class_weight='balanced')
    # clf = RandomForestClassifier()
    clf = tree.DecisionTreeClassifier()
    X = preprocessing.normalize(X)
    res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
    res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
    print('Accuracy '+res)
    res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='precision')
    res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
    print('precision '+res)
    res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='recall')
    res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
    print('recall '+res)
    res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='f1')
    res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
    print('f1 '+res)
    print

def LIWC_Prediction2(data_type):
    X_real = []
    y_real = []
    X_fake = []
    y_fake = []
    X=[]
    y=[]
    with open('./'+data_type+'/LIWCFeats.txt') as f_rst:
        for line in f_rst:
            line = line.strip()
            line_str = line.split('\t')
            ID = line_str[0]
            feats = [float(x) for x in line_str[1:]]
            if 'Real' in ID:
                X_real.append(feats)
                y_real.append(0)
            else:
                X_fake.append(feats)
                y_fake.append(1)
    ## Balance fake and true news
    num = len(y_fake)
    X_real = X_real[:num]
    y_real = y_real[:num]
    for i in range(num):
        X.append(X_real[i])
        X.append(X_fake[i])
        y.append(y_real[i])
        y.append(y_fake[i])

    X = np.array(X)
    y = np.array(y)
    # # shuffle the rows
    arry = range(X.shape[0])
    shuffle(arry)
    X = X[arry, :]
    y = y[arry]
    clfs = [
        linear_model.LogisticRegression(random_state=22),
        MultinomialNB(),
        tree.DecisionTreeClassifier(random_state=21),
        RandomForestClassifier(random_state=22),
        XGBClassifier(),
        AdaBoostClassifier(random_state=22),
        svm.SVC(kernel='linear', class_weight='balanced'),
        GradientBoostingClassifier(random_state=22),
        BaggingClassifier(random_state=22),
        KNeighborsClassifier()
    ]
    clf_names = [
        'Logistic Regression',
        'Naive Bayes',
        'Decision Tree',
        'Random Forest',
        'XGBoost',
        'AdaBoost',
        'SVM',
        'GradientBoosting',
        'Bagging Clf',
        'KNeighbors Clf'
    ]

    X = preprocessing.normalize(X)
    cols = ['alg', 'avg_acc', 'std_acc','avg_prec','std_prec','avg_rec','std_rec', 'avg_f1','std_f1']

    df = pd.DataFrame(columns=cols)
    df = df.set_index('alg')
    for i in range(len(clfs)):
        clf = clone(clfs[i])
        clf_name = clf_names[i]
        df = test(clf,clf_name,df,cols,X,y)
    print(df)
    df.to_csv('./LIWC_'+data_type+'_results.csv', header=True,sep='\t',columns=cols)

def test(clf, clf_name, df, cols, X, y,train_ratio):
    acc = []
    prec = []
    recall = []
    f1 = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
    tmp = pd.DataFrame([[clf_name, np.average(acc), np.std(acc), np.average(prec), np.std(prec), np.average(recall),
                         np.std(recall), np.average(f1), np.std(f1)]], columns=cols)
    df = df.append(tmp)
    return df

def LIWC_Prediction2_curve(data_type):
    X_real = []
    y_real = []
    X_fake = []
    y_fake = []
    X=[]
    y=[]
    with open('./'+data_type+'/LIWCFeats.txt') as f_rst:
        for line in f_rst:
            line = line.strip()
            line_str = line.split('\t')
            ID = line_str[0]
            feats = [float(x) for x in line_str[1:]]
            if 'Real' in ID:
                X_real.append(feats)
                y_real.append(0)
            else:
                X_fake.append(feats)
                y_fake.append(1)
    ## Balance fake and true news
    num = len(y_fake)
    X_real = X_real[:num]
    y_real = y_real[:num]
    for i in range(num):
        X.append(X_real[i])
        X.append(X_fake[i])
        y.append(y_real[i])
        y.append(y_fake[i])

    X = np.array(X)
    y = np.array(y)
    # # shuffle the rows
    arry = range(X.shape[0])
    shuffle(arry)
    X = X[arry, :]
    y = y[arry]
    clfs = [
        # linear_model.LogisticRegression(random_state=22),
        # MultinomialNB(),
        # tree.DecisionTreeClassifier(random_state=21),
        # RandomForestClassifier(random_state=22),
        # XGBClassifier(),
        AdaBoostClassifier(random_state=22),
        # svm.SVC(kernel='linear', class_weight='balanced'),
        # GradientBoostingClassifier(random_state=22),
        # BaggingClassifier(random_state=22),
        # KNeighborsClassifier()
    ]
    clf_names = [
    #     'Logistic Regression',
    #     'Naive Bayes',
    #     'Decision Tree',
    #     'Random Forest',
    #     'XGBoost',
        'AdaBoost',
        # 'SVM',
        # 'GradientBoosting',
        # 'Bagging Clf',
        # 'KNeighbors Clf'
    ]

    X = preprocessing.normalize(X)
    cols = ['alg', 'avg_acc', 'std_acc','avg_prec','std_prec','avg_rec','std_rec', 'avg_f1','std_f1']

    df = pd.DataFrame(columns=cols)
    df = df.set_index('alg')
    tr = [0.2,0.4,0.6]
    for t in tr:
        for i in range(len(clfs)):
            clf = clone(clfs[i])
            clf_name = clf_names[i]
            df = test(clf, clf_name, df, cols, X, y,t)
    with pd.option_context('expand_frame_repr', False):
        print (df)
    df.to_csv('./LIWC_'+data_type+'_results_curve.csv', header=True,sep='\t',columns=cols)

if __name__ == '__main__':
    data_type = 'BuzzFeed'
    # LIWC_Representation(data_type)
    # LIWC_Prediction2('BuzzFeed')
    # LIWC_Prediction2('PolitiFact')
    LIWC_Prediction2_curve('BuzzFeed')
    LIWC_Prediction2_curve('PolitiFact')
    print