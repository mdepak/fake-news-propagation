# This is an implementation of Rhetorical Structure Theory for Vector Space Model
# The basic idea is from the paper: Identification of Truth and Deception in Text: Application of Vector Space Model to Rhetorical Structure Theory
from os import listdir
from os.path import isfile, join
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def RSTRepresentation(data_type):
    dir_path = './'+data_type+'/RST'
    f_out = open('./'+data_type+'/RSTFeats.txt','w+')
    all_relations = set()
    org_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    News_RSTFeats = dict()
    for of in org_files:
        ID = of[:of.index('.txt')]
        file_name = dir_path+'/'+of
        relation_num = dict()
        with open(file_name) as f_rst:
            for line in f_rst:
                line = line.replace('\'','')
                line = line.replace(' ','')
                tmp_relation = line.split(',')[3]
                relation = tmp_relation[:-2]
                all_relations.add(relation)
                if relation in relation_num:
                    num = relation_num[relation]
                    num+=1
                    relation_num[relation] = num
                else:
                    relation_num[relation]=1
        News_RSTFeats[ID] = relation_num

    all_relations = list(all_relations)
    print(all_relations)
    for news, rn in News_RSTFeats.items():
        f_out.write(news+'\t')
        feats = []
        for al in all_relations:
            if al in rn:
                num = rn[al]
            else:
                num=0
            feats.append(num)
        f_out.write('\t'.join(str(x) for x in feats))
        f_out.write('\n')
    f_out.close()


def RSTPrediction2(data_type):
    X_real = []
    y_real = []
    X_fake = []
    y_fake = []
    X=[]
    y=[]
    with open('./'+data_type+'/RSTFeats.txt') as f_rst:
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
        df = test(clf,clf_name,df,cols,X,y,0.8)

    print(df)
    df.to_csv('./RST_'+data_type+'_results.csv', header=True,sep='\t',columns=cols)


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

def RSTPrediction2_curve(data_type):
    X_real = []
    y_real = []
    X_fake = []
    y_fake = []
    X=[]
    y=[]
    with open('./'+data_type+'/RSTFeats.txt') as f_rst:
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
        'Logistic Regression',
        # 'Naive Bayes',
        # 'Decision Tree',
        # 'Random Forest',
        # 'XGBoost',
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
    df.to_csv('./RST_'+data_type+'_results_curve.csv', header=True,sep='\t',columns=cols)

if __name__ == '__main__':
    data_type = 'PolitiFact'
    # RSTRepresentation(data_type)
    # RSTPrediction2('BuzzFeed')
    # RSTPrediction2('PolitiFact')
    RSTPrediction2_curve('BuzzFeed')
    RSTPrediction2_curve('PolitiFact')