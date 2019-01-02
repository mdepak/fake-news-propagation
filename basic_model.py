import numpy as np

from sklearn import preprocessing, svm, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from construct_sample_features import get_TPNF_dataset, get_train_test_split


def train_model(classifier, X_train, X_test, y_train, y_test):
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_score_values = []

    for i in range(5):
        classifier_clone = clone(classifier)
        classifier_clone.fit(X_train, y_train)

        predicted_output = classifier_clone.predict(X_test)
        accuracy, precision, recall, f1_score_val = get_metrics(y_test, predicted_output, one_hot_rep=False)
        accuracy_values.append(accuracy)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_score_values.append(f1_score_val)

    print_metrics(np.mean(accuracy_values), np.mean(precision_values), np.mean(recall_values), np.mean(f1_score_values))


def print_metrics(accuracy, precision, recall, f1_score_val):
    print("Accuracy : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("F1 : {}".format(f1_score_val))


def get_metrics(target, logits, one_hot_rep=True):
    """
    Two numpy one hot arrays
    :param target:
    :param logits:
    :return:
    """

    if one_hot_rep:
        label = np.argmax(target, axis=1)
        predict = np.argmax(logits, axis=1)
    else:
        label = target
        predict = logits

    accuracy = accuracy_score(label, predict)

    precision = precision_score(label, predict)
    recall = recall_score(label, predict)
    f1_score_val = f1_score(label, predict)

    return accuracy, precision, recall, f1_score_val


def get_basic_model_results(X_train, X_test, y_train, y_test):
    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifiers = [GaussianNB(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(),
                   svm.SVC(kernel='linear')]
    classifier_names = ["GaussianNB", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier",
                        "SVM -linear kernel"]

    for idx in range(len(classifiers)):
        print("======={}=======".format(classifier_names[idx]))
        train_model(classifier_names[idx], X_train, X_test, y_train, y_test)


def get_classificaton_results_tpnf(data_dir, news_source):
    include_micro = True
    include_macro = True

    include_structural = True
    include_temporal = True
    include_linguistic = False

    sample_feature_array = get_TPNF_dataset(data_dir, news_source, include_micro, include_macro, include_structural,
                                            include_temporal, include_linguistic)

    print("Sample feature array dimensions")
    print(sample_feature_array.shape, flush=True)

    num_samples = len(sample_feature_array)
    target_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)

    X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)
    get_basic_model_results(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    get_classificaton_results_tpnf("data/train_test_data", "politifact")
