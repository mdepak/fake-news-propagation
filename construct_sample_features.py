import sklearn

import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from analysis_util import get_propagation_graphs, equal_samples
from structure_temp_analysis import get_all_structural_features
from temporal_analysis import get_all_temporal_features


def get_features(news_graphs):
    temporal_features = get_all_temporal_features(news_graphs)
    structural_features = get_all_structural_features(news_graphs)

    sample_features = np.concatenate([temporal_features, structural_features], axis=1)
    return sample_features


def get_dataset():
    fake_prop_graph, real_prop_graph = get_propagation_graphs("politifact")
    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    print("fake samples len : {} real samples len : {}".format(len(fake_prop_graph), len(real_prop_graph)))

    fake_news_samples = get_features(fake_prop_graph)
    real_news_samples = get_features(real_prop_graph)

    print("Fake feature array ")
    print(fake_news_samples.shape)

    print("real feature array")
    print(real_news_samples.shape)

    return np.concatenate([fake_news_samples, real_news_samples], axis=0), \
           np.concatenate([np.ones(len(fake_news_samples)), np.zeros(len(real_news_samples))], axis=0)


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


def get_train_test_split(samples_features, target_labels):
    X_train, X_test, y_train, y_test = train_test_split(samples_features, target_labels,
                                                        test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)

    predicted_output = classifier.predict(X_test)
    accuracy, precision, recall, f1_score_val = get_metrics(y_test, predicted_output, one_hot_rep=False)
    print_metrics(accuracy, precision, recall, f1_score_val)


def print_metrics(accuracy, precision, recall, f1_score_val):
    print("Accuracy : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("F1 : {}".format(f1_score_val))


def perform_pca(train_data, target_labels):
    pca = PCA(n_components=10)
    return pca.fit_transform(train_data, target_labels)


if __name__ == "__main__":
    train_data, target_labels = get_dataset()

    train_data = perform_pca(train_data, target_labels)

    X_train, X_test, y_train, y_test = get_train_test_split(train_data, target_labels)

    # classifier = GaussianNB()
    # classifier =LogisticRegression()

    classifier = svm.SVC(kernel='linear')
    train_model(classifier, X_train, X_test, y_train, y_test)
