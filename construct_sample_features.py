import os
import pickle
from pathlib import Path

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from analysis_util import get_propagation_graphs, equal_samples
from linguistic_analysis import get_all_linguistic_features, LinguisticFeatureHelper
from structure_temp_analysis import get_all_structural_features, StructureFeatureHelper
from temporal_analysis import get_all_temporal_features, TemporalFeatureHelper


def get_features(news_graphs, micro_features, macro_features):
    temporal_features = get_all_temporal_features(news_graphs, micro_features, macro_features)
    structural_features = get_all_structural_features(news_graphs, micro_features, macro_features)

    linguistic_features = get_all_linguistic_features(news_graphs, micro_features, macro_features)

    sample_features = np.concatenate([temporal_features, structural_features, linguistic_features], axis=1)
    return sample_features


def get_dataset(news_source, load_dataset=False, micro_features=True, macro_features=True):
    if load_dataset:
        sample_features = pickle.load(open("{}_samples_features.pkl".format(news_source), "rb"))
        target_labels = pickle.load(open("{}_target_labels.pkl".format(news_source), "rb"))

    else:
        fake_prop_graph, real_prop_graph = get_propagation_graphs(news_source)
        fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

        print("fake samples len : {} real samples len : {}".format(len(fake_prop_graph), len(real_prop_graph)))

        fake_news_samples = get_features(fake_prop_graph, micro_features, macro_features)
        real_news_samples = get_features(real_prop_graph, micro_features, macro_features)

        print("Fake feature array ")
        print(fake_news_samples.shape)

        print("real feature array")
        print(real_news_samples.shape)

        sample_features = np.concatenate([fake_news_samples, real_news_samples], axis=0)
        target_labels = np.concatenate([np.ones(len(fake_news_samples)), np.zeros(len(real_news_samples))], axis=0)

        pickle.dump(sample_features, (open("{}_samples_features.pkl".format(news_source), "wb")))
        pickle.dump(target_labels, (open("{}_target_labels.pkl".format(news_source), "wb")))

    return sample_features, target_labels


def get_train_test_split(samples_features, target_labels):
    X_train, X_test, y_train, y_test = train_test_split(samples_features, target_labels,
                                                        test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def perform_pca(train_data, target_labels):
    pca = PCA(n_components=min(20, len(train_data[0])))
    pca.fit(train_data, target_labels)
    return pca


def get_dataset_file_name(file_dir, news_source, include_micro=True, include_macro=True, include_structural=True,
                          include_temporal=True,
                          include_linguistic=True):
    file_names = [news_source]
    if include_micro:
        file_names.append("micro")

    if include_macro:
        file_names.append("macro")

    if include_structural:
        file_names.append("struct")

    if include_temporal:
        file_names.append("temp")

    if include_linguistic:
        file_names.append("linguistic")

    return "{}/{}.pkl".format(file_dir, "_".join(file_names))


def get_TPNF_dataset(out_dir, news_source, include_micro=True, include_macro=True, include_structural=None,
                     include_temporal=None,
                     include_linguistic=None):
    file_name = get_dataset_file_name(out_dir, news_source, include_micro, include_macro, include_structural,
                                      include_temporal, include_linguistic)

    data_file = Path(file_name)

    if data_file.is_file():
        return pickle.load(open(file_name, "rb"))

    else:
        fake_sample_features, real_sample_features = get_dataset_feature_array(news_source, include_micro,
                                                                               include_macro, include_structural,
                                                                               include_temporal, include_linguistic)

        sample_features = np.concatenate([fake_sample_features, real_sample_features], axis=0)
        pickle.dump(sample_features, open(data_file, "wb"))


def get_dataset_feature_array(news_source, include_micro=True, include_macro=True, include_structural=None,
                              include_temporal=None,
                              include_linguistic=None):
    fake_prop_graph, real_prop_graph = get_propagation_graphs("data/saved_new_no_filter", news_source)

    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    feature_helpers = []

    if include_structural:
        feature_helpers.append(StructureFeatureHelper())

    if include_temporal:
        feature_helpers.append(TemporalFeatureHelper())

    if include_linguistic:
        feature_helpers.append(LinguisticFeatureHelper())

    for feature_helper in feature_helpers:
        fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=include_micro,
                                                          macro_features=include_macro, news_source=news_source,
                                                          label="fake")
        real_features = feature_helper.get_features_array(real_prop_graph, micro_features=include_micro,
                                                          macro_features=include_macro, news_source=news_source,
                                                          label="real")
        return fake_features, real_features


def get_dataset_statistics(news_source):
    fake_prop_graph, real_prop_graph = get_propagation_graphs("data/saved_new_no_filter", news_source)

    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    # feature_helpers = []

    # feature_helpers = [StructureFeatureHelper(), TemporalFeatureHelper() , LinguisticFeatureHelper()]
    feature_helpers = [LinguisticFeatureHelper()]

    for feature_helper in feature_helpers:
        fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="fake")
        real_features = feature_helper.get_features_array(real_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="real")

        feature_helper.save_blox_plots_for_features(fake_feature_array=fake_features,
                                                    real_feature_array=real_features, micro_features=True,
                                                    macro_features=True, save_folder="data/feature_images/gossipcop")

        # Print the statistics of the dataset
        print("------------Fake------------")
        feature_helper.print_statistics_for_all_features(feature_array=fake_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)

        print("------------Real------------")
        feature_helper.print_statistics_for_all_features(feature_array=real_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)


if __name__ == "__main__":
    get_dataset_statistics("gossipcop")

    exit(1)

    train_data, target_labels = get_dataset(news_source="politifact", load_dataset=False, micro_features=True,
                                            macro_features=True)

    train_data = preprocessing.scale(train_data)

    X_train, X_test, y_train, y_test = get_train_test_split(train_data, target_labels)

    train_data = perform_pca(train_data, target_labels)
    pca = perform_pca(X_train, y_train)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # classifier = GaussianNB()
    classifier = LogisticRegression()
    # classifier = DecisionTreeClassifier()

    # classifier = RandomForestClassifier()
    # classifier = svm.SVC(kernel='linear')

    # train_model(classifier, X_train, X_test, y_train, y_test)
