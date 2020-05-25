import pickle
import queue
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from analysis_util import equal_samples
from linguistic_analysis import get_all_linguistic_features, LinguisticFeatureHelper
from load_dataset import load_from_nx_graphs
from structure_temp_analysis import get_all_structural_features, StructureFeatureHelper, get_first_post_time
from temporal_analysis import get_all_temporal_features, TemporalFeatureHelper
from util.util import tweet_node


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
        fake_prop_graph, real_prop_graph = get_nx_propagation_graphs(news_source)
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
    X_train, X_test, y_train, y_test = train_test_split(samples_features, target_labels, stratify=target_labels,
                                                        test_size=0.2, random_state=42)
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
                     include_linguistic=None, time_interval=None, use_cache=False):
    file_name = get_dataset_file_name(out_dir, news_source, include_micro, include_macro, include_structural,
                                      include_temporal, include_linguistic)

    data_file = Path(file_name)

    if use_cache and data_file.is_file():
        return pickle.load(open(file_name, "rb"))

    else:
        fake_sample_features, real_sample_features = get_dataset_feature_array(news_source, include_micro,
                                                                               include_macro, include_structural,
                                                                               include_temporal, include_linguistic,
                                                                               time_interval)

        sample_features = np.concatenate([fake_sample_features, real_sample_features], axis=0)
        pickle.dump(sample_features, open(file_name, "wb"))

        return sample_features


def get_dataset_feature_names(include_micro=True, include_macro=True, include_structural=None,
                              include_temporal=None,
                              include_linguistic=None):
    feature_helpers = []

    if include_structural:
        feature_helpers.append(StructureFeatureHelper())

    if include_temporal:
        feature_helpers.append(TemporalFeatureHelper())

    if include_linguistic:
        feature_helpers.append(LinguisticFeatureHelper())

    feature_names_all = []
    short_feature_names_all = []

    for idx, feature_helper in enumerate(feature_helpers):
        features_names, short_feature_names = feature_helper.get_feature_names(include_micro, include_macro)

        feature_names_all.extend(features_names)
        short_feature_names_all.extend(short_feature_names)

    return feature_names_all, short_feature_names_all


def is_valid_graph(prop_graph: tweet_node, retweet=True, reply=True):
    """ Check if the prop graph has alteast one retweet or reply"""

    for post_node in prop_graph.children:
        if (retweet and len(post_node.reply_children) > 0) or (reply and len(post_node.retweet_children) > 0):
            return True

    return False


def remove_node_by_time(graph: tweet_node, limit_time):
    start_time = get_first_post_time(graph)
    end_time = start_time + limit_time

    q = queue.Queue()

    q.put(graph)

    while q.qsize() != 0:
        node = q.get()

        children = node.children

        retweet_children = set(node.retweet_children)
        reply_children = set(node.reply_children)

        for child in children.copy():

            if child.created_time <= end_time:
                q.put(child)
            else:
                node.children.remove(child)
                try:
                    retweet_children.remove(child)
                except KeyError:  # Element not found in the list
                    pass
                try:
                    reply_children.remove(child)
                except KeyError:  # Element not found in the list
                    pass

        node.retweet_children = list(retweet_children)
        node.reply_children = list(reply_children)

    return graph


def filter_propagation_graphs(graphs, limit_time):
    result_graphs = []

    for prop_graph in graphs:
        filtered_prop_graph = remove_node_by_time(prop_graph, limit_time)
        if is_valid_graph(filtered_prop_graph):
            result_graphs.append(filtered_prop_graph)

    return result_graphs


def get_nx_propagation_graphs(data_folder, news_source):
    fake_propagation_graphs = load_from_nx_graphs(data_folder, news_source, "fake")
    real_propagation_graphs = load_from_nx_graphs(data_folder, news_source, "real")

    return fake_propagation_graphs, real_propagation_graphs


def get_dataset_feature_array(news_source, include_micro=True, include_macro=True, include_structural=None,
                              include_temporal=None,
                              include_linguistic=None, time_interval=None):
    fake_prop_graph, real_prop_graph = get_nx_propagation_graphs("data/nx_network_data", news_source)

    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    if time_interval is not None:
        time_limit = time_interval * 60 * 60

        print("Time limit in seconds : {}".format(time_limit))

        fake_prop_graph = filter_propagation_graphs(fake_prop_graph, time_limit)
        real_prop_graph = filter_propagation_graphs(real_prop_graph, time_limit)

        print("After time based filtering ")
        print("No. of fake samples : {}  No. of real samples: {}".format(len(fake_prop_graph), len(real_prop_graph)))

        fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    feature_helpers = []
    feature_group_names = []

    if include_structural:
        feature_helpers.append(StructureFeatureHelper())
        feature_group_names.append("Structural")

    if include_temporal:
        feature_helpers.append(TemporalFeatureHelper())
        feature_group_names.append("Temporal")

    if include_linguistic:
        feature_helpers.append(LinguisticFeatureHelper())
        feature_group_names.append("Linguistic")

    fake_feature_all = []
    real_feature_all = []
    for idx, feature_helper in enumerate(feature_helpers):
        fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=include_micro,
                                                          macro_features=include_macro, news_source=news_source,
                                                          label="fake")
        real_features = feature_helper.get_features_array(real_prop_graph, micro_features=include_micro,
                                                          macro_features=include_macro, news_source=news_source,
                                                          label="real")

        feature_names = feature_helper.get_feature_names(micro_features=include_micro, macro_features=include_macro)
        print(feature_names)
        if fake_features is not None and real_features is not None:
            fake_feature_all.append(fake_features)
            real_feature_all.append(real_features)

            print("Feature group : {}".format(feature_group_names[idx]))
            print(len(fake_features))
            print(len(real_features), flush=True)

    return np.concatenate(fake_feature_all, axis=1), np.concatenate(real_feature_all, axis=1)


def get_dataset_statistics(news_source):
    fake_prop_graph, real_prop_graph = get_nx_propagation_graphs("data/saved_new_no_filter", news_source)

    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    feature_helpers = [StructureFeatureHelper(), TemporalFeatureHelper(), LinguisticFeatureHelper()]
    feature_group_names = ["StructureFeatureHelper", "TemporalFeatureHelper", "LinguisticFeatureHelper"]

    for idx, feature_helper in enumerate(feature_helpers):
        print("Feature group : {}".format(feature_group_names[idx]))

        fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="fake")
        real_features = feature_helper.get_features_array(real_prop_graph, micro_features=True,
                                                          macro_features=True, news_source=news_source, label="real")

        feature_helper.save_blox_plots_for_features(fake_feature_array=fake_features,
                                                    real_feature_array=real_features, micro_features=True,
                                                    macro_features=True,
                                                    save_folder="data/feature_images/{}".format(news_source))

        feature_helper.get_feature_significance_t_tests(fake_features, real_features, micro_features=True,
                                                        macro_features=True)

        # Print the statistics of the dataset
        print("------------Fake------------")
        feature_helper.print_statistics_for_all_features(feature_array=fake_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)

        print("------------Real------------")
        feature_helper.print_statistics_for_all_features(feature_array=real_features, prop_graphs=fake_prop_graph,
                                                         micro_features=True, macro_features=True)


if __name__ == "__main__":
    get_dataset_statistics("politifact")
    get_dataset_statistics("gossipcop")
