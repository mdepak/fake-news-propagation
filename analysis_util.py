import errno
import os

import numpy as np
import pickle

from stat_test import get_box_plots
from util.util import twitter_datetime_str_to_object, tweet_node

from abc import ABCMeta, abstractmethod


class BaseFeatureHelper(metaclass=ABCMeta):

    @abstractmethod
    def get_micro_feature_method_references(self):
        pass

    @abstractmethod
    def get_micro_feature_method_names(self):
        pass

    @abstractmethod
    def get_micro_feature_short_names(self):
        pass

    @abstractmethod
    def get_macro_feature_method_references(self):
        pass

    @abstractmethod
    def get_macro_feature_method_names(self):
        pass

    @abstractmethod
    def get_macro_feature_short_names(self):
        pass

    def get_features_array(self, prop_graphs, micro_features, macro_features, news_source=None, label=None):
        function_refs = []

        if micro_features:
            function_refs.extend(self.get_micro_feature_method_references())

        if macro_features:
            function_refs.extend(self.get_macro_feature_method_references())

        if len(function_refs) == 0:
            return None

        all_features = []

        for function_reference in function_refs:
            features_set = get_sample_feature_value(prop_graphs, function_reference)
            all_features.append(features_set)

        return np.transpose(get_numpy_array(all_features))

    def get_feature_names(self, micro_features, macro_features):
        features_names = []
        short_feature_names = []

        if micro_features:
            features_names.extend(self.get_micro_feature_method_names())
            short_feature_names.extend(self.get_micro_feature_short_names())

        if macro_features:
            features_names.extend(self.get_macro_feature_method_names())
            short_feature_names.extend(self.get_macro_feature_short_names())

        return features_names, short_feature_names

    def print_statistics_for_all_features(self, feature_array=None, prop_graphs=None, micro_features=None,
                                          macro_features=None):

        if feature_array is None:
            feature_array = self.get_features_array(prop_graphs, micro_features, macro_features)

        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            feature_values = feature_array[:, idx]
            print_stat_values(feature_names[idx], feature_values, short_feature_names[idx])

    def save_blox_plots_for_features(self, fake_feature_array=None, real_feature_array=None, fake_prop_graphs=None,
                                     real_prop_graphs=None, micro_features=None, macro_features=None, save_folder=None):

        if fake_feature_array is None:
            fake_feature_array = self.get_features_array(fake_prop_graphs, micro_features, macro_features)
            real_feature_array = self.get_features_array(real_prop_graphs, micro_features, macro_features)

        [feature_names, short_feature_names] = self.get_feature_names(micro_features, macro_features)

        for idx in range(len(feature_names)):
            fake_feature_values = fake_feature_array[:, idx]
            real_feature_values = real_feature_array[:, idx]
            get_box_plots(fake_feature_values, real_feature_values, save_folder, feature_names[idx],
                          short_feature_names[idx])


def get_sample_feature_value(news_graps: list, get_feature_fun_ref):
    result = []
    for graph in news_graps:
        result.append(get_feature_fun_ref(graph))

    return result


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def get_epoch_timestamp_from_retweet(retweet):
    return twitter_datetime_str_to_object(retweet["created_at"])


def sort_retweet_object_by_time(retweets: list):
    retweets.sort(key=get_epoch_timestamp_from_retweet)

    return retweets


def get_noise_news_ids():
    with open("data/news_id_ignore_list") as file:
        lines = file.readlines()
        return [line.strip() for line in lines]


def load_prop_graph(data_folder, news_source, news_label):
    news_graphs = pickle.load(open("{}/{}_{}_news_prop_graphs.pkl".format(data_folder, news_source, news_label), "rb"))
    return news_graphs


def remove_prop_graph_noise(news_graphs, noise_ids):
    noise_ids = set(noise_ids)
    return [graph for graph in news_graphs if graph.tweet_id not in noise_ids]


def sort_tweet_node_object_by_created_time(tweet_nodes: list):
    tweet_nodes.sort(key=lambda x: x.created_time)

    return tweet_nodes


def equal_samples(sample1, sample2):
    target_len = min(len(sample1), len(sample2))

    np.random.seed(0)

    np.random.shuffle(sample1)
    np.random.shuffle(sample2)

    return sample1[:target_len], sample2[:target_len]


def get_propagation_graphs(data_folder, news_source):
    fake_propagation_graphs = load_prop_graph(data_folder, news_source, "fake")
    real_propagation_graphs = load_prop_graph(data_folder, news_source, "real")

    print("Before filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
    print("Before filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))

    fake_propagation_graphs = remove_prop_graph_noise(fake_propagation_graphs, get_noise_news_ids())
    real_propagation_graphs = remove_prop_graph_noise(real_propagation_graphs, get_noise_news_ids())

    print("After filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
    print("After filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))
    print(flush=True)

    return fake_propagation_graphs, real_propagation_graphs


def get_numpy_array(list_of_list):
    np_array_lists = []
    for list_obj in list_of_list:
        np_array_lists.append(np.array(list_obj))

    return np.array(np_array_lists)


def print_stat_values(feature_name, values, short_feature_name=""):
    print("=========================================")
    print("Feature {} : {}".format(short_feature_name, feature_name))
    print("Min value : {}".format(min(values)))
    print("Max value : {}".format(max(values)))
    print("Mean value : {}".format(np.mean(np.array(values))))
    print("=========================================")
