import pickle

import numpy as np
import queue

from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.spatial.distance import cosine
from analysis_util import get_propagation_graphs, equal_samples, get_numpy_array, BaseFeatureHelper, \
    get_sample_feature_value
from stat_test import perform_t_test
from structure_temp_analysis import get_post_tweet_deepest_cascade
from temporal_analysis import print_stat_values
from util.constants import REPLY_NODE, POST_NODE
from util.util import tweet_node

all_reply_id_sentiment_score_dict = pickle.load(open("{}/all_reply_id_sentiment_result.pkl"
                                                     .format("data/pre_process_data/vader_sentiment"), "rb"))


def tweet_text_sentiment(reply_id):
    if reply_id in all_reply_id_sentiment_score_dict:
        return all_reply_id_sentiment_score_dict[reply_id]["compound"]
    else:
        return 0


# def tweet_text_sentiment(text):
#     analyzer = SentimentIntensityAnalyzer()
#     sentiment_result = analyzer.polarity_scores(text)
#     return sentiment_result["compound"]


def get_deepest_cascade_reply_nodes_avg_sentiment(prop_graph: tweet_node):
    deep_cascade, max_height = get_post_tweet_deepest_cascade(prop_graph)

    return get_reply_nodes_average_sentiment(deep_cascade)


def get_deepest_cascade_first_level_reply_sentiment(prop_graph: tweet_node):
    deep_cascade, max_height = get_post_tweet_deepest_cascade(prop_graph)
    return get_first_reply_nodes_average_sentiment(deep_cascade)


def get_first_reply_nodes_average_sentiment(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    reply_diff_values = list()

    while q.qsize() != 0:
        node = q.get()
        for child in node.reply_children:
            q.put(child)

            if child.node_type == REPLY_NODE and node.node_type == POST_NODE:
                if node.text:
                    reply_diff_values.append(tweet_text_sentiment(child.tweet_id))

    if len(reply_diff_values) == 0:
        return 0
    else:
        return np.mean(np.array(reply_diff_values))


def get_reply_nodes_average_sentiment(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    reply_diff_values = list()

    while q.qsize() != 0:
        node = q.get()
        for child in node.reply_children:
            q.put(child)

        if node.node_type == REPLY_NODE:
            if node.text:
                reply_diff_values.append(tweet_text_sentiment(node.tweet_id))

    if len(reply_diff_values) == 0:
        return 0
    else:
        return np.mean(np.array(reply_diff_values))


def get_cosine_similarity(reply_node1, reply_node2, reply_id_index_dict, reply_lat_embeddings):
    try:
        if reply_node1 in reply_id_index_dict and reply_node2 in reply_id_index_dict:
            reply1_idx = reply_id_index_dict[reply_node1]
            reply2_idx = reply_id_index_dict[reply_node2]

            return cosine(reply_lat_embeddings[reply1_idx], reply_lat_embeddings[reply2_idx])

        else:
            return 0
    except:
        return 0


def get_supporting_opposing_replies_ratio(prop_graph: tweet_node, news_source, label):
    q = queue.Queue()

    q.put(prop_graph)
    similarity_values = list()

    reply_id_index_dict = pickle.load(
        open("data/pre_process_data/elmo_features/{}_{}_reply_id_latent_mat_index.pkl".format(news_source, label),
             "rb"))
    reply_content_latent_embeddings = pickle.load(
        open("data/pre_process_data/elmo_features/{}_{}_elmo_lat_embeddings.pkl".format(news_source, label), "rb"))

    while q.qsize() != 0:
        node = q.get()
        for child in node.reply_children:
            q.put(child)

            if node.node_type == REPLY_NODE and child.node_type == REPLY_NODE:
                similarity_values.append(get_cosine_similarity(node.tweet_id, child.tweet_id,
                                                               reply_id_index_dict, reply_content_latent_embeddings))

    if len(similarity_values) == 0:
        return 0
    else:
        supporting = 1
        opposing = 1

        for value in similarity_values:
            if value > 0.5:
                supporting += 1
            else:
                opposing += 1

        return supporting / opposing


def get_reply_nodes_sentiment_ratio(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    reply_diff_values = list()

    while q.qsize() != 0:
        node = q.get()
        for child in node.reply_children:
            q.put(child)

        if node.node_type == REPLY_NODE:
            if node.text:
                reply_diff_values.append(tweet_text_sentiment(node.tweet_id))

    if len(reply_diff_values) == 0:
        return 0
    else:
        positive_sentiment = 1
        negative_sentiment = 1
        for value in reply_diff_values:
            if value > 0.05:
                positive_sentiment += 1
            elif value < -0.05:
                negative_sentiment += 1

        return positive_sentiment / negative_sentiment


def get_stats_for_features(news_graps: list, get_feature_fun_ref, print=False, feature_name=None):
    result = []
    for graph in news_graps:
        result.append(get_feature_fun_ref(graph))

    if print:
        print_stat_values(feature_name, result)

    return result


def get_all_linguistic_features(news_graphs, micro_features, macro_features):
    all_features = []

    if macro_features:
        retweet_function_references = []

        for function_reference in retweet_function_references:
            features_set = get_stats_for_features(news_graphs, function_reference, print=False, feature_name=None)
            all_features.append(features_set)

    if micro_features:

        reply_function_references = [get_reply_nodes_average_sentiment, get_first_reply_nodes_average_sentiment,
                                     get_deepest_cascade_reply_nodes_avg_sentiment,
                                     get_deepest_cascade_first_level_reply_sentiment]

        for function_reference in reply_function_references:
            features_set = get_stats_for_features(news_graphs, function_reference, print=False, feature_name=None)
            all_features.append(features_set)

    return np.transpose(get_numpy_array(all_features))


def dump_tweet_reply_sentiment(data_dir, out_dir):
    reply_id_content_dict = dict()

    reply_id_content_dict.update(pickle.load(
        open("{}/{}_{}_reply_id_content_dict.pkl".format(data_dir, "politifact", "fake"), "rb")))

    reply_id_content_dict.update(pickle.load(
        open("{}/{}_{}_reply_id_content_dict.pkl".format(data_dir, "politifact", "real"), "rb")))

    reply_id_content_dict.update(pickle.load(
        open("{}/{}_{}_reply_id_content_dict.pkl".format(data_dir, "gossipcop", "fake"), "rb")))

    reply_id_content_dict.update(pickle.load(
        open("{}/{}_{}_reply_id_content_dict.pkl".format(data_dir, "gossipcop", "real"), "rb")))

    print("Total no. of replies : {}".format(len(reply_id_content_dict)))

    analyzer = SentimentIntensityAnalyzer()

    reply_id_sentiment_output = dict()

    for reply_id, content in tqdm(reply_id_content_dict.items()):
        sentiment_result = analyzer.polarity_scores(content)
        reply_id_sentiment_output[reply_id] = sentiment_result

    pickle.dump(reply_id_sentiment_output, open("{}/all_reply_id_sentiment_result.pkl".format(out_dir), "wb"))


class LinguisticFeatureHelper(BaseFeatureHelper):

    def get_micro_feature_method_references(self):
        method_refs = [get_reply_nodes_sentiment_ratio,
                       get_reply_nodes_average_sentiment,
                       get_first_reply_nodes_average_sentiment,
                       get_deepest_cascade_reply_nodes_avg_sentiment,
                       get_deepest_cascade_first_level_reply_sentiment,
                       get_supporting_opposing_replies_ratio]

        return method_refs

    def get_micro_feature_method_names(self):
        feature_names = ["Sentiment ratio of all replies",
                         "Average sentiment of all replies",
                         "Average sentiment of first level replies",
                         "Average sentiment of replies in deepest cascade",
                         "Average setiment of first level replies in deepest cascade",
                         "Supporting or opposing ratio"]

        return feature_names

    def get_micro_feature_short_names(self):
        feature_names = ["L1", "L2", "L3", "L4", "L5","L6"]
        return feature_names

    def get_macro_feature_method_references(self):
        method_refs = []

        return method_refs

    def get_macro_feature_method_names(self):
        feature_names = []

        return feature_names

    feature_names = []

    def get_macro_feature_short_names(self):
        feature_names = []
        return feature_names

    def get_features_array(self, prop_graphs, micro_features, macro_features, news_source=None, label=None):
        function_refs = []

        if micro_features:
            function_refs.extend(self.get_micro_feature_method_references())

        if len(function_refs) == 0:
            return None

        all_features = []

        for idx in range(len(function_refs) - 1):
            features_set = get_sample_feature_value(prop_graphs, function_refs[idx])
            all_features.append(features_set)

        all_features.append(get_feature_involving_additional_args(prop_graphs, function_refs[-1],news_source, label))

        return np.transpose(get_numpy_array(all_features))



def get_feature_involving_additional_args(prop_graphs, function_reference, news_source, label):
    feature_values = []
    for prop_graph in prop_graphs:
        feature_values.append(function_reference(prop_graph, news_source, label))

    return feature_values


if __name__ == "__main__":
    dump_tweet_reply_sentiment("data/pre_process_data", "data/pre_process_data/vader_sentiment")
    exit(1)

    fake_prop_graph, real_prop_graph = get_propagation_graphs("politifact")
    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    print("After equal random sampling")
    print("Fake samples : {}  Real samples : {}".format(len(fake_prop_graph), len(real_prop_graph)))

    # feature_name = "get_reply_nodes_average_sentiment"
    feature_name = "get_first_reply_nodes_average_sentiment"
    # feature_name = "get_deepest_cascade_reply_nodes_avg_sentiment"
    # feature_name = "get_deepest_cascade_first_level_reply_sentiment"

    # function_ref = get_reply_nodes_average_sentiment
    function_ref = get_first_reply_nodes_average_sentiment
    # function_ref = get_deepest_cascade_reply_nodes_avg_sentiment
    # function_ref = get_deepest_cascade_first_level_reply_sentiment

    # count_graph_with_no_retweets(fake_prop_graph)
    # count_graph_with_no_retweets(real_prop_graph)

    print("FAKE")
    fake_prop_features = get_stats_for_features(fake_prop_graph, function_ref, print=True, feature_name=feature_name)

    print("REAL")
    real_prop_features = get_stats_for_features(real_prop_graph, function_ref, print=True, feature_name=feature_name)

    perform_t_test(fake_prop_features, real_prop_features)
