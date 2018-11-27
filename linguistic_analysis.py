import numpy as np
import queue

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from analysis_util import get_propagation_graphs, equal_samples, get_numpy_array
from stat_test import perform_t_test
from structure_temp_analysis import get_post_tweet_deepest_cascade
from temporal_analysis import print_stat_values
from util.constants import REPLY_NODE, POST_NODE
from util.util import tweet_node


def tweet_text_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_result = analyzer.polarity_scores(text)
    return sentiment_result["compound"]


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
                    reply_diff_values.append(tweet_text_sentiment(node.text))

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
                reply_diff_values.append(tweet_text_sentiment(node.text))

    if len(reply_diff_values) == 0:
        return 0
    else:
        return np.mean(np.array(reply_diff_values))


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


if __name__ == "__main__":
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
