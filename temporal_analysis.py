import queue

import numpy as np

from analysis_util import sort_tweet_node_object_by_created_time, get_propagation_graphs, equal_samples
from stat_test import perform_t_test
from structure_temp_analysis import get_first_post_time, get_post_tweet_deepest_cascade
from util.constants import RETWEET_NODE, REPLY_NODE
from util.util import tweet_node


def get_avg_retweet_time_deepest_cascade(news_graph: tweet_node):
    deep_cascade, max_height = get_post_tweet_deepest_cascade(news_graph)
    return get_avg_time_between_retweets(deep_cascade)


def get_time_diff_post_time_last_retweet_time_deepest_cascade(news_graph: tweet_node):
    deep_cascade, max_height = get_post_tweet_deepest_cascade(news_graph)
    first_post_time = deep_cascade.created_time

    last_retweet_time = get_last_retweet_by_time(deep_cascade)
    return last_retweet_time - first_post_time


def get_avg_time_between_replies(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    retweet_diff_values = list()

    while q.qsize() != 0:
        node = q.get()
        for child in node.children:
            q.put(child)
            if node.node_type == RETWEET_NODE and child.node_type == REPLY_NODE:
                retweet_diff_values.append(child.created_time - node.created_time)

    if len(retweet_diff_values) == 0:
        return 0
    else:
        return np.mean(np.array(retweet_diff_values))


def get_avg_time_between_retweets(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)
    retweet_diff_values = list()

    while q.qsize() != 0:
        node = q.get()
        for child in node.retweet_children:
            q.put(child)
            if node.node_type == RETWEET_NODE and child.node_type == RETWEET_NODE:
                retweet_diff_values.append(child.created_time - node.created_time)

    if len(retweet_diff_values) == 0:
        return 0
    else:
        return np.mean(np.array(retweet_diff_values))


def get_last_retweet_by_time(news_graph: tweet_node):
    max_time = 0

    if news_graph:
        for node in news_graph.retweet_children:
            max_time = max(max_time, get_last_retweet_by_time(node))

    if news_graph and news_graph.created_time is not None:
        max_time = max(max_time, news_graph.created_time)

    return max_time


def get_first_retweet_by_time(news_graph: tweet_node):
    min_time = float("inf")

    if news_graph:
        for node in news_graph.retweet_children:
            min_time = min(min_time, get_first_retweet_by_time(node))

    if news_graph and news_graph.created_time is not None and news_graph.node_type == RETWEET_NODE:
        min_time = min(min_time, news_graph.created_time)

    return min_time


def get_time_diff_first_post_last_retweet(news_graph: tweet_node):
    first_post_time = get_first_post_time(news_graph)
    last_retweet_time = get_last_retweet_by_time(news_graph)
    return last_retweet_time - first_post_time


def get_time_diff_first_post_first_retweet(news_graph: tweet_node):
    first_post_time = get_first_post_time(news_graph)
    first_retweet_time = get_first_retweet_by_time(news_graph)

    if first_retweet_time == float("inf"):
        return 0

    return first_retweet_time - first_post_time


def get_time_diff_first_last_post_tweet(news_graph: tweet_node):
    post_tweets = list(news_graph.children)

    if len(post_tweets) <= 1:
        print("only one tweet")
        return 0

    post_tweets = sort_tweet_node_object_by_created_time(post_tweets)

    return post_tweets[len(post_tweets) - 1].created_time - post_tweets[0].created_time


def get_average_time_between_post_tweets(news_graph: tweet_node):
    post_tweets = list(news_graph.children)

    if len(post_tweets) <= 1:
        print("only one tweet")
        return 0

    post_tweets = sort_tweet_node_object_by_created_time(post_tweets)

    time_diff = 0

    for i in range(1, len(post_tweets)):
        time_diff += (post_tweets[i].created_time - post_tweets[i - 1].created_time)

    return float(time_diff) / (len(post_tweets) - 1)


def get_stats_for_features(news_graps: list, get_feature_fun_ref, print=False, feature_name=None):
    result = []
    for graph in news_graps:
        result.append(get_feature_fun_ref(graph))

    if print:
        print_stat_values(feature_name, result)

    return result


def print_stat_values(feature_name, values):
    print("=========================================")
    print("Feature : {}".format(feature_name))
    print("Min value : {}".format(min(values)))
    print("Max value : {}".format(max(values)))
    print("Mean value : {}".format(np.mean(np.array(values))))
    print("=========================================")


def graph_has_retweet(news_graph: tweet_node):
    post_tweets = news_graph.children

    for post in post_tweets:
        if len(post.retweet_children) > 0:
            return True

    return False


def count_graph_with_no_retweets(news_graphs: list):
    count = 0

    for prop_graph in news_graphs:
        if not graph_has_retweet(prop_graph):
            count += 1

    print("Graph with no retweets : {}".format(count))


if __name__ == "__main__":
    fake_prop_graph, real_prop_graph = get_propagation_graphs("politifact")
    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    print("After equal random sampling")
    print("Fake samples : {}  Real samples : {}".format(len(fake_prop_graph), len(real_prop_graph)))
    # feature_name = "avg. time between post tweets"
    # feature_name = "time_diff_first_last_post_tweet"
    # feature_name = "time_diff_first_post_last_retweet"
    # feature_name = "time_diff_first_post_first_retweet"
    # feature_name = "avg_time_between_retweets"

    # feature_name = "get_avg_retweet_time_deepest_cascade"
    feature_name = "get_time_diff_post_time_last_retweet_time_deepest_cascade"

    # function_ref = get_average_time_between_post_tweets
    # function_ref = get_time_diff_first_last_post_tweet
    # function_ref = get_time_diff_first_post_last_retweet
    # function_ref = get_time_diff_first_post_first_retweet
    # function_ref = get_avg_time_between_retweets
    # function_ref = get_avg_retweet_time_deepest_cascade
    function_ref = get_time_diff_post_time_last_retweet_time_deepest_cascade

    # count_graph_with_no_retweets(fake_prop_graph)
    # count_graph_with_no_retweets(real_prop_graph)

    print("FAKE")
    fake_prop_features = get_stats_for_features(fake_prop_graph, function_ref, print=True,
                                                feature_name=feature_name)

    print("REAL")
    real_prop_features = get_stats_for_features(real_prop_graph, function_ref, print=True,
                                                feature_name=feature_name)

    perform_t_test(fake_prop_features, real_prop_features)
