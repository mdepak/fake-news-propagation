import time

import numpy as np

from preprocess_data import load_prop_graph
from util.util import tweet_node

def get_num_cascade(node:tweet_node, edge_type="retweet"):
    if edge_type == "retweet":
        return len(node.retweet_children)
    elif edge_type == "reply":
        return len(node.reply_children)
    else:
        return len(node.children)


def get_temp_num_cascade(node: tweet_node, edge_type ="retweet", max_time=time.time()):

    if edge_type == "retweet":
        children = node.retweet_children
    elif edge_type == "reply":
        children = node.reply_children
    else:
        children = node.children

    cascade_count = 0

    for child in children:
        if child.created_time <= max_time:
            cascade_count += 1

    return cascade_count





def get_tree_height(node, edge_type="retweet"):
    if node is None:
        return 0

    max_child_height = 0

    if edge_type == "retweet":
        children = node.retweet_children
    elif edge_type == "reply":
        children = node.reply_children
    else:
        children = node.children

    for child in children:
        max_child_height = max(max_child_height, get_tree_height(child, edge_type))

    return max_child_height + 1


def get_nodes_count(node: tweet_node, edge_type="retweet"):
    if node is None:
        return 0

    node_count = 0

    if edge_type == "retweet":
        children = node.retweet_children
    elif edge_type == "reply":
        children = node.reply_children
    else:
        children = node.children

    for child in children:
        node_count += get_nodes_count(child, edge_type)

    return node_count + 1


def get_temporal_nodes_count(node: tweet_node, edge_type="retweet", max_time=time.time()):
    if node is None or (node.created_time is not None and node.created_time > max_time):
        return 0

    node_count = 0

    if edge_type == "retweet":
        children = node.retweet_children
    elif edge_type == "reply":
        children = node.reply_children
    else:
        children = node.children

    for child in children:
        node_count += get_temporal_nodes_count(child, edge_type, max_time)

    return node_count + 1


def get_node_size_by_time(prop_graphs: list, edge_type: str, time_interval_sec: list):
    temporal_tree_node_size = []
    for news_node in prop_graphs:
        temp_node_sizes = []
        first_post_time = get_first_post_time(news_node)
        for time_limit in time_interval_sec:
            node_count = get_temporal_nodes_count(news_node, edge_type, first_post_time + time_limit)
            temp_node_sizes.append(node_count)

        temporal_tree_node_size.append(temp_node_sizes)

    return temporal_tree_node_size


def get_temporal_tree_height(node: tweet_node, edge_type="retweet", max_time=time.time()):
    if node is None or (node.created_time is not None and node.created_time > max_time):
        return 0

    max_child_height = 0

    if edge_type == "retweet":
        children = node.retweet_children
    elif edge_type == "reply":
        children = node.reply_children
    else:
        children = node.children

    for child in children:
        max_child_height = max(max_child_height, get_temporal_tree_height(child, edge_type, max_time))

    return max_child_height + 1


def get_num_cascades_by_time(prop_graphs: list, edge_type: str, time_interval_sec: list):
    temporal_num_cascades = []
    for news_node in prop_graphs:
        temp_cascade_num = []
        first_post_time = get_first_post_time(news_node)
        for time_limit in time_interval_sec:
            node_count = get_temp_num_cascade(news_node, edge_type, first_post_time + time_limit)
            temp_cascade_num.append(node_count)

        temporal_num_cascades.append(temp_cascade_num)

    return temporal_num_cascades


def analyze_height(news_graphs: list, edge_type):
    heights = []

    for news_node in news_graphs:
        heights.append(get_tree_height(news_node, edge_type))

    print("max", max(heights))
    print("min", min(heights))
    print("avg", np.mean(heights))


def analyze_cascade(news_graphs: list, edge_type):
    heights = []

    for news_node in news_graphs:
        heights.append(get_num_cascade(news_node, edge_type))

    print("max", max(heights))
    print("min", min(heights))
    print("avg", np.mean(heights))


def analyze_node_count(news_graphs: list, edge_type):
    node_counts = []

    for news_node in news_graphs:
        node_counts.append(get_nodes_count(news_node, edge_type))

    print("max", max(node_counts))
    print("min", min(node_counts))
    print("avg", np.mean(node_counts))


def get_height_by_time(prop_graphs: list, edge_type: str, time_interval_sec: list):
    temporal_tree_height = []
    for news_node in prop_graphs:
        temp_heights = []
        first_post_time = get_first_post_time(news_node)
        for time_limit in time_interval_sec:
            height = get_temporal_tree_height(news_node, edge_type, first_post_time + time_limit)
            temp_heights.append(height)

        temporal_tree_height.append(temp_heights)

    return temporal_tree_height


def analyze_height_by_time(prop_graphs: list, edge_type: str, time_interval_sec: list):
    temporal_tree_height = get_height_by_time(prop_graphs, edge_type, time_interval_sec)

    temporal_tree_height = np.array([np.array(val) for val in temporal_tree_height])

    for idx, time_limit_sec in enumerate(time_interval_sec):
        heights_at_time = temporal_tree_height[:, idx]
        print("Time limit: {}".format(time_limit_sec))
        print("Min height : {}".format(np.min(heights_at_time)))
        print("Max height : {}".format(np.max(heights_at_time)))
        print("Mean height : {}".format(np.mean(heights_at_time)))
        print(flush=True)


def analyze_cascade_num_by_time(prop_graphs: list, edge_type: str, time_interval_sec: list):
    temporal_cascade_num = get_num_cascades_by_time(prop_graphs, edge_type, time_interval_sec)

    temporal_cascade_num = np.array([np.array(val) for val in temporal_cascade_num])

    for idx, time_limit_sec in enumerate(time_interval_sec):
        heights_at_time = temporal_cascade_num[:, idx]
        print("Time limit: {}".format(time_limit_sec))
        print("Min num cascade : {}".format(np.min(heights_at_time)))
        print("Max num cascade : {}".format(np.max(heights_at_time)))
        print("Mean num cascade : {}".format(np.mean(heights_at_time)))
        print(flush=True)



def analyze_node_size_by_time(prop_graphs: list, edge_type: str, time_interval_sec: list):
    temporal_tree_node_sizes = get_node_size_by_time(prop_graphs, edge_type, time_interval_sec)

    temporal_tree_node_sizes = np.array([np.array(val) for val in temporal_tree_node_sizes])

    for idx, time_limit_sec in enumerate(time_interval_sec):
        heights_at_time = temporal_tree_node_sizes[:, idx]
        print("Time limit: {}".format(time_limit_sec))
        print("Min node size : {}".format(np.min(heights_at_time)))
        print("Max node size : {}".format(np.max(heights_at_time)))
        print("Mean node size : {}".format(np.mean(heights_at_time)))
        print(flush=True)


def get_first_post_time(node: tweet_node):
    first_post_time = time.time()

    for child in node.children:
        first_post_time = min(first_post_time, child.created_time)

    return first_post_time


if __name__ == "__main__":
    propagation_graphs = load_prop_graph("politifact", "fake")

    RETWEET_EDGE = "retweet"
    REPLY_EDGE = "reply"

    analyze_cascade(propagation_graphs, RETWEET_EDGE)
    analyze_cascade_num_by_time(propagation_graphs, RETWEET_EDGE, [60, 300, 600, 900, 18000, 36000, 72000]
                                )

    exit(1)
    analyze_height(propagation_graphs, RETWEET_EDGE)
    analyze_height_by_time(propagation_graphs, RETWEET_EDGE, [60, 300, 600, 900, 18000, 36000, 72000])

    analyze_node_count(propagation_graphs, RETWEET_EDGE)
    analyze_node_size_by_time(propagation_graphs, RETWEET_EDGE, [60, 300, 600, 900, 18000, 36000, 72000])