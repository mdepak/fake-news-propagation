import pickle
import queue
import time
from pathlib import Path

import numpy as np

from analysis_util import get_numpy_array, BaseFeatureHelper, \
    get_sample_feature_value
from util.constants import NEWS_ROOT_NODE, RETWEET_EDGE, REPLY_EDGE, RETWEET_NODE, REPLY_NODE
from util.util import tweet_node


def get_post_tweet_deepest_cascade(prop_graph: tweet_node, edge_type=RETWEET_EDGE):
    max_height = 0
    max_height_node = None

    for node in prop_graph.children:
        height = get_tree_height(node, edge_type)
        if height > max_height:
            max_height = height
            max_height_node = node

    return max_height_node, max_height


def get_num_cascade(node: tweet_node, edge_type="retweet"):
    if edge_type == "retweet":
        return len(node.retweet_children)
    elif edge_type == "reply":
        return len(node.reply_children)
    else:
        return len(node.children)


def get_temp_num_cascade(node: tweet_node, edge_type="retweet", max_time=time.time()):
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


def get_node_count_deepest_cascade(news_graphs: tweet_node, edge_type):
    node_counts = []

    for prop_graph in news_graphs:
        max_height_node, max_height = get_post_tweet_deepest_cascade(prop_graph)

        node_counts.append(get_nodes_count(max_height_node, edge_type))

    return node_counts


def get_max_outdegree(node: tweet_node, edge_type="retweet"):
    if node is None:
        return 0

    if edge_type == "retweet":
        children = node.retweet_children
    elif edge_type == "reply":
        children = node.reply_children
    else:
        children = node.children

    if node.node_type == NEWS_ROOT_NODE:
        max_outdegree = 0
    else:
        max_outdegree = len(children)

    for child in children:
        max_outdegree = max(max_outdegree, get_max_outdegree(child, edge_type))

    return max_outdegree


def get_max_out_degree_node(node: tweet_node, edge_type=RETWEET_EDGE):
    if node is None:
        return None

    if edge_type == "retweet":
        children = node.retweet_children
    elif edge_type == "reply":
        children = node.reply_children
    else:
        children = node.children

    if node.node_type == NEWS_ROOT_NODE:
        max_outdegree_node, max_out_degree = None, 0

    else:
        max_outdegree_node, max_out_degree = node, len(children)

    for child in children:
        child_max_out_degree_node, child_max_out_degree = get_max_out_degree_node(child, edge_type)
        if child_max_out_degree > max_out_degree:
            max_out_degree = child_max_out_degree
            max_outdegree_node = child_max_out_degree_node

    return max_outdegree_node, max_out_degree


def get_target_node_level(root_node: tweet_node, target_node, level=0):
    if root_node is None:
        return 0

    if root_node.tweet_id == target_node.tweet_id:
        return level

    for child in root_node.children:
        res_level = get_target_node_level(child, target_node, level + 1)
        if res_level != 0:
            return res_level

    return 0


def get_depth_of_max_degree_node(prop_graph, edge_type=RETWEET_EDGE):
    max_out_degree_node, max_out_degree = get_max_out_degree_node(prop_graph, edge_type)

    if max_out_degree_node is None:
        return 0

    return get_target_node_level(prop_graph, max_out_degree_node, 0)


def get_max_out_degree_depths(prop_graphs, edge_type=RETWEET_EDGE):
    out_degree_depths = []

    for news_node in prop_graphs:
        out_degree_depths.append(get_depth_of_max_degree_node(news_node, edge_type))

    return out_degree_depths


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


def get_tree_heights(news_graphs: list, edge_type):
    heights = []

    for news_node in news_graphs:
        heights.append(get_tree_height(news_node, edge_type))

    return heights


def analyze_height(news_graphs: list, edge_type):
    heights = get_tree_heights(news_graphs, edge_type)

    print("----HEIGHT-----")

    print("max", max(heights))
    print("min", min(heights))
    print("avg", np.mean(heights))


def get_max_outdegrees(news_graphs: list, edge_type):
    max_outdegrees = []

    for news_node in news_graphs:
        max_outdegrees.append(get_max_outdegree(news_node, edge_type))

    return max_outdegrees


def analyze_max_outdegree(news_graphs: list, edge_type):
    max_outdegrees = get_max_outdegrees(news_graphs, edge_type)
    print("-----MAX - OUT DEGREE -----")
    print("max", max(max_outdegrees))
    print("min", min(max_outdegrees))
    print("avg", np.mean(max_outdegrees))


def get_prop_graps_cascade_num(news_graphs: list, edge_type):
    cascade_num = []

    for news_node in news_graphs:
        cascade_num.append(get_num_cascade(news_node, edge_type))

    return cascade_num


def analyze_cascade(news_graphs: list, edge_type):
    cascade_num = get_prop_graps_cascade_num(news_graphs, edge_type)

    print("-----CASCADE-----")
    print("max", max(cascade_num))
    print("min", min(cascade_num))
    print("avg", np.mean(cascade_num))


def get_prop_graphs_node_counts(news_graphs: list, edge_type):
    node_counts = []

    for news_node in news_graphs:
        node_counts.append(get_nodes_count(news_node, edge_type))

    return node_counts


def analyze_node_count(news_graphs: list, edge_type):
    node_counts = get_prop_graphs_node_counts(news_graphs, edge_type)

    print("----NODE SIZE-----")

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


def get_num_of_cascades_with_retweets(root_node: tweet_node):
    num_cascades = 0
    for node in root_node.retweet_children:
        if len(node.retweet_children) > 0:
            num_cascades += 1

    return num_cascades


def get_prop_graphs_num_of_cascades_with_retweets(prop_graphs, edge_type=RETWEET_EDGE):
    return get_sample_feature_value(prop_graphs, get_num_of_cascades_with_retweets)


def get_fraction_of_cascades_with_retweets(root_node: tweet_node):
    total_cascades = len(root_node.retweet_children)

    cascade_with_retweet = 0
    for node in root_node.retweet_children:
        if len(node.retweet_children) > 0:
            cascade_with_retweet += 1

    return cascade_with_retweet / total_cascades


def get_prop_graphs_fraction_of_cascades_with_retweets(prop_graphs, edge_type=RETWEET_EDGE):
    return get_sample_feature_value(prop_graphs, get_fraction_of_cascades_with_retweets)


def get_num_of_cascades_with_replies(root_node: tweet_node):
    num_cascades = 0
    for node in root_node.reply_children:
        if len(node.reply_children) > 0:
            num_cascades += 1

    return num_cascades


def get_prop_graphs_num_of_cascades_with_replies(prop_graphs, edge_type=RETWEET_EDGE):
    return get_sample_feature_value(prop_graphs, get_num_of_cascades_with_replies)


def get_fraction_of_cascades_with_replies(root_node: tweet_node):
    total_cascades = len(root_node.reply_children)

    cascade_with_replies = 0
    for node in root_node.reply_children:
        if len(node.reply_children) > 0:
            cascade_with_replies += 1

    return cascade_with_replies / total_cascades


def get_users_in_network(prop_graph: tweet_node, edge_type=None):
    q = queue.Queue()

    q.put(prop_graph)

    users_list = list()

    while q.qsize() != 0:
        node = q.get()

        if edge_type == RETWEET_EDGE:
            children = node.retweet_children
        elif edge_type == REPLY_EDGE:
            children = node.reply_children
        else:
            children = node.children

        for child in children:
            q.put(child)
            if child.user_id is not None:
                users_list.append(child.user_id)

    return users_list


def get_users_replying_in_prop_graph(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)

    users_list = list()

    while q.qsize() != 0:
        node = q.get()

        for child in node.reply_children:
            q.put(child)
            if child.node_type == REPLY_NODE and child.user_id is not None:
                users_list.append(child.user_id)

    return users_list


def get_users_retweeting_in_prop_graph(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)

    users_list = list()

    while q.qsize() != 0:
        node = q.get()

        for child in node.retweet_children:
            q.put(child)
            if child.node_type == RETWEET_NODE and child.user_id is not None:
                users_list.append(child.user_id)

    return users_list


def get_user_names_retweeting_in_prop_graph(prop_graph: tweet_node):
    q = queue.Queue()

    q.put(prop_graph)

    users_list = list()

    while q.qsize() != 0:
        node = q.get()

        for child in node.retweet_children:
            q.put(child)
            if child.node_type == RETWEET_NODE and child.user_name is not None:
                users_list.append(child.user_name)

    return users_list


def get_num_user_retweet_and_reply(prop_graph: tweet_node):
    retweet_users = set(get_users_retweeting_in_prop_graph(prop_graph))
    replying_users = set(get_users_replying_in_prop_graph(prop_graph))

    return len(retweet_users.intersection(replying_users))


def get_ratio_of_retweet_to_reply(prop_graph: tweet_node):
    retweet_users = set(get_users_retweeting_in_prop_graph(prop_graph))
    replying_users = set(get_users_replying_in_prop_graph(prop_graph))

    return (len(retweet_users) + 1) / (len(replying_users) + 1)


def get_prop_graphs_num_user_retweet_and_reply(prop_graphs, edge_type=None):
    return get_sample_feature_value(prop_graphs, get_num_user_retweet_and_reply)


def get_prop_graphs_ratio_of_retweet_to_reply(prop_graphs, edge_type=None):
    return get_sample_feature_value(prop_graphs, get_ratio_of_retweet_to_reply)


def get_unique_users_in_graph(prop_graph: tweet_node, edge_type=None):
    user_list = get_users_in_network(prop_graph, edge_type)
    return len(set(user_list))


def get_fraction_of_unique_users(prop_graph: tweet_node, edge_type=None):
    user_list = get_users_in_network(prop_graph, edge_type)
    try:
        return len(set(user_list)) / len(user_list)
    except:
        print("Exception in fraction of unique users")
        return 0


def get_num_bot_users(prop_graph: tweet_node):
    q = queue.Queue()
    q.put(prop_graph)

    num_bot_users = 0

    while q.qsize() != 0:
        node = q.get()

        for child in node.retweet_children:
            q.put(child)
            if child.node_type == RETWEET_NODE and child.user_id is not None:
                if child.botometer_score and child.botometer_score > 0.5:
                    num_bot_users += 1

    return num_bot_users


def get_fraction_of_bot_users_retweeting(prop_graph: tweet_node):
    q = queue.Queue()
    q.put(prop_graph)

    num_bot_users = 1
    num_human_users = 1

    while q.qsize() != 0:
        node = q.get()

        for child in node.retweet_children:
            q.put(child)
            if child.node_type == RETWEET_NODE and child.user_id is not None:
                if child.botometer_score:
                    if child.botometer_score > 0.5:
                        num_bot_users += 1
                    else:
                        num_human_users += 1

    return num_bot_users / (num_human_users + num_bot_users)


def get_prop_graphs_num_bot_users_retweeting(prop_graphs: tweet_node, edge_type=None):
    global user_id_bot_score_dict
    return get_sample_feature_value(prop_graphs, get_num_bot_users)


def get_prop_graphs_fraction_of_bot_users_retweeting(prop_graphs: tweet_node, edge_type=None):
    return get_sample_feature_value(prop_graphs, get_fraction_of_bot_users_retweeting)


def get_breadth_at_each_level(prop_graph, edge_type=RETWEET_EDGE):
    q1 = queue.Queue()
    q2 = queue.Queue()

    q1.put(prop_graph)

    level_breadths = []

    while q1.qsize() != 0 or q2.qsize() != 0:

        if q1.qsize() != 0:
            level_breadths.append(q1.qsize())

        while q1.qsize() != 0:
            node = q1.get()

            if edge_type == RETWEET_EDGE:
                children = node.retweet_children
            elif edge_type == REPLY_EDGE:
                children = node.reply_children
            else:
                children = node.children

            for child in children:
                q2.put(child)

        if q2.qsize() != 0:
            level_breadths.append(q2.qsize())

        while q2.qsize() != 0:
            node = q2.get()

            if edge_type == RETWEET_EDGE:
                children = node.retweet_children
            elif edge_type == REPLY_EDGE:
                children = node.reply_children
            else:
                children = node.children

            for child in children:
                q1.put(child)

    return max(level_breadths)


def get_prop_graphs_max_breadth(prop_graphs, edge_type=RETWEET_EDGE):
    return get_sample_feature_value(prop_graphs, get_breadth_at_each_level)


def get_prop_graphs_num_unique_users(prop_graphs, edge_type=RETWEET_EDGE):
    unique_users_cnts = []

    for graph in prop_graphs:
        unique_users_cnts.append(get_unique_users_in_graph(graph, edge_type))

    return unique_users_cnts


def get_prop_graphs_fraction_of_unique_users(prop_graphs, edge_type=RETWEET_EDGE):
    unique_users_fract_cnts = []

    for graph in prop_graphs:
        unique_users_fract_cnts.append(get_fraction_of_unique_users(graph, edge_type))

    return unique_users_fract_cnts


def get_prop_graphs_fraction_of_cascades_with_replies(prop_graphs, edge_type=RETWEET_EDGE):
    return get_sample_feature_value(prop_graphs, get_fraction_of_cascades_with_replies)


def get_prop_graphs_min_time_to_reach_level_1(news_graphs: list, edge_type=None):
    return get_sample_feature_value(news_graphs, get_min_time_to_reach_level_1)


def get_prop_graphs_min_time_to_reach_level_2(news_graphs: list, edge_type=None):
    return get_sample_feature_value(news_graphs, get_min_time_to_reach_level_2)


def get_min_time_to_reach_level_1(new_graph: tweet_node):
    return get_min_time_to_reach_level(new_graph, 1)


def get_min_time_to_reach_level_2(news_graph: tweet_node):
    return get_min_time_to_reach_level(news_graph, 2)


def get_min_time_to_reach_level(new_graph: tweet_node, target_depth):
    time_to_reach_depth = []
    for post_node in new_graph.retweet_children:
        post_time = post_node.created_time
        level_node_times = dfs_traverse(post_node, 0, target_depth)
        if len(level_node_times) > 0:
            time_to_reach_depth.append(min(level_node_times) - post_time)

    if len(time_to_reach_depth) > 0:
        return np.mean(time_to_reach_depth)
    else:
        return 0


def get_unique_users_untill_level(new_graph: tweet_node, target_depth):
    dfs_traverse_get_users(new_graph, target_depth)


def dfs_traverse(node: tweet_node, level: int, target: int):
    result = []

    if level == target:
        return [node.created_time]

    elif level > target:
        return None

    else:
        for child in node.retweet_children:
            level_nodes = dfs_traverse(child, level + 1, target)
            if level_nodes:
                result.extend(level_nodes)

    return result


def get_num_unique_users_under_level_2(node: tweet_node, edge_type=None):
    return len(dfs_traverse_get_users(node, 0, 2))


def get_num_unique_users_under_level_4(node: tweet_node, edge_type=None):
    return len(dfs_traverse_get_users(node, 0, 4))


def get_prop_graphs_num_unique_user_under_level_2(prop_graphs, edge_type=RETWEET_EDGE):
    return get_sample_feature_value(prop_graphs, get_num_unique_users_under_level_2)


def get_prop_graphs_num_unique_user_under_level_4(prop_graphs, edge_type=RETWEET_EDGE):
    return get_sample_feature_value(prop_graphs, get_num_unique_users_under_level_4)


def dfs_traverse_get_users(node: tweet_node, level: int, target: int):
    result = list()

    if level > target:
        return None

    else:
        result.append(node.user_id)

        for child in node.retweet_children:
            level_nodes = dfs_traverse(child, level + 1, target)
            if level_nodes:
                result.extend(level_nodes)

    return result


def get_all_structural_features(news_graphs, micro_features, macro_features):
    all_features = []
    target_edge_type = RETWEET_EDGE

    if macro_features:
        retweet_function_references = [get_tree_heights, get_prop_graphs_node_counts, get_prop_graps_cascade_num,
                                       get_max_outdegrees, get_num_of_cascades_with_retweets,
                                       get_fraction_of_cascades_with_retweets]
        for function_ref in retweet_function_references:
            features = function_ref(news_graphs, target_edge_type)
            all_features.append(features)

    if micro_features:
        target_edge_type = REPLY_EDGE

        reply_function_references = [get_tree_heights, get_prop_graphs_node_counts, get_max_outdegrees]
        for function_ref in reply_function_references:
            features = function_ref(news_graphs, target_edge_type)
            all_features.append(features)

    return np.transpose(get_numpy_array(all_features))


class StructureFeatureHelper(BaseFeatureHelper):

    def get_feature_group_name(self):
        return "struct"

    def get_micro_feature_method_references(self):
        method_refs = [get_tree_heights, get_prop_graphs_node_counts, get_max_outdegrees,
                       get_prop_graphs_num_of_cascades_with_replies,
                       get_prop_graphs_fraction_of_cascades_with_replies]

        return method_refs

    def get_micro_feature_method_names(self):
        feature_names = ["Micro - Tree depth", "Micro - No of nodes", "Micro - Maximum out degree",
                         "No. of cascades with replies", "Fraction of cascades with replies"]
        return feature_names

    def get_micro_feature_short_names(self):
        feature_names = ["S10", "S11", "S12", "S13", "S14"]
        return feature_names

    def get_macro_feature_method_references(self):
        method_refs = [get_tree_heights, get_prop_graphs_node_counts, get_max_outdegrees, get_prop_graps_cascade_num,
                       get_max_out_degree_depths,
                       get_prop_graphs_num_of_cascades_with_retweets,
                       get_prop_graphs_fraction_of_cascades_with_retweets,
                       get_prop_graphs_num_bot_users_retweeting,
                       get_prop_graphs_fraction_of_bot_users_retweeting,
                       ]

        return method_refs

    def get_macro_feature_method_names(self):
        feature_names = ["Macro - Tree depth",
                         "Macro - No of nodes",
                         "Macro - Maximum out degree",
                         "Macro - No of cascades",
                         "Macro - Max out degree node's level",
                         "No. of cascades with retweets",
                         "Fraction of cascades with retweets",
                         "No. of bot users retweeting",
                         "Fraction of bot user retweeting"]

        return feature_names

    feature_names = []

    def get_macro_feature_short_names(self):
        feature_names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
        return feature_names

    def get_features_array(self, prop_graphs, micro_features, macro_features, news_source=None, label=None,
                           file_dir="data/features", use_cache=False):
        all_features = []

        file_name = self.get_dump_file_name(news_source, micro_features, macro_features, label, file_dir)
        data_file = Path(file_name)

        if use_cache and data_file.is_file():
            return pickle.load(open(file_name, "rb"))

        if micro_features:
            target_edge_type = REPLY_EDGE

            reply_function_references = self.get_micro_feature_method_references()
            for function_ref in reply_function_references:
                features = function_ref(prop_graphs, target_edge_type)
                all_features.append(features)

        if macro_features:
            target_edge_type = RETWEET_EDGE
            retweet_function_references = self.get_macro_feature_method_references()
            for function_ref in retweet_function_references:
                features = function_ref(prop_graphs, target_edge_type)
                all_features.append(features)

        feature_array = np.transpose(get_numpy_array(all_features))

        pickle.dump(feature_array, open(file_name, "wb"))

        return feature_array
