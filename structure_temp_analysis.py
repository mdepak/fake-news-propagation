import time

import numpy as np

from analysis_util import get_propagation_graphs, equal_samples, get_numpy_array, BaseFeatureHelper
from stat_test import perform_t_test, plot_normal_distributions, get_box_plots
from util.constants import NEWS_ROOT_NODE, RETWEET_EDGE, REPLY_EDGE
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


def get_all_structural_features(news_graphs, micro_features, macro_features):
    all_features = []
    target_edge_type = RETWEET_EDGE

    if macro_features:
        retweet_function_references = [get_tree_heights, get_prop_graphs_node_counts, get_prop_graps_cascade_num,
                                       get_max_outdegrees]
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

    def get_micro_feature_method_references(self):
        method_refs = [get_tree_heights, get_prop_graphs_node_counts, get_max_outdegrees]
        return method_refs

    def get_micro_feature_method_names(self):
        feature_names = ["Micro - Tree depth", "Micro - No of nodes", "Micro - Maximum out degree"]
        return feature_names

    def get_micro_feature_short_names(self):
        feature_names = ["S7", "S8", "S9"]
        return feature_names

    def get_macro_feature_method_references(self):
        method_refs = [get_tree_heights, get_prop_graphs_node_counts, get_max_outdegrees, get_prop_graps_cascade_num,
                       get_max_out_degree_depths]
        return method_refs

    def get_macro_feature_method_names(self):
        feature_names = ["Macro - Tree depth", "Macro - No of nodes", "Macro - Maximum out degree",
                         "Macro - No of cascades", "Macro - Max out degree node's level"]
        return feature_names

    feature_names = []

    def get_macro_feature_short_names(self):
        feature_names = ["S1", "S2", "S3", "S4", "S5"]
        return feature_names

    def get_features_array(self, prop_graphs, micro_features, macro_features, news_source=None, label=None):
        all_features = []

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

        return np.transpose(get_numpy_array(all_features))


if __name__ == "__main__":
    fake_prop_graph, real_prop_graph = get_propagation_graphs("data/saved_new_no_filter", "politifact")

    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    structure_feature_helper = StructureFeatureHelper()
    fake_features = structure_feature_helper.get_features_array(fake_prop_graph, micro_features=True,
                                                                macro_features=True)
    real_features = structure_feature_helper.get_features_array(real_prop_graph, micro_features=True,
                                                                macro_features=True)

    structure_feature_helper.save_blox_plots_for_features(fake_feature_array=fake_features,
                                                          real_feature_array=real_features, micro_features=True,
                                                          macro_features=True, save_folder="data/feature_images")

    # Print the statistics of the dataset
    print("------------Fake------------")
    structure_feature_helper.print_statistics_for_all_features(feature_array=fake_features, prop_graphs=fake_prop_graph,
                                                               micro_features=True, macro_features=True)

    print("------------Real------------")
    structure_feature_helper.print_statistics_for_all_features(feature_array=real_features, prop_graphs=fake_prop_graph,
                                                               micro_features=True, macro_features=True)

    target_edge_type = RETWEET_EDGE

    # fake_prop_features = get_tree_heights(fake_prop_graph, target_edge_type)
    # real_prop_features = get_tree_heights(real_prop_graph, target_edge_type)
    #

    fake_prop_features = get_prop_graphs_node_counts(fake_prop_graph, target_edge_type)
    real_prop_features = get_prop_graphs_node_counts(real_prop_graph, target_edge_type)

    # fake_prop_features = get_node_count_deepest_cascade(fake_prop_graph, target_edge_type)
    # real_prop_features = get_node_count_deepest_cascade(real_prop_graph, target_edge_type)

    # fake_prop_features = get_prop_graps_cascade_num(fake_prop_graph, target_edge_type)
    # real_prop_features = get_prop_graps_cascade_num(real_prop_graph, target_edge_type)

    perform_t_test(fake_prop_features, real_prop_features)

    # get_box_plots(fake_prop_features, real_prop_features, "Deepest cascade node height")

    # plot_normal_distributions(fake_prop_features, real_prop_features)

    # analyze_height(propagation_graphs, target_edge_type)

    # analyze_height_by_time(propagation_graphs, target_edge_type, [60, 300, 600, 900, 18000, 36000, 72000])

    # analyze_node_count(propagation_graphs, target_edge_type)
    # analyze_node_size_by_time(propagation_graphs, target_edge_type, [60, 300, 600, 900, 18000, 36000, 72000])

    # analyze_max_outdegree(propagation_graphs, target_edge_type)

    # analyze_cascade(propagation_graphs, target_edge_type)
    # analyze_cascade_num_by_time(propagation_graphs, target_edge_type, [60, 300, 600, 900, 18000, 36000, 72000] )

    # exit(1)
