import json
import os

import networkx as nx
from networkx.readwrite import json_graph

from util import constants
from util.util import tweet_node


def construct_tweet_node_from_json(json_data):
    new_graph = json_graph.tree_graph(json_data)
    root_node = [node for node, in_degree in nx.DiGraph.in_degree(new_graph).items() if in_degree == 0][0]
    node_id_obj_dict = dict()
    dfs_node_construction_helper(root_node, new_graph, set(), node_id_obj_dict)
    return node_id_obj_dict[root_node]


def dfs_node_construction_helper(node_id, graph: nx.DiGraph, visited: set, node_id_obj_dict: dict):
    if node_id in visited:
        return None

    visited.add(node_id)

    tweet_node_obj = construct_tweet_node_from_nx_node(node_id, graph)

    node_id_obj_dict[node_id] = tweet_node_obj

    for neighbor_node_id in graph.successors(node_id):
        if neighbor_node_id not in visited:
            dfs_node_construction_helper(neighbor_node_id, graph, visited, node_id_obj_dict)
            add_node_object_edge(node_id, neighbor_node_id, node_id_obj_dict)


def add_node_object_edge(parent_node_id: int, child_node_id: int, node_id_obj_dict: dict):
    parent_node = node_id_obj_dict[parent_node_id]
    child_node = node_id_obj_dict[child_node_id]

    if child_node.node_type == constants.RETWEET_NODE:
        parent_node.add_retweet_child(child_node)
    elif child_node.node_type == constants.REPLY_NODE:
        parent_node.add_reply_child(child_node)
    else:
        # news node add both retweet and reply edge
        parent_node.add_retweet_child(child_node)
        parent_node.add_reply_child(child_node)


def construct_tweet_node_from_nx_node(node_id, graph: nx.DiGraph):
    return tweet_node(tweet_id=graph.node[node_id]['tweet_id'],
                      created_time=graph.node[node_id]['time'],
                      node_type=graph.node[node_id]['type'],
                      user_id=graph.node[node_id]['user'],
                      botometer_score=graph.node[node_id].get('bot_score', None),
                      sentiment=graph.node[node_id].get('sentiment', None))


def get_dataset_sample_ids(news_source, news_label, dataset_dir="data/sample_ids"):
    sample_list = []
    with open("{}/{}_{}_ids_list.txt".format(dataset_dir, news_source, news_label)) as file:
        for id in file:
            sample_list.append(id.strip())

    return sample_list


def load_from_nx_graphs(dataset_dir: str, news_source: str, news_label: str):
    tweet_node_objects = []

    news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)

    for sample_id in get_dataset_sample_ids(news_source, news_label, "data/sample_ids"):
        with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
            tweet_node_objects.append(construct_tweet_node_from_json(json.load(file)))

    return tweet_node_objects


def load_networkx_graphs(dataset_dir: str, news_source: str, news_label: str):
    news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)

    news_samples = []

    for news_file in os.listdir(news_dataset_dir):
        with open("{}/{}.json".format(news_dataset_dir, news_file)) as file:
            news_samples.append(json_graph.tree_graph(json.load(file)))

    return news_samples


def load_dataset(dataset_dir: str, news_source: str):
    fake_news_samples = load_networkx_graphs(dataset_dir, news_source, "fake")
    real_news_samples = load_networkx_graphs(dataset_dir, news_source, "real")

    return fake_news_samples, real_news_samples


if __name__ == '__main__':
    fake_samples, real_samples = load_dataset("data/nx_network_data", "politifact")
