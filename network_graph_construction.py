import networkx as nx

from util.util import tweet_node


def construct_networkxx_graph(graph: tweet_node, network_type):
    G = nx.DiGraph()

    G.add_node(graph.tweet_id)

    if network_type == "retweet":
        for node in graph.retweet_children:
            add_networkxx_retweet_data(node)
            G.add_edge(node.tweet_id, graph.tweet_id)
    else:
        for node in graph.reply_children:
            add_network_reply_data(node)
            G.add_edge(node.tweet_id, graph.tweet_id)


def add_networkxx_retweet_data(nx_graph: nx.DiGraph, node: tweet_node):
    nx_graph.add_node(node.tweet_id)

    for child in node.retweet_children:
        add_networkxx_retweet_data(nx_graph, child)
        nx_graph.add_edge(node.tweet_id, child.tweet_id)


def add_network_reply_data(nx_graph: nx.DiGraph, node: tweet_node):
    nx_graph.add_node(node.tweet_id)

    for child in node.reply_children:
        add_network_reply_data(nx_graph, child)
        nx_graph.add_edge(node.tweet_id, child.tweet_id)
