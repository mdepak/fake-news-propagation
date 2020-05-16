# from util.util import tweet_node
#
#
# def dumps_graph(root_node: tweet_node, params):
#     tweet_info_object_dict = dict()
#     edges_list = []
#     nodes_list = []
#
#     tweet_id_node_id_dict = dict()
#
#     add_tweet_node_if_not_exists(tweet_id_node_id_dict, root_node, nodes_list, tweet_info_object_dict, params)
#
#     root_node_id = tweet_id_node_id_dict[root_node.tweet_id]
#
#     for child in root_node.children:
#         child_node_id = add_tweet_node_if_not_exists(tweet_id_node_id_dict, child, nodes_list, tweet_info_object_dict,
#                                                      params)
#
#         edges_list.append(get_edge(root_node_id, child_node_id))
#
#         dump_children_network(child, nodes_list, edges_list, tweet_id_node_id_dict, tweet_info_object_dict, params)
#
#     legend_node_id = len(tweet_id_node_id_dict)+1
#     return [tweet_info_object_dict, nodes_list, edges_list]
#
#
# def get_edge(parent_node_id, child_node_id):
#     return {"from": parent_node_id, "to": child_node_id}
#
#
# def add_tweet_node_if_not_exists(tweet_id_node_id_dict, node: tweet_node, nodes_list, tweet_info_object_dict: dict,
#                                  params):
#     if node.tweet_id not in tweet_id_node_id_dict:
#         tweet_id_node_id_dict[node.tweet_id] = len(tweet_id_node_id_dict) + 1
#
#         nodes_list.append({"id": tweet_id_node_id_dict[node.tweet_id], "tweet_id": str(node.tweet_id),
#                            "label": tweet_id_node_id_dict[node.tweet_id],
#                            "color": params["node_color"][node.node_type]})
#
#         tweet_info_object_dict[str(node.tweet_id)] = node.get_contents()
#
#     return tweet_id_node_id_dict[node.tweet_id]
#
#
# def dump_children_network(node, nodes_list: list, edge_list: list, tweet_id_node_id_dict: dict,
#                           tweet_info_object_dict: dict, params):
#     node_id = add_tweet_node_if_not_exists(tweet_id_node_id_dict, node, nodes_list, tweet_info_object_dict, params)
#
#     for child in node.children:
#         dump_children_network(child, nodes_list, edge_list, tweet_id_node_id_dict, tweet_info_object_dict, params)
#         child_id = tweet_id_node_id_dict[child.tweet_id]
#
#         edge_list.append(get_edge(node_id, child_id))
#
# # def dump_reply_network(node: tweet_node, nodes_list: list, edge_list: list, tweet_info_object_dict: dict):
# #     node_id = add_tweet_node_if_not_exists(tweet_id_node_id_dict, node, nodes_list, params)
# #
# #     for child in node.reply_children:
# #         dump_retweet_network(child, nodes_list, edge_list, tweet_id_node_id_dict, params)
# #         child_id = tweet_id_node_id_dict[child.tweet_id]
# #
# #         edge_list.append(get_edge(node_id, child_id))
