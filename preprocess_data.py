import json
import numpy as np
import pickle

from pymongo import MongoClient

from util.util import tweet_node, twitter_datetime_str_to_object


def get_epoch_timestamp_from_retweet(retweet):
    return twitter_datetime_str_to_object(retweet["created_at"])


def sort_retweet_object_by_time(retweets: list):
    retweets.sort(key=get_epoch_timestamp_from_retweet)

    return retweets


def construct_retweet_graph(root_node, retweets, user_friends_dict):
    retweets = sort_retweet_object_by_time(retweets)

    tweet_id_user_dict = dict()
    user_id_tweet_id_dict = dict()

    tweet_id_node_obj_dict = dict()
    user_id_node_obj_dict = dict()

    users_so_far = set()

    for retweet in retweets:

        retweet_user_id = retweet["user"]["id"]
        retweet_tweet_id = retweet["id"]

        tweet_id_user_dict[retweet_tweet_id] = retweet_user_id
        user_id_tweet_id_dict[retweet_user_id] = retweet_tweet_id

        retweet_node = tweet_node(tweet_id=retweet_tweet_id, text=retweet["text"],
                                  created_time=twitter_datetime_str_to_object(retweet["created_at"]),
                                  user_name=retweet["user"]["screen_name"], user_id=retweet_user_id,
                                  news_id=root_node.news_id)

        tweet_id_node_obj_dict[retweet_tweet_id] = retweet_node
        user_id_node_obj_dict[retweet_user_id] = retweet_node

        retweet_user_friends = set()

        if retweet_user_id in user_friends_dict:
            retweet_user_friends = set(user_friends_dict[retweet_user_id])
        else:
            print("user id not found...")

        potential_path_users = users_so_far.intersection(retweet_user_friends)

        parent_node = None

        # User retweeted a tweet from his friends
        if len(potential_path_users) > 0:

            parent_user = list(potential_path_users)[0]

            parent_node = user_id_node_obj_dict[parent_user]
            pass
        else:  # user tweeted original tweet
            parent_node = root_node

        add_retweet_link(parent_node, retweet_node)
        users_so_far.add(retweet_user_id)

    return root_node


def add_retweet_link(parent_node, child_node):
    parent_node.add_retweet_child(child_node)
    child_node.set_parent_node(parent_node)


def has_retweet_or_replies(tweets):
    for tweet in tweets:
        if len(tweet["retweet"]) > 0 or len(tweet["reply"]) > 0:
            return True

    return False


def get_forest_from_tweets(news, user_id_friends_dict):
    news_id = news["id"]
    tweets = news["tweets"]

    if not has_retweet_or_replies(tweets):
        return None

    tweet_id_node_dict = dict()

    news_article_text_content = ""

    if "text" in news["text_content"]:
        news_article_text_content = news["text_content"]["text"]

    news_root_node = tweet_node(-1, news_article_text_content, None, None, None, news_id)

    for tweet in tweets:
        node = tweet_node(tweet["tweet_id"], tweet["text"], tweet["created_at"], tweet["user_name"], tweet["user_id"],
                          news_id)
        tweet_id_node_dict[tweet["tweet_id"]] = node
        add_retweet_link(news_root_node, node)
        construct_retweet_graph(node, tweet["retweet"], user_id_friends_dict)
        add_reply_nodes(node, tweet["reply"])

    return news_root_node


def add_reply_nodes(node: tweet_node, replies: list):
    # TODO: As of now only reply of replies are considered...info about retweet of replies are also there...can add
    # those info also to network
    for reply in replies:
        reply_node = tweet_node(reply["id"], reply["text"], int(reply["created_at"]), reply["username"], reply["user"],
                                node.news_id)

        node.add_reply_child(reply_node)
        reply_node.set_parent_node(node)

        if "engagement" in reply:
            add_reply_nodes(reply_node, reply["engagement"]["tweet_replies"])


# def add_retweet_nodes(node: tweet_node, retweets: list, tweet_id_node_dict: dict):
#     for retweet in retweets:
#         retweet_node = tweet_node(retweet["id"], retweet["text"], twitter_datetime_str_to_object(retweet["created_at"]),
#                                   retweet["user"]["screen_name"],
#                                   retweet["user"]["id"], node.news_id)
#
#         if retweet_node.tweet_id not in tweet_id_node_dict:
#             tweet_id_node_dict[retweet_node.tweet_id] = retweet_node
#         else:
#             print("retweet Node already found")
#
#

#
# def add_node_links(tweet_id_node_dict: dict, tweets: list):
#     for tweet in tweets:
#         root_node = tweet_id_node_dict[tweet["tweet_id"]]
#
#         add_retweet_edges(tweet["retweet"], tweet_id_node_dict, root_node)
#         add_reply_edges(tweet["reply"], tweet_id_node_dict)
#
#
# def add_retweet_edges(retweets: list, tweet_id_node_dict: dict, root_node: tweet_node):
#     for retweet in retweets:
#         retweet_node = tweet_id_node_dict[retweet["id"]]
#         root_node.add_retweet_child(retweet_node)
#         retweet_node.set_parent_node(root_node)
#
#
# def add_reply_edges(replies: list, tweet_id_node_dict: dict):
#     for reply in replies:
#         reply_node = tweet_id_node_dict[reply["id"]]
#         root_node = tweet_id_node_dict[reply["in_reply_to_status_id"]]
#         root_node.add_reply_child(reply_node)
#         reply_node.set_parent_node(root_node)


def get_user_friends_dict(user_friends_file):
    user_id_friends_dict = dict()

    with open(user_friends_file) as file:
        for line in file:
            json_obj = json.loads(line)
            user_id_friends_dict[json_obj["user_id"]] = json_obj["followees"]

    return user_id_friends_dict


def constuct_dataset_forests(dataset_file, user_friends_file):
    with open(dataset_file, "r") as file:
        dataset = json.load(file)

    user_id_friends_dict = get_user_friends_dict(user_friends_file)

    dataset = dataset["dataset"]
    # dataset = dataset[:10]

    news_graphs = []

    for news in dataset:
        graph = get_forest_from_tweets(news, user_id_friends_dict)

        if graph:
            news_graphs.append(graph)

    print(len(news_graphs))

    pickle.dump(news_graphs, open("data/saved/politifact_fake_news_prop_graphs.pkl", "wb"))

    return news_graphs


def find_tree_height(node, type="retweet"):
    if node is None:
        return 0

    max_child_height = 0

    children = []
    if type == "retweet":
        children = node.retweet_children
    elif type == "reply":
        children = node.reply_children
    else:
        children = node.children

    for child in node.retweet_children:
        max_child_height = max(max_child_height, find_tree_height(child, type))

    return max_child_height + 1


def load_prop_graph(news_source, news_label):
    news_graphs = pickle.load(open("data/saved/{}_{}_news_prop_graphs.pkl".format(news_source, news_label), "rb"))
    return news_graphs


def analyze_height(news_graphs, type):
    heights = []

    for news_node in news_graphs:
        heights.append(find_tree_height(news_node, type))

    print("max", max(heights))
    print("min", min(heights))
    print("avg", np.mean(heights))


if __name__ == "__main__":
    politifact_fake_dataset_file = "data/politifact_fake_news_dataset.json"
    politifact_real_dataset_file = "/Users/deepak/Desktop/DMML/GitRepo/FakeNewsPropagation/data/politifact_real_news_dataset.json"

    politifact_fake_user_friends_file = "data/politifact_fake_user_friends_ids_complete.txt"

    constuct_dataset_forests(politifact_fake_dataset_file, politifact_fake_user_friends_file)

    news_graphs = load_prop_graph("politifact", "fake")

    analyze_height(news_graphs, "retweet")
    analyze_height(news_graphs)
