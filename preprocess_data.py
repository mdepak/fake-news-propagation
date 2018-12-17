import configparser
import json
import mmap
import os
import re

import numpy as np
import pickle

from tqdm import tqdm
from pymongo import MongoClient

from analysis_util import sort_retweet_object_by_time
from misc_process import get_politifact_tweet_filter_dates
from pre_process_util import load_configuration, get_database_connection
from util.constants import RETWEET_NODE, NEWS_ROOT_NODE, POST_NODE, REPLY_NODE
from util.graph_dumper import dumps_graph
from util.util import tweet_node, twitter_datetime_str_to_object


def is_user_followees_info_file_present(folder, user_id):
    file_path = "{}/{}.json".format(folder, str(user_id))
    return os.path.exists(file_path)


def get_user_id_followees(folder, user_id):
    file_path = "{}/{}.json".format(folder, user_id)
    try:
        return set(json.load(open(file_path))["followees"])
    except:
        return set()


def get_user_name_followees(folder, user_name):
    file_path = "{}/{}.json".format(folder, user_name)
    try:
        return set(json.load(open(file_path))["friends_name"])
    except:
        return set()


def construct_retweet_graph(root_node, retweets, user_id_friends_dict, user_name_friends_dict):
    retweets = sort_retweet_object_by_time(retweets)

    tweet_id_user_dict = dict()
    user_id_tweet_id_dict = dict()

    tweet_id_node_obj_dict = dict()
    user_id_node_obj_dict = dict()
    user_name_node_obj_dict = dict()

    user_ids_so_far = set()
    user_names_so_far = set()

    user_ids_folder = "/Users/deepak/Desktop/social_network_single_files/user_ids_files"
    user_name_folder = "/Users/deepak/Desktop/social_network_single_files/user_names_files"

    for retweet in retweets:

        retweet_user_id = retweet["user"]["id"]
        retweet_tweet_id = retweet["id"]
        retweet_user_name = retweet["user"]["screen_name"]

        tweet_id_user_dict[retweet_tweet_id] = retweet_user_id
        user_id_tweet_id_dict[retweet_user_id] = retweet_tweet_id

        retweet_node = tweet_node(tweet_id=retweet_tweet_id, text=retweet["text"],
                                  created_time=twitter_datetime_str_to_object(retweet["created_at"]),
                                  user_name=retweet["user"]["screen_name"], user_id=retweet_user_id,
                                  news_id=root_node.news_id, node_type=RETWEET_NODE)

        tweet_id_node_obj_dict[retweet_tweet_id] = retweet_node
        user_id_node_obj_dict[retweet_user_id] = retweet_node
        user_name_node_obj_dict[retweet_user_name] = retweet_node

        parent_node = root_node  # user tweeted original tweet or the path info or network data not available

        # if retweet_user_id in user_id_friends_dict:
        if is_user_followees_info_file_present(user_ids_folder, retweet_user_id):
            # retweet_user_friends = user_id_friends_dict[retweet_user_id]
            retweet_user_friends = get_user_id_followees(user_ids_folder, retweet_user_id)
            potential_path_users = user_ids_so_far.intersection(retweet_user_friends)

            # User retweeted a tweet from his friends
            if len(potential_path_users) > 0:
                parent_user = list(potential_path_users)[0]

                parent_node = user_id_node_obj_dict[parent_user]

        # elif retweet_user_name in user_name_friends_dict:
        elif is_user_followees_info_file_present(user_name_folder, retweet_user_name):
            # retweet_user_friends = user_name_friends_dict[retweet_user_name]
            retweet_user_friends = get_user_name_followees(user_name_folder, retweet_user_name)
            potential_path_users = user_names_so_far.intersection(retweet_user_friends)

            # User retweeted a tweet from his friends
            if len(potential_path_users) > 0:
                parent_user = list(potential_path_users)[0]

                parent_node = user_name_node_obj_dict[parent_user]

        else:
            print("user id : {} or user name : {} not found...".format(retweet_user_id, retweet_user_name), flush=True)

        add_retweet_link(parent_node, retweet_node)
        user_ids_so_far.add(retweet_user_id)
        user_names_so_far.add(retweet_user_name)

    return root_node


def add_retweet_link(parent_node, child_node):
    parent_node.add_retweet_child(child_node)
    child_node.set_parent_node(parent_node)


def add_reply_link(parent_node: tweet_node, child_node: tweet_node):
    parent_node.add_reply_child(child_node)
    child_node.set_parent_node(parent_node)


def has_retweet_or_replies(tweets, tweet_filter_date):
    for tweet in tweets:
        if tweet["created_at"] >= int(tweet_filter_date):
            if len(tweet["retweet"]) > 0 or len(tweet["reply"]) > 0:
                return True

    return False


def get_forest_from_tweets(news, user_id_friends_dict, user_name_friends_dict, news_id_tweet_filter_date_dict):
    news_id = news["id"]
    tweets = news["tweets"]
    tweet_filter_date = (news_id_tweet_filter_date_dict[news_id] - (3600 * 24 * 30))

    if not has_retweet_or_replies(tweets, tweet_filter_date):
        return None

    tweet_id_node_dict = dict()

    news_article_text_content = ""

    if "text" in news["text_content"]:
        news_article_text_content = news["text_content"]["text"]

    news_publication_time = ""

    news_root_node = tweet_node(news_id, news_article_text_content, None, None, None, news_id, node_type=NEWS_ROOT_NODE)

    for tweet in tweets:
        if tweet["created_at"] >= tweet_filter_date:
            node = tweet_node(tweet["tweet_id"], tweet["text"], tweet["created_at"], tweet["user_name"], tweet["user_id"],
                              news_id, node_type=POST_NODE)

            tweet_id_node_dict[tweet["tweet_id"]] = node

            add_retweet_link(news_root_node, node)
            add_reply_link(news_root_node, node)

            construct_retweet_graph(node, tweet["retweet"], user_id_friends_dict, user_name_friends_dict)
            add_reply_nodes(node, tweet["reply"])

    return news_root_node


def get_reply_of_replies(replies: list):
    replies_list = []

    for reply in replies:
        if reply:

            if "engagement" in reply:
                replies_list.extend(get_reply_of_replies(reply["engagement"]["tweet_replies"]))

            replies_list.append(reply)

    return replies_list


def add_reply_nodes(node: tweet_node, replies: list):
    # TODO: As of now only reply of replies are considered...info about retweet of replies are also there...can add
    # those info also to network
    for reply in replies:
        if reply:
            if "username" in reply:
                user_name = reply["username"]
            else:
                user_name = reply["user_name"]

            if "user_id" in reply:
                user_id = reply["user_id"]
            else:
                user_id = reply["user"]

            reply_node = tweet_node(reply["id"], reply["text"], int(reply["created_at"]), user_name, user_id,
                                    node.news_id,
                                    node_type=REPLY_NODE)

            node.add_reply_child(reply_node)
            reply_node.set_parent_node(node)

            if "engagement" in reply:
                add_reply_nodes(reply, reply["engagement"]["tweet_replies"])

        else:
            print("---------REPLY NOT FOUND----------------")


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


def get_user_friends_dict(user_friends_file, id_reference_file):
    user_id_friends_dict = dict()
    id_reference_dict = json.load(open(id_reference_file))

    with open(user_friends_file) as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                if str(json_obj["user_id"]) in id_reference_dict:
                    user_id_friends_dict[json_obj["user_id"]] = json_obj["followees"]
            except:
                print("Exception loading data")

    return user_id_friends_dict


def get_user_name_friends_dict(user_name_friends_file, user_names_refernce_file):
    user_name_friends_dict = dict()
    user_name_reference_set = set(json.load(open(user_names_refernce_file))["user_names"])

    with open(user_name_friends_file) as file:
        for line in file:
            json_obj = json.loads(line)
            if json_obj["user_name"] in user_name_reference_set:
                user_name_friends_dict[json_obj["user_name"]] = set(json_obj["friends_name"])

    return user_name_friends_dict


def get_news_articles(data_file):
    news_articles = []

    with open(data_file) as file:
        for line in file:
            news_articles.append(json.loads(line))

    return news_articles


def constuct_dataset_forests(enagement_file_dir, social_network_dir, out_dir, news_source, label, db, is_fake):
    dataset_file = "{}/{}_{}_news_dataset_format.json".format(enagement_file_dir, news_source, label)
    user_ids_friends_file = "{}/{}_user_ids_friends_network.txt".format(social_network_dir, news_source)
    user_name_friends_file = "{}/{}_user_names_friends_network.txt".format(social_network_dir, news_source)
    id_reference_file = "data/format/{}_{}_user_id_user_name_dict.json".format(news_source, label)
    user_names_refernce_file = "data/format/{}_{}_prop_user_names.json".format(news_source, label)

    out_file = "{}/{}_{}_news_prop_graphs.pkl".format(out_dir, news_source, label)

    # print(dataset_file)
    # file = open(dataset_file, "r")
    # file_contents = file.readlines()
    # dataset = json.loads("\n".join(file_contents))
    # file.close()
    # dataset = json.load(dataset_file)
    # dataset = dataset["dataset"]

    dataset = get_news_articles(dataset_file)

    # user_id_friends_dict = get_user_friends_dict(user_ids_friends_file, id_reference_file)
    # user_name_friends_dict = get_user_name_friends_dict(user_name_friends_file, user_names_refernce_file)

    user_id_friends_dict = {}
    user_name_friends_dict = {}

    news_id_tweet_filter_date_dict = get_politifact_tweet_filter_dates(db, is_fake)

    print("Construction of forest : {} - {}".format(news_source, label))
    print("No of user name - friends : {}".format(len(user_name_friends_dict)))
    print("No. of user id - friends : {}".format(len(user_id_friends_dict)), flush=True)

    input("Press Enter to continue...")

    news_graphs = []

    for news in tqdm(dataset):
        graph = get_forest_from_tweets(news, user_id_friends_dict, user_name_friends_dict,
                                       news_id_tweet_filter_date_dict)

        if graph:
            news_graphs.append(graph)
            print("Added graph for news id : {}".format(graph.tweet_id), flush=True)

    print(len(news_graphs))

    pickle.dump(news_graphs, open(out_file, "wb"))

    return news_graphs


def dump_graphs(graphs):
    params = {"node_color": {}}
    params["node_color"][NEWS_ROOT_NODE] = "#77ab59"
    params["node_color"][POST_NODE] = "#d2b4d7"
    params["node_color"][RETWEET_NODE] = "#87cefa"
    params["node_color"][REPLY_NODE] = "#735144"

    tweet_info = dict()
    news_graphs = []

    for news in graphs:
        [tweet_info_object_dict, nodes_list, edges_list] = dumps_graph(news, params)
        graph = {"nodes": nodes_list, "edges": edges_list}
        news = {"news_id": news.news_id, "graph": graph}

        tweet_info.update(tweet_info_object_dict)
        news_graphs.append(news)

    return [news_graphs, tweet_info]





# def get_networ

def write_graph_data_to_db(db, news_graphs, tweet_info):
    for news in news_graphs:
        db.news_prop_graphs.update({"news_id": news["news_id"]}, {"$set": news}, upsert=True)

    # for tweet_id, tweet_info in tweet_info.items():
    #     db.propagation_tweet_info.update({"tweet_id": tweet_id}, {"$set": tweet_info}, upsert=True)


def dump_files_as_lines(dataset_file, out_file):
    dataset = json.load(open(dataset_file))
    with open(out_file, "w") as file:
        for news in dataset["dataset"]:
            file.write(json.dumps(news) + "\n")

    print("Dumped file : {}".format(out_file), flush=True)


if __name__ == "__main__":
    politifact_fake_dataset_file = "data/politifact_fake_news_dataset.json"
    politifact_real_dataset_file = "data/politifact_real_news_dataset.json"

    politifact_fake_user_friends_file = "data/politifact_fake_user_friends_ids_complete.txt"

    # dump_files_as_lines(politifact_real_dataset_file, "data/politfact_real_news_dataset_format.json")
    # dump_files_as_lines(politifact_fake_dataset_file, "data/politfact_fake_news_dataset_format.json")

    config = load_configuration("project.config")
    db = get_database_connection(config)

    constuct_dataset_forests("data/engagement_data", "data/social_network_data", "data/saved_new", "politifact", "fake",
                             db, is_fake=True)

    # constuct_dataset_forests("data/engagement_data", "data/social_network_data", "data/saved", "politifact", "real")

    # constuct_dataset_forests("data/engagement_data", "data/social_network_data", "data/saved", "gossipcop", "fake")
    # constuct_dataset_forests("data/engagement_data", "data/social_network_data", "data/saved", "gossipcop", "real")

    # fake_news_graphs = load_prop_graph("politifact", "fake")
    # real_news_graphs = load_prop_graph("politifact", "real")
    #
    # analyze_height(fake_news_graphs)
    # analyze_height(real_news_graphs)

    # [fake_news_graphs, fake_tweet_info] = dump_graphs(fake_news_graphs)
    #
    # [real_news_graphs, real_tweet_info] = dump_graphs(real_news_graphs)
    # json.dump(news_graphs, open("data/saved/news_graphs.json", "w"))
    # json.dump(tweet_info, open("data/saved/tweet_info.json", "w"))

    # config = load_configuration("project.config")
    # db = get_database_connection(config)

    # write_graph_data_to_db(db, fake_news_graphs, real_tweet_info)

    # analyze_height(news_graphs, "retweet")
    # analyze_height(news_graphs)
