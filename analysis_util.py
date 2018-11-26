import numpy as np
import pickle

from util.util import twitter_datetime_str_to_object, tweet_node


def get_epoch_timestamp_from_retweet(retweet):
    return twitter_datetime_str_to_object(retweet["created_at"])


def sort_retweet_object_by_time(retweets: list):
    retweets.sort(key=get_epoch_timestamp_from_retweet)

    return retweets


def get_noise_news_ids():
    with open("data/news_id_ignore_list") as file:
        lines = file.readlines()
        return [line.strip() for line in lines]


def load_prop_graph(news_source, news_label):
    news_graphs = pickle.load(open("data/saved/{}_{}_news_prop_graphs.pkl".format(news_source, news_label), "rb"))
    return news_graphs


def remove_prop_graph_noise(news_graphs, noise_ids):
    noise_ids = set(noise_ids)
    return [graph for graph in news_graphs if graph.tweet_id not in noise_ids]


def sort_tweet_node_object_by_created_time(tweet_nodes: list):
    tweet_nodes.sort(key=lambda x: x.created_time)

    return tweet_nodes


def equal_samples(sample1, sample2):
    target_len = min(len(sample1), len(sample2))

    np.random.seed(0)

    np.random.shuffle(sample1)
    np.random.shuffle(sample2)

    return sample1[:target_len], sample2[:target_len]


def get_propagation_graphs(news_source):
    fake_propagation_graphs = load_prop_graph(news_source, "fake")
    real_propagation_graphs = load_prop_graph(news_source, "real")

    print("Before filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
    print("Before filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))

    fake_propagation_graphs = remove_prop_graph_noise(fake_propagation_graphs, get_noise_news_ids())
    real_propagation_graphs = remove_prop_graph_noise(real_propagation_graphs, get_noise_news_ids())

    print("After filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
    print("After filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))
    print(flush=True)

    return fake_propagation_graphs, real_propagation_graphs


def get_numpy_array(list_of_list):
    np_array_lists = []
    for list_obj in list_of_list:
        np_array_lists.append(np.array(list_obj))

    return np.array(np_array_lists)
