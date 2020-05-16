import errno
import os
import pickle
import queue
import time
from math import ceil
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from twitter_tokenize import twitter_tokenize
from util.util import tweet_node


def construct_networkx_graph(graph: tweet_node, network_type):
    G = nx.DiGraph()

    tweet_id_node_id_dict = dict()

    G.add_node(get_tweet_id_node_id_mapping(graph.tweet_id, tweet_id_node_id_dict))

    if network_type == "retweet":
        for node in graph.retweet_children:
            add_networkxx_retweet_data(G, node, tweet_id_node_id_dict)
            G.add_edge(get_tweet_id_node_id_mapping(graph.tweet_id, tweet_id_node_id_dict),
                       get_tweet_id_node_id_mapping(node.tweet_id, tweet_id_node_id_dict))
    else:
        for node in graph.reply_children:
            add_network_reply_data(G, node, tweet_id_node_id_dict)
            G.add_edge(get_tweet_id_node_id_mapping(graph.tweet_id, tweet_id_node_id_dict),
                       get_tweet_id_node_id_mapping(node.tweet_id, tweet_id_node_id_dict))

    return G, tweet_id_node_id_dict


def get_tweet_id_node_id_mapping(tweet_id, tweet_id_node_id_dict):
    if tweet_id not in tweet_id_node_id_dict:
        tweet_id_node_id_dict[tweet_id] = len(tweet_id_node_id_dict)

    return tweet_id_node_id_dict[tweet_id]


def add_networkxx_retweet_data(nx_graph: nx.DiGraph, node: tweet_node, tweet_id_node_id_dict: dict):
    nx_graph.add_node(get_tweet_id_node_id_mapping(node.tweet_id, tweet_id_node_id_dict))

    for child in node.retweet_children:
        add_networkxx_retweet_data(nx_graph, child, tweet_id_node_id_dict)
        nx_graph.add_edge(get_tweet_id_node_id_mapping(node.tweet_id, tweet_id_node_id_dict),
                          get_tweet_id_node_id_mapping(child.tweet_id, tweet_id_node_id_dict))


def add_network_reply_data(nx_graph: nx.DiGraph, node: tweet_node, tweet_id_node_id_dict: dict):
    nx_graph.add_node(node.tweet_id)

    for child in node.reply_children:
        add_network_reply_data(nx_graph, child, tweet_id_node_id_dict)
        nx_graph.add_edge(get_tweet_id_node_id_mapping(node.tweet_id, tweet_id_node_id_dict),
                          get_tweet_id_node_id_mapping(child.tweet_id, tweet_id_node_id_dict))


def get_noise_news_ids():
    with open("data/news_id_ignore_list") as file:
        lines = file.readlines()
        return [line.strip() for line in lines]


def get_propagation_graphs(data_folder, news_source):
    fake_propagation_graphs = load_prop_graph(data_folder, news_source, "fake")
    # fake_propagation_graphs = []
    real_propagation_graphs = load_prop_graph(data_folder, news_source, "real")

    print("Before filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
    print("Before filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))

    fake_propagation_graphs = remove_prop_graph_noise(fake_propagation_graphs, get_noise_news_ids())
    real_propagation_graphs = remove_prop_graph_noise(real_propagation_graphs, get_noise_news_ids())

    print("After filtering no. of FAKE prop graphs: {}".format(len(fake_propagation_graphs)))
    print("After filtering no. of REAL prop graphs: {}".format(len(real_propagation_graphs)))
    print(flush=True)

    return fake_propagation_graphs, real_propagation_graphs


def load_prop_graph(data_folder, news_source, news_label):
    news_graphs = pickle.load(open("{}/{}_{}_news_prop_graphs.pkl".format(data_folder, news_source, news_label), "rb"))
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


def filter_propagation_graphs(graphs, limit_time, retweet=True, reply=True):
    result_graphs = []

    for prop_graph in graphs:
        filtered_prop_graph = remove_node_by_time(prop_graph, limit_time)
        if is_valid_graph(filtered_prop_graph, retweet, reply):
            result_graphs.append(filtered_prop_graph)

    return result_graphs


def is_valid_graph(prop_graph: tweet_node, retweet=True, reply=True):
    """ Check if the prop graph has alteast one retweet or reply"""

    for post_node in prop_graph.children:
        if (retweet and len(post_node.reply_children) > 0) or (reply and len(post_node.retweet_children) > 0):
            return True

    return False


def get_first_post_time(node: tweet_node):
    first_post_time = time.time()

    for child in node.children:
        first_post_time = min(first_post_time, child.created_time)

    return first_post_time


def remove_node_by_time(graph: tweet_node, limit_time):
    start_time = get_first_post_time(graph)
    end_time = start_time + limit_time

    q = queue.Queue()

    q.put(graph)

    while q.qsize() != 0:
        node = q.get()

        children = node.children

        retweet_children = set(node.retweet_children)
        reply_children = set(node.reply_children)

        for child in children.copy():

            if child.created_time <= end_time:
                q.put(child)
            else:
                node.children.remove(child)
                try:
                    retweet_children.remove(child)
                except KeyError:  # Element not found in the list
                    pass
                try:
                    reply_children.remove(child)
                except KeyError:  # Element not found in the list
                    pass

        node.retweet_children = list(retweet_children)
        node.reply_children = list(reply_children)

    return graph


def get_all_propagation_graphs(news_source="politifact", time_interval=None, args=None):
    if Path.is_file(Path("data/{}_graphs_data.pkl".format(news_source))):
        graph_data = pickle.load(open("data/{}_graphs_data.pkl".format(news_source), "rb"))
        return graph_data

    fake_prop_graph, real_prop_graph = get_propagation_graphs("data/prop_graph_save", news_source)

    # fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    # fake_prop_graph = fake_prop_graph[:100]
    # real_prop_graph = real_prop_graph[:100]

    if time_interval is not None:
        time_limit = time_interval * 60 * 60

        print("Time limit in seconds : {}".format(time_limit))

        fake_prop_graph = filter_propagation_graphs(fake_prop_graph, time_limit, reply=False)
        real_prop_graph = filter_propagation_graphs(real_prop_graph, time_limit, reply=False)

        print("After time based filtering ")
        print("No. of fake samples : {}  No. of real samples: {}".format(len(fake_prop_graph), len(real_prop_graph)))

        fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    all_network_xx_graphs = []
    all_tweet_id_node_ids_dicts = []
    all_tweet_id_text_dict = dict()
    one_hot_labels = []

    labels = []

    max_num_nodes = 0

    graph_hidden_states = []

    news_article_text_contents = []

    for graph in fake_prop_graph:
        get_textual_features(graph, all_tweet_id_text_dict)
        news_article_text_contents.append(graph.text)
        # TODO: Uncomment after dumping time series data - prune graphs for network generation
        # graph = prune_graph_by_max_nodes_time(graph, args.max_num_node)
        graph, sample_tweet_id_node_id_dict = construct_networkx_graph(graph, "retweet")
        all_network_xx_graphs.append(graph)
        max_num_nodes = max(max_num_nodes, nx.number_of_nodes(graph))
        all_tweet_id_node_ids_dicts.append(sample_tweet_id_node_id_dict)
        one_hot_labels.append([1, 0])
        labels.append(1)

    for graph in real_prop_graph:
        get_textual_features(graph, all_tweet_id_text_dict)
        news_article_text_contents.append(graph.text)
        # TODO: Uncomment after dumping time series data - prune graphs for network generation
        # graph = prune_graph_by_max_nodes_time(graph, args.max_num_node)
        graph, sample_tweet_id_node_id_dict = construct_networkx_graph(graph, "retweeet")
        all_network_xx_graphs.append(graph)
        max_num_nodes = max(max_num_nodes, nx.number_of_nodes(graph))
        all_tweet_id_node_ids_dicts.append(sample_tweet_id_node_id_dict)
        one_hot_labels.append([0, 1])
        labels.append(0)

    print("max number of nodes : {}".format(max_num_nodes))

    # TODO: Construct hidden state of the network using Glove embedding

    # model_path = "/home/dmahudes/temporal_event_analysis/pre_train_model/glove.twitter.27B.200d.w2vformat.txt"

    model_path = "data/glove.twitter.27B.200d.w2vformat.txt"

    glove_model = get_gensim_model(model_path)

    for news_article in news_article_text_contents:
        news_feature = get_tweet_latent_embeddings(news_article, glove_model)
        news_feature = np.expand_dims(np.array(news_feature), axis=1).transpose()
        graph_hidden_states.append(news_feature)

    # return all_network_xx_graphs, all_tweet_id_text_dict, one_hot_labels, labels, all_tweet_id_node_ids_dicts
    # return all_network_xx_graphs, all_tweet_id_text_dict, one_hot_labels, labels, all_tweet_id_node_ids_dicts, \
    #        np.concatenate(graph_hidden_states)

    graph_data = [all_network_xx_graphs, all_tweet_id_text_dict, one_hot_labels, labels, all_tweet_id_node_ids_dicts,
                  np.concatenate(graph_hidden_states)]

    pickle.dump(graph_data, open("data/{}_graphs_data.pkl".format(news_source), "wb"))

    return graph_data


def get_textual_features(graph: tweet_node, tweet_id_text_dict):
    q = queue.Queue()

    q.put(graph)

    while q.qsize() != 0:
        node = q.get()
        tweet_id_text_dict[node.tweet_id] = node.text
        for child in node.retweet_children:
            q.put(child)


def nodes_stats(all_network_xx_graphs):
    node_sizes = []

    for graph in all_network_xx_graphs:
        node_sizes.append(nx.number_of_nodes(graph))

    print("Min : {}".format(min(node_sizes)))
    print("Max : {}".format(max(node_sizes)))
    print("Mean : {}".format(np.mean(node_sizes)))
    print("STD: {} ".format(np.std(node_sizes)))
    print("Total nodes : {}".format(np.sum(node_sizes)))


def filter_graphs(all_network_xx_graphs, max_nodes):
    graphs = []
    for graph in all_network_xx_graphs:
        nodes_count = nx.number_of_nodes(graph)
        if nodes_count <= max_nodes:
            graphs.append(graph)

    return graphs


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


def get_K_node_time(graph, max_nodes):
    node_creation_times = []

    q = queue.Queue()

    q.put(graph)

    while q.qsize() != 0:
        node = q.get()

        children = node.retweet_children

        for child in children:
            q.put(child)
            node_creation_times.append(child.created_time)

    node_creation_times.sort()

    return node_creation_times[max_nodes - 1]


def prune_graph_by_max_nodes_time(graph, max_nodes):
    if get_nodes_count(graph) < max_nodes:
        return graph

    node_k_time = get_K_node_time(graph, max_nodes)

    return remove_node_by_end_time(graph, node_k_time)


def remove_node_by_end_time(graph: tweet_node, end_time):
    q = queue.Queue()

    q.put(graph)

    while q.qsize() != 0:
        node = q.get()

        children = node.children

        for child in list(children):

            if child.created_time <= end_time:
                q.put(child)
            else:
                node.children.remove(child)
                try:
                    node.retweet_children.remove(child)
                except ValueError:  # Element not found in the list
                    pass
                try:
                    node.reply_children.remove(child)
                except ValueError:  # Element not found in the list
                    pass

    return graph


def reverse_dict(tweet_id_node_id_dict):
    node_id_tweet_id_dict = dict()

    for key, value in tweet_id_node_id_dict.items():
        node_id_tweet_id_dict[value] = key

    return node_id_tweet_id_dict


def get_batch_pooling_matrix(graphs):
    node_sizes = []

    for graph in graphs:
        nx.nodes(graph)
        node_sizes.append(nx.number_of_nodes(graph))

    num_graphs = len(graphs)
    num_nodes = np.sum(node_sizes)

    pooling_matrix = np.zeros((num_graphs, num_nodes))

    start = 0

    indexes = []

    for idx, graph in enumerate(graphs):
        indexes.append(start)

        start += len(nx.nodes(graph))

    indexes.append(start)

    for idx in range(num_graphs):
        pooling_matrix[idx, range(indexes[idx], indexes[idx + 1])] = (1 / (indexes[idx + 1] - indexes[idx]))

    return pooling_matrix


def get_overall_adjoint_matrix(graphs):
    node_sizes = []

    for graph in graphs:
        nx.nodes(graph)
        node_sizes.append(nx.number_of_nodes(graph))

    num_graphs = len(graphs)
    num_nodes = np.sum(node_sizes)

    print("num of nodes : {}".format(num_nodes))

    adj_matrix = [[0 for i in range(num_nodes)] for k in range(num_nodes)]

    start = 0

    indexes = []

    for idx, graph in tqdm(enumerate(graphs)):
        edges = nx.to_edgelist(graph)
        indexes.append(start)
        for edge in tqdm(edges):
            u = edge[0]
            v = edge[1]

            u += start
            v += start

            adj_matrix[u][v] = 1

        start += len(nx.nodes(graph))

    adj_matrix = np.matrix(adj_matrix)
    adj_matrix = sp.coo_matrix(adj_matrix)

    # sp.save_npz("politifact_adj_matrix_basic", adj_matrix)
    return adj_matrix


def get_all_documents(news_source, all_tweet_id_text_dict):
    tweet_ids = []
    documents = []

    for tweet_id, text in all_tweet_id_text_dict.items():
        tweet_ids.append(tweet_id)

        if str(news_source) in str(tweet_id):
            documents.append(" ")
            print("Root node tweet id : {}".format(tweet_id))
        else:
            documents.append(text)

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    vectorizer.fit(documents)

    pickle.dump(vectorizer, open("{}_doc_vectorizer.pkl".format(news_source), "wb"))

    transformed_docs = vectorizer.transform(documents).todense()

    from sklearn.decomposition import PCA

    pca = PCA(n_components=10)
    transformed_docs = pca.fit_transform(transformed_docs)

    print("tranformed docs ", transformed_docs.shape)

    single_node_embeddings = transformed_docs[0, :].transpose()

    print("single doc transofmred doc", single_node_embeddings.shape, flush=True)

    all_tweet_id_text_dict = dict()

    for idx in range(transformed_docs.shape[0]):
        all_tweet_id_text_dict[tweet_ids[idx]] = transformed_docs[idx, :]

    return all_tweet_id_text_dict


def get_all_documents_glove_embeddings(news_source, all_tweet_id_text_dict):
    # model_path = "/home/dmahudes/temporal_event_analysis/pre_train_model/glove.twitter.27B.100d.w2vformat.txt"
    # model_path = "/home/dmahudes/temporal_event_analysis/pre_train_model/glove.twitter.27B.25d.w2vformat.txt"

    # model_path = "/home/dmahudes/temporal_event_analysis/pre_train_model/glove.twitter.27B.200d.w2vformat.txt"
    model_path = "data/glove.twitter.27B.200d.w2vformat.txt"

    glove_model = get_gensim_model(model_path)

    tweet_id_embddings_dict = dict()

    for tweet_id, text in tqdm(all_tweet_id_text_dict.items()):
        if str(news_source) in str(tweet_id):
            tweet_id_embddings_dict[tweet_id] = np.zeros((200,))
            print("root tweet id : {}".format(tweet_id), flush=True)

        else:
            tweet_id_embddings_dict[tweet_id] = get_tweet_latent_embeddings(text, glove_model)

    pickle.dump(tweet_id_embddings_dict, open("{}_tweet_id_glove_embeddings_dict.pkl".format(news_source), "wb"))

    # vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    # vectorizer.fit(documents)

    # pickle.dump(vectorizer, open("{}_doc_vectorizer.pkl".format(news_source), "wb"))

    # transformed_docs = vectorizer.transform(documents).todense()

    # print("tranformed docs ", transformed_docs.shape)
    #
    # single_node_embeddings = transformed_docs[0, :].transpose()
    #
    # print("single doc transofmred doc", single_node_embeddings.shape, flush=True)
    #
    # all_tweet_id_text_dict = dict()

    # for idx in range(transformed_docs.shape[0]):
    #     all_tweet_id_text_dict[tweet_ids[idx]] = transformed_docs[idx, :]

    return tweet_id_embddings_dict


def get_feature_matrix(graphs, tweet_id_feature_dict, graph_tweet_id_node_id_dicts):
    node_features = []

    for idx in range(len(graphs)):
        graph = graphs[idx]
        tweet_id_node_id_dict = graph_tweet_id_node_id_dicts[idx]
        node_id_tweet_id_dict = reverse_dict(tweet_id_node_id_dict)
        for node_id in nx.nodes(graph):
            # print("node id", node_id)
            tweet_id = node_id_tweet_id_dict[node_id]
            tweet_feature = np.array(tweet_id_feature_dict[tweet_id]).transpose()
            # print("tweet feature ", tweet_feature.shape)
            # node_features.append(tweet_feature)
            node_features.append(np.expand_dims(np.array(tweet_id_feature_dict[tweet_id]), axis=1).transpose())
    # print("no. of nodes : {}".format(len(node_features)))
    # return sp.csr_matrix(np.concatenate(node_features, axis=1).transpose())

    return sp.csr_matrix(np.concatenate(node_features))


def get_glove_feature_matrix(graphs, tweet_id_feature_dict, graph_tweet_id_node_id_dicts):
    node_features = []

    for idx in range(len(graphs)):
        graph = graphs[idx]
        tweet_id_node_id_dict = graph_tweet_id_node_id_dicts[idx]
        node_id_tweet_id_dict = reverse_dict(tweet_id_node_id_dict)
        for node_id in nx.nodes(graph):
            # print("node id", node_id)
            tweet_id = node_id_tweet_id_dict[node_id]
            tweet_feature = np.expand_dims(np.array(tweet_id_feature_dict[tweet_id]), axis=1).transpose()
            if len(tweet_feature.shape) > 1:
                if tweet_feature.shape[0] != 1 or tweet_feature.shape[1] != 200:
                    print("tweet feature : ", tweet_feature.shape)
            else:
                tweet_feature = np.zeros((1, 200))
                print(tweet_feature.shape)

            node_features.append(tweet_feature)

    # print("no. of nodes : {}".format(len(node_features)))

    print("batch_embdding_Size before concat ", len(node_features), node_features[0].shape)

    return sp.csr_matrix(np.concatenate(node_features))


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


# trasforma matrici in tuple
def to_tuple(mat):
    if not sp.isspmatrix_coo(mat):
        mat = mat.tocoo()
    idxs = np.vstack((mat.row, mat.col)).transpose()
    values = mat.data
    shape = mat.shape
    return idxs, values, shape


# trasforma matrici sparse in tuble
def sparse_to_tuple(sparse_mat):
    if isinstance(sparse_mat, list):
        for i in range(len(sparse_mat)):
            sparse_mat[i] = to_tuple(sparse_mat[i])
    else:
        sparse_mat = to_tuple(sparse_mat)
    return sparse_mat


# normalizza la matrice delle feature per riga e la trasforma in tupla
def process_features(features: object) -> object:
    features /= features.sum(1).reshape(-1, 1)
    features[np.isnan(features) | np.isinf(features)] = 0  # serve per le features dei nodi globali, che sono di soli 0.
    return sparse_to_tuple(sp.csr_matrix(features))


# renormalization trick della matrice di adiacenza
def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1.0).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return sp.csr_matrix(a_norm)


# conversione a tupla e normalizzazione della matrice d'adiacenza
def preprocess_adj(adj, is_gcn, symmetric=True):
    if is_gcn:
        adj = adj + sp.eye(adj.shape[0])  # ogni nodo ha come vicino anche se stesso, fa parte di GCN
    adj = normalize_adj(adj, symmetric)
    return sparse_to_tuple(adj)


def get_input_for_batches(news_source, batch_size, time_interval, input_dim):
    all_network_x_graphs, all_tweet_id_text_dict, one_hot_labels, labels, all_tweet_id_node_ids_dicts = get_all_propagation_graphs(
        news_source, time_interval)

    # all_tweet_id_text_dict = get_all_documents(news_source, all_tweet_id_text_dict)

    # tweet_id_embeddings_dict = all_tweet_id_text_dict

    tweet_id_embeddings_dict = get_all_documents_glove_embeddings(news_source, all_tweet_id_text_dict)

    # tweet_id_embeddings_dict = pickle.load(open("{}_tweet_id_glove_embeddings_dict.pkl".format(news_source), "rb"))

    print("all_network_x_graphs count : {}".format(len(all_network_x_graphs)))
    print("all_tweet_id_text_dict count : {}".format(len(one_hot_labels)))
    print("all_tweet_id_node_ids_dicts count: {}".format(len(all_tweet_id_node_ids_dicts)))

    train_network_x_graphs, test_network_x_graphs, train_one_hot_labels, test_one_hot_labels, train_tweet_id_node_ids_dicts, test_tweet_id_node_ids_dicts = train_test_split(
        all_network_x_graphs, one_hot_labels, all_tweet_id_node_ids_dicts, stratify=labels,
        test_size=0.2, random_state=42)

    # all_network_x_graphs = train_network_x_graphs
    # labels = train_one_hot_labels
    # all_tweet_id_node_ids_dicts = train_tweet_id_node_ids_dicts

    dump_batch_inputs(batch_size, news_source, "train", train_network_x_graphs, train_tweet_id_node_ids_dicts,
                      train_one_hot_labels, tweet_id_embeddings_dict, time_interval, input_dim)

    dump_batch_inputs(batch_size, news_source, "test", test_network_x_graphs, test_tweet_id_node_ids_dicts,
                      test_one_hot_labels, tweet_id_embeddings_dict, time_interval, input_dim)

    # dump_glove_feature_batch_embeddings(batch_size, news_source, "train", train_network_x_graphs,
    #                                     train_tweet_id_node_ids_dicts, train_one_hot_labels, tweet_id_embeddings_dict, time_interval)
    # dump_glove_feature_batch_embeddings(batch_size, news_source, "test", test_network_x_graphs,
    #                                     test_tweet_id_node_ids_dicts,
    #                                     test_one_hot_labels, tweet_id_embeddings_dict, time_interval)


def dump_glove_feature_batch_embeddings(batch_size, news_source, split_label, all_network_x_graphs,
                                        all_tweet_id_node_ids_dicts, labels,
                                        all_tweet_id_text_dict, time_interval):
    data_dir = "data/time_batch_data"
    create_dir(data_dir)

    data_dir = "{}/batch_{}".format(data_dir, time_interval)
    create_dir(data_dir)

    data_dir = "{}/{}".format(data_dir, news_source)

    create_dir(data_dir)

    data_dir = "{}/glove_feat".format(data_dir)
    create_dir(data_dir)

    data_dir = "{}/{}".format(data_dir, split_label)
    create_dir(data_dir)

    num_samples = len(labels)

    num_batches = int(ceil(num_samples / batch_size))

    for idx in tqdm(range(num_batches)):
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size

        batch_graphs = all_network_x_graphs[start_idx: end_idx]
        batch_labels = labels[start_idx: end_idx]
        batch_mapping_dicts = all_tweet_id_node_ids_dicts[start_idx: end_idx]

        batch_node_features = get_glove_feature_matrix(batch_graphs, all_tweet_id_text_dict, batch_mapping_dicts)

        # print("node feature matrix shape ", batch_node_features.shape)

        batch_node_features = process_features(batch_node_features)

        batch_input = [batch_node_features]
        # batch_inputs.append(batch_input)

        pickle.dump(batch_input, open("{}/batch_{}.pkl".format(data_dir, idx), "wb"))


def dump_batch_inputs(batch_size, news_source, split_label, all_network_x_graphs, all_tweet_id_node_ids_dicts, labels,
                      all_tweet_id_text_dict, time_interval, input_dim):
    data_dir = "data/time_batch_data"
    create_dir(data_dir)

    data_dir = "{}/batch_{}".format(data_dir, time_interval)
    create_dir(data_dir)

    data_dir = "{}/{}_{}".format(data_dir, news_source, input_dim)

    create_dir(data_dir)

    data_dir = "{}/{}".format(data_dir, split_label)
    create_dir(data_dir)

    batch_inputs = []

    num_samples = len(labels)

    num_batches = int(ceil(num_samples / batch_size))

    for idx in tqdm(range(num_batches)):
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size

        batch_graphs = all_network_x_graphs[start_idx: end_idx]
        batch_labels = labels[start_idx: end_idx]
        batch_mapping_dicts = all_tweet_id_node_ids_dicts[start_idx: end_idx]

        batch_adj_matrix = get_overall_adjoint_matrix(batch_graphs)
        batch_pooling_matrix = get_batch_pooling_matrix(batch_graphs)

        batch_adj_matrix = preprocess_adj(batch_adj_matrix, True, False)

        # batch_node_features = get_feature_matrix(batch_graphs, all_tweet_id_text_dict, batch_mapping_dicts)

        batch_node_features = get_glove_feature_matrix(batch_graphs, all_tweet_id_text_dict, batch_mapping_dicts)

        print("node feature matrix shape ", batch_node_features.shape)

        batch_node_features = process_features(batch_node_features)

        batch_input = [batch_adj_matrix, batch_node_features, batch_labels, batch_pooling_matrix]
        # batch_inputs.append(batch_input)

        pickle.dump(batch_input, open("{}/batch_{}.pkl".format(data_dir, idx), "wb"))

    # pickle.dump(batch_inputs, open("{}_batched_inputs.pkl".format(news_source), "wb"))
    # return batch_inputs


def get_gensim_model(model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    return model


def get_tweet_latent_embeddings(text_contents, model):
    word_embeddings = []

    tokens = twitter_tokenize(text_contents)

    for token in tokens.split():
        try:
            word_embeddings.append(model[token])
        except:
            pass

    if len(word_embeddings) > 0:
        try:
            return np.mean(word_embeddings, axis=0)
        except:
            return np.zeros((200,))

    return np.zeros((200,))


def analyze_dataset(news_source):
    graphs, all_tweet_id_text_dict, one_hot_labels, labels, all_tweet_id_node_ids_dicts = get_all_propagation_graphs(
        news_source)

    # graphs = filter_graphs(graphs, 1500)

    graph_sizes = []

    for graph in graphs:
        graph_sizes.append(nx.number_of_nodes(graph))

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    plt.hist(graph_sizes, normed=True, bins=30)

    plt.savefig("figures/{}_graph_distribution.png".format(news_source))


def get_random_bfs_sequence(G):
    start_id = 0
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]

    max_prev_nodes = 0

    while len(start) > 0:
        next = []

        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)

            if neighbor is not None:
                next = next + neighbor

        max_prev_nodes = max(max_prev_nodes, len(next))

        start = next

    # print("max previous nodes : {}".format(max_prev_nodes))
    return max_prev_nodes


if __name__ == "__main__":
    news_source = "gossipcop"

    news_source = "politifact"

    # analyze_dataset(news_source)

    # time_intervals = [12, 24, 36, 48, 60, 72, 84, 96]

    # time_intervals = [12, 24, 36, 48, 60, 72, 84, 96]

    # time_intervals = [12, 24, 36,48,  60, 72, 84, 96]
    #
    input_dim = 200
    #
    time_intervals = [3, 6]
    #
    # # time_intervals = [None]
    #
    for time_interval in time_intervals:
        print("=============Time Interval : {}  ==========".format(time_interval))
        start_time = time.time()
        # get_classificaton_results_tpnf("data/train_test_data", "politifact", time_interval)
        # get_classificaton_results_tpnf("data/train_test_data", "gossipcop", time_interval)
        get_input_for_batches(news_source, 8, time_interval, input_dim)

        print("\n\n================Exectuion time - {} ==================================\n".format(
            time.time() - start_time))

    # graphs, all_tweet_id_text_dict, one_hot_labels, labels, all_tweet_id_node_ids_dicts, hidden_state = get_all_propagation_graphs(
    #     news_source="gossipcop", args = Args())

    # max_nodes = 5000

    # graphs = filter_graphs(graphs, max_nodes)

    # max_breadths = []
    #
    # for graph in graphs:
    #     max_breadths.append(get_random_bfs_sequence(graph))
    #
    # print("Mean : {}".format(np.mean(max_breadths)))
    # print("Max : {}".format(max(max_breadths)))
    # print("Min : {}".format(min(max_breadths)))
    # print(np.histogram(max_breadths))

    exit(1)

    all_network_x_graphs, all_tweet_id_text_dict, labels = get_all_propagation_graphs(news_source)

    all_tweet_id_text_dict = get_all_documents(news_source, all_tweet_id_text_dict)

    pickle.dump(all_network_x_graphs, open("{}_all_networkx_graphs.pkl".format(news_source), "wb"))
    pickle.dump(all_tweet_id_text_dict, open("{}_graphs_text_dict.pkl".format(news_source), "wb"))
    pickle.dump(labels, open("{}_labels.pkl".format(news_source), "wb"))

    # all_network_xx_graphs = filter_graphs(all_network_x_graphs, 2000)

    # adj_matrix = get_overall_adjoint_matrix(all_network_x_graphs)

    # nodes_stats(all_network_xx_graphs)
    #
    # print(len(all_network_xx_graphs))
