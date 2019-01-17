import csv
import pickle
from os import listdir
from os.path import isfile, join

import numpy as np

from analysis_util import get_propagation_graphs, create_dir, equal_samples
from pre_process_util import load_configuration, get_database_connection


def get_news_ids_used_for_propagation_network(news_source):
    fake_prop_graph, real_prop_graph = get_propagation_graphs(news_source)

    fake_news_ids = []
    for graph in fake_prop_graph:
        fake_news_ids.append(graph.tweet_id)

    real_news_ids = []
    for graph in real_prop_graph:
        real_news_ids.append(graph.tweet_id)

    print("No. of fake news ids : {}".format(len(fake_news_ids)))
    print("No. of real news ids : {}".format(len(real_news_ids)))


def get_alternate_text_content(db, is_fake, news_source, news_id):
    if is_fake:
        news_article_source_collection = db.fake_news_source_article
        news_format_collection = db.fake_news_format
    else:
        news_format_collection = db.real_news_format
        if news_source == "politifact":
            news_article_source_collection = db.real_news_source_article
        else:
            news_article_source_collection = db.real_news_source_article_mod

    news_source_article = news_article_source_collection.find_one({"id": news_id})
    if news_source_article:
        if "title" in news_source_article and len(news_source_article["title"]) > 0:
            return news_source_article["title"]
        else:
            try:
                return news_source_article["meta_data"]["og"]["description"]
            except:
                pass

    # Get the news format object and use the query used as the text content
    news_article_format = news_format_collection.find_one({"id": news_id})
    return news_article_format["statement"]


def get_samples_order(news_graphs_save_dir, news_source):
    fake_prop_graph, real_prop_graph = get_propagation_graphs(news_graphs_save_dir, news_source)
    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    fake_graph_ids = []
    for graph in fake_prop_graph:
        fake_graph_ids.append(graph.tweet_id)

    pickle.dump(fake_graph_ids,
                open("data/baseline_data/{}_fake_sample_news_ordered_ids.pkl".format(news_source), "wb"))

    real_graph_ids = []
    for graph in real_prop_graph:
        real_graph_ids.append(graph.tweet_id)

    pickle.dump(real_graph_ids,
                open("data/baseline_data/{}_real_sample_news_ordered_ids.pkl".format(news_source), "wb"))


def get_news_article_text_content_used_for_propagation_network(db, news_graphs_save_dir, news_source):
    fake_prop_graph, real_prop_graph = get_propagation_graphs(news_graphs_save_dir, news_source)
    count = 0

    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)

    fake_news_text_contents = dict()
    for graph in fake_prop_graph:
        fake_news_text_contents[graph.tweet_id] = graph.text
        if len(graph.text) < 3:
            print(graph.tweet_id)
            count += 1
            fake_news_text_contents[graph.tweet_id] = get_alternate_text_content(db, True, news_source, graph.tweet_id)

    print("Fake graphs without text : {}".format(count))

    count = 0
    real_news_text_contents = dict()
    for graph in real_prop_graph:
        real_news_text_contents[graph.tweet_id] = graph.text
        if len(graph.text) < 3:
            print(graph.tweet_id)
            count += 1
            real_news_text_contents[graph.tweet_id] = get_alternate_text_content(db, False, news_source, graph.tweet_id)

    print("real graphs without text : {}".format(count))
    return fake_news_text_contents, real_news_text_contents


def dump_text_content(dump_folder, label_folder, news_id_text_content_dict):
    create_dir(dump_folder)
    data_dir = dump_folder + "/" + label_folder
    create_dir(data_dir)

    for news_id, text_content in news_id_text_content_dict.items():
        with open("{}/{}.txt".format(data_dir, news_id), "w", encoding="UTF-8") as file:
            file.write(text_content)

    print("dumping files completed : {}".format(label_folder))


def count_news_articles_without_text_content(news_id_text_content_dict):
    count = 0
    words_lengths = []

    shortened_lengths = []
    for news_id, text_content in news_id_text_content_dict.items():
        words = text_content.split()

        words_lengths.append(len(words))
        shortened_lengths.append(min(len(words), 1000))

        if len(text_content) < 5:
            count += 1

    print("News ids without text content : {}".format(count))
    print("Original Mean  : {}  SD : {}   Min: {}  Max: {}".format(np.mean(words_lengths), np.std(words_lengths),
                                                                   np.min(words_lengths), np.max(words_lengths)))
    print(
        "Shortened Mean  : {}  SD : {}   Min: {}  Max: {}".format(np.mean(shortened_lengths), np.std(shortened_lengths),
                                                                  np.min(shortened_lengths), np.max(shortened_lengths)))


def remove_escape_characters(text_content):
    text_content = text_content.replace('\n', ' ')
    text_content = text_content.replace('\t', ' ')
    words = text_content.split(" ")
    return " ".join(words[:1000])


def remove_escape_characters_ordered_file(text_content):
    text_content = text_content.replace('\n', ' ')
    text_content = text_content.replace('\t', ' ')
    words = text_content.split(" ")
    return " ".join(words[:1000])


def dump_csv_format_file(dump_folder, file_name, news_id_text_content_dict):
    with open("{}/{}.csv".format(dump_folder, file_name), "w", encoding="UTF-8") as file:
        writer = csv.writer(file, delimiter='\t')
        for news_id, text_content in news_id_text_content_dict.items():
            writer.writerow([news_id, remove_escape_characters(text_content)])


def dump_csv_format_file_comma_separated(dump_folder, file_name, news_id_text_content_dict, news_source, label):
    ordered_sample_ids = pickle.load(
        open("data/baseline_data/{}_{}_sample_news_ordered_ids.pkl".format(news_source, label), "rb"))
    with open("{}/{}.csv".format(dump_folder, file_name), "w", encoding="UTF-8") as file:
        writer = csv.writer(file)
        for news_id in ordered_sample_ids:
            writer.writerow([news_id, remove_escape_characters_ordered_file(news_id_text_content_dict[news_id])])


def dump_LIWC_Representation(liwc_file_path, output_file):
    f_out = open(output_file, 'w+')
    with open(liwc_file_path, encoding="UTF-8") as f_fake:
        for line in f_fake:
            line = line.strip()
            all_data = line.split('\t')
            if all_data[0] == 'Source (A)':
                continue
            ID = all_data[0]
            feats = all_data[2:]
            f_out.write(ID + ',')
            f_out.write(','.join(f for f in feats))
            f_out.write('\n')

    f_out.close()


def dump_ordered_rst_representation(rst_folder, news_source, fake_out_file, real_out_file):
    dir_path = rst_folder

    all_relations = set()
    org_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    News_RSTFeats = dict()
    for of in org_files:
        ID = of[:of.index('.txt')]
        file_name = dir_path + '/' + of
        relation_num = dict()
        with open(file_name) as f_rst:
            for line in f_rst:
                line = line.replace('\'', '')
                line = line.replace(' ', '')
                tmp_relation = line.split(',')[3]
                relation = tmp_relation[:-2]
                all_relations.add(relation)
                if relation in relation_num:
                    num = relation_num[relation]
                    num += 1
                    relation_num[relation] = num
                else:
                    relation_num[relation] = 1
        News_RSTFeats[ID] = relation_num

    all_relations = list(all_relations)
    print(all_relations)

    fake_ordered_sample_ids = pickle.load(
        open("data/baseline_data/{}_{}_sample_news_ordered_ids.pkl".format(news_source, "fake"), "rb"))

    f_out = open(fake_out_file, 'w+')
    for news_id in fake_ordered_sample_ids:
        # for news, rn in News_RSTFeats.items():
        #     f_out.write(news + '\t')
        rn = News_RSTFeats[news_id]
        feats = []
        for al in all_relations:
            if al in rn:
                num = rn[al]
            else:
                num = 0
            feats.append(num)
        f_out.write('\t'.join(str(x) for x in feats))
        f_out.write('\n')
    f_out.close()

    real_ordered_sample_ids = pickle.load(
        open("data/baseline_data/{}_{}_sample_news_ordered_ids.pkl".format(news_source, "real"), "rb"))

    f_out = open(real_out_file, 'w+')
    for news_id in real_ordered_sample_ids:
        # for news, rn in News_RSTFeats.items():
        #     f_out.write(news + '\t')
        rn = News_RSTFeats[news_id]
        feats = []
        for al in all_relations:
            if al in rn:
                num = rn[al]
            else:
                num = 0
            feats.append(num)
        f_out.write('\t'.join(str(x) for x in feats))
        f_out.write('\n')
    f_out.close()


if __name__ == "__main__":
    # get_news_ids_used_for_propagation_network("politifact")

    config = load_configuration("project.config")
    db = get_database_connection(config)

    news_source = "gossipcop"

    dump_ordered_rst_representation("data/baseline_features/rst/raw_parsed_data/gossipcop",news_source,
                                    "data/baseline_features/rst/raw_parsed_data/gossipcop_fake_rst_features.csv",
                                    "data/baseline_features/rst/raw_parsed_data/gossipcop_real_rst_features.csv"
                                    )

    # dump_LIWC_Representation("data/baseline_features/liwc_features/LIWC2015_{}_fake_text_contents_ordered_new.txt".format(news_source),
    #                          "data/baseline_features/liwc_features/{}_fake_liwc.csv".format(news_source))
    #
    # dump_LIWC_Representation("data/baseline_features/liwc_features/LIWC2015_{}_real_text_contents_ordered_new.txt".format(news_source),
    #                          "data/baseline_features/liwc_features/{}_real_liwc.csv".format(news_source))

    exit(1)

    fake_news_text_contents, real_news_text_contents = get_news_article_text_content_used_for_propagation_network(db,
                                                                                                                  "data/saved_new_no_filter",
                                                                                                                  news_source)

    get_samples_order("data/saved_new_no_filter", news_source)
    # dump_csv_format_file("data/baseline_data", "{}_fake_text_contents".format(news_source), fake_news_text_contents)
    # dump_csv_format_file("data/baseline_data", "{}_real_text_contents".format(news_source), real_news_text_contents)
    #
    dump_csv_format_file_comma_separated("data/baseline_data", "{}_fake_text_contents_ordered".format(news_source),
                                         fake_news_text_contents, news_source, "fake")
    dump_csv_format_file_comma_separated("data/baseline_data", "{}_real_text_contents_ordered".format(news_source),
                                         real_news_text_contents, news_source, "real")

    # count_news_articles_without_text_content(fake_news_text_contents)
    # count_news_articles_without_text_content(real_news_text_contents)

    # dump_text_content("data/baseline_data", "gossipcop_fake", fake_news_text_contents)
    # dump_text_content("data/baseline_data", "gossipcop_real", real_news_text_contents)
