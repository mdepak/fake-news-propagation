import csv
import json
import mmap
import os
import pickle
import queue
import re
import shutil
import string
import sys
import traceback
from datetime import datetime
from pathlib import Path

import datefinder
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from pymongo import UpdateOne
from tqdm import tqdm
import newspaper

from analysis_util import get_propagation_graphs
from baseline_feature_extraction import dump_LIWC_Representation
from pre_process_util import load_configuration, get_database_connection, get_news_articles
from util.constants import RETWEET_EDGE, REPLY_EDGE, RETWEET_NODE, REPLY_NODE
from util.util import tweet_node


def get_reply_of_replies(replies: list, result_dict: dict):
    for reply in replies:
        if reply:
            if "engagement" in reply:
                get_reply_of_replies(reply["engagement"]["tweet_replies"], result_dict)

            result_dict[reply["id"]] = reply["text"]


def get_web_archieve_results(search_url):
    try:
        archieve_url = "http://web.archive.org/cdx/search/cdx?url={}&output=json".format(search_url)

        response = requests.get(archieve_url)
        response_json = json.loads(response.content)

        response_json = response_json[1:]

        return response_json

    except:
        return None


def get_website_url_from_arhieve(url):
    archieve_results = get_web_archieve_results(url)
    if archieve_results:
        modified_url = "https://web.archive.org/web/{}/{}".format(archieve_results[0][1], archieve_results[0][2])
        return modified_url
    else:
        return url


def dump_friends_file_as_lines(dataset_file, out_file):
    pattern = re.compile(rb'{([^{}]+)}',
                         re.DOTALL | re.IGNORECASE | re.MULTILINE)

    with open(out_file, "w", 100) as out_file:
        with open(dataset_file, 'r') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                for match in pattern.findall(m):
                    data = "{" + str(match.decode('utf-8')) + "}\n"
                    out_file.write(data)


def dump_social_network_to_db(db, folder):
    friends_coll = db.twitter_user_friends_collection

    batch_update_list = []

    files = os.listdir(folder)
    user_names = set([file[:file.find(".csv")] for file in files])

    print("actual files : {}".format(len(user_names)), flush=True)

    saved_user_names = set(friends_coll.distinct("user_name"))
    print("saved user names  : {}".format(len(saved_user_names)), flush=True)

    user_names = user_names.difference(saved_user_names)

    print("user names to be saved : {}".format(len(user_names)), flush=True)

    for idx, user_name in enumerate(user_names):
        try:
            friends_user_names = get_friends_names("{}/{}.csv".format(folder, user_name))

            batch_update_list.append(UpdateOne({"user_name": user_name},
                                               {"$set": {"user_name": user_name, "friends_name": friends_user_names}},
                                               upsert=True))

            if idx % 10000 == 0:
                try:
                    friends_coll.bulk_write(batch_update_list, ordered=False)
                except:
                    print("Exception")
                    traceback.print_exc(file=sys.stdout)

                batch_update_list = []
                print("bulk update {}".format(idx), flush=True)

        except Exception as ex:
            print("Exception in file : {}/{} : {}".format(folder, user_name, str(ex)))
            traceback.print_exc(file=sys.stdout)

    if len(batch_update_list) > 0:
        friends_coll.bulk_write(batch_update_list, ordered=False)
        print("bulk update", flush=True)

    print("completed dumping for folder {}".format(folder))


def get_user_to_fetch(all_user_file, user_ids_user_name_dict, db):
    user_names = set(json.load(open(all_user_file))["user_names"])

    friends_coll = db.twitter_user_friends_collection

    fake_friends_collection = db.fake_twitter_user_followees
    real_friends_collection = db.real_twitter_user_followees

    fake_users_ids = set(fake_friends_collection.distinct("user_id"))
    real_users_ids = set(real_friends_collection.distinct("user_id"))

    all_user_ids = set()
    all_user_ids.update(fake_users_ids)
    all_user_ids.update(real_users_ids)

    id_fetched_user_names = set()

    user_ids_user_name_dict = json.load(open(user_ids_user_name_dict))

    for user_id, user_name in user_ids_user_name_dict.items():
        if int(user_id) in all_user_ids:
            id_fetched_user_names.add(user_name)

    print("actual files : {}".format(len(user_names)), flush=True)

    saved_user_names = set(friends_coll.distinct("user_name"))
    print("saved user names  : {}".format(len(saved_user_names)), flush=True)

    user_names = user_names.difference(saved_user_names)

    print("user names to be collected : {}".format(len(user_names)), flush=True)

    print("ID fetched users : {}".format(len(id_fetched_user_names)))

    user_names = user_names.difference(id_fetched_user_names)

    print("Final set of user names to be fetched : {}".format(len(user_names)))

    json.dump({"user_names": list(user_names)}, open("politifact_user_names_to_collect.json", "w"))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def dump_user_friends_data(db, user_names_file, dump_out_file):
    user_names = json.load(open(user_names_file))["user_names"]
    friends_collection = db.twitter_user_friends_collection
    with open(dump_out_file, "w", 1000) as file:
        for user_name_chunk in chunks(list(user_names), 10000):
            for user_info in friends_collection.find({"user_name": {"$in": user_name_chunk}}, {"_id": 0}):
                file.write(json.dumps(user_info))
                file.write("\n")

    print("Compeleted dumping {}".format(dump_out_file))


def dump_user_id_friends_data(db, user_id_dict_file, dump_out_file):
    user_id_name_dict = json.load(open(user_id_dict_file))

    user_ids = user_id_name_dict.keys()

    user_ids = [int(user_id) for user_id in user_ids]

    user_ids = set(user_ids)

    fake_friends_collection = db.fake_twitter_user_followees
    real_friends_collection = db.real_twitter_user_followees

    with open(dump_out_file, "w", 1000) as file:

        for user_ids_chunk in chunks(list(user_ids), 10000):
            for user_info in fake_friends_collection.find({"user_id": {"$in": user_ids_chunk}}, {"_id": 0}):
                user_ids.remove(user_info["user_id"])
                file.write(json.dumps(user_info) + "\n")

        for user_ids_chunk in chunks(list(user_ids), 10000):
            for user_info in real_friends_collection.find({"user_id": {"$in": user_ids_chunk}}, {"_id": 0}):
                user_ids.remove(user_info["user_id"])
                file.write(json.dumps(user_info) + "\n")

    print("Compeleted dumping {}".format(dump_out_file))


def get_friends_names(friends_file):
    try:
        with open(friends_file, encoding="UTF-8") as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            return lines[1:]

    except:
        return []


def write_file_if_not_exist(output_folder, user_id_followee_json_data):
    file_path = "{}/{}.json".format(output_folder, user_id_followee_json_data["user_id"])
    if not os.path.exists(file_path):
        json.dump(user_id_followee_json_data, open(file_path, "w"))


def write_file_user_name_if_not_exist(output_folder, user_name_followee_json_data):
    file_path = "{}/{}.json".format(output_folder, user_name_followee_json_data["user_name"])
    if not os.path.exists(file_path):
        json.dump(user_name_followee_json_data, open(file_path, "w"))


def dump_social_network_user_id_single_file(input_ids_file, output_folder):
    with open(input_ids_file) as file:
        for line in tqdm(file):
            write_file_if_not_exist(output_folder, json.loads(line))


def dump_social_network_user_name_single_file(input_names_file, output_folder):
    with open(input_names_file) as file:
        for line in tqdm(file):
            write_file_user_name_if_not_exist(output_folder, json.loads(line))


def download_news_article(url):
    news_article = Article(url)
    news_article.download()
    news_article.parse()
    return news_article


def get_dataset_publication_date(news_id_publish_time_dict, news_id_fact_statement_date_dict, news_id_source_date_dict):
    """
    Check the different dates and choose the right date for filtering noise
    :param news_id_publish_time:
    :param news_id_fact_statement_date_dict:
    :param news_id_source_date_dict:
    :return:
    """
    all_news_ids = news_id_fact_statement_date_dict.keys()

    news_id_selected_filter_date = dict()

    for news_id in all_news_ids:
        if news_id in news_id_publish_time_dict:
            news_id_selected_filter_date[news_id] = news_id_publish_time_dict[news_id].timestamp()
        elif news_id in news_id_source_date_dict:
            news_id_selected_filter_date[news_id] = news_id_source_date_dict[news_id].timestamp()
        elif news_id in news_id_fact_statement_date_dict:
            news_id_selected_filter_date[news_id] = datetime.strptime(news_id_fact_statement_date_dict[news_id],
                                                                      "%Y-%m-%d").timestamp()

    return news_id_selected_filter_date


def get_news_articles_published_time(db, is_fake):
    news_id_publish_time_dict = dict()

    if is_fake:
        news_source_article_collection = db.fake_news_source_article
    else:
        news_source_article_collection = db.real_news_source_article

    for news_source in news_source_article_collection.find({"news_source": "politifact"}):
        news_id = news_source["id"]
        if news_source and news_source["publish_date"]:
            news_id_publish_time_dict[news_id] = news_source["publish_date"]

    return news_id_publish_time_dict


# def get_news_articles_published_time(dataset_file):
#     dataset = get_news_articles(dataset_file)
#     news_id_publish_time = dict()
#     count = 0
#     print("total no. of articles : {}".format(len(dataset)))
#     for news in dataset:
#         if "publish_date" in news["text_content"] and news["text_content"]["publish_date"]:
#             count += 1
#             print(news["text_content"]["publish_date"])
#
#         # if "url" in news["text_content"]:
#         #     try:
#         #         formatted_url = news["text_content"]["url"].lstrip("'").rstrip("'").lstrip("/")
#         #
#         #         print("Formatted url : {}".format(formatted_url))
#         #
#         #         news_article = download_news_article(formatted_url)
#         #         print("News id : {} publish data : {}".format(news["id"], news_article.publish_date), flush=True)
#         #         news_id_publish_time[news["id"]] = news_article.publish_date.timestamp()
#         #     except Exception as ex:
#         #         print(ex)
#     print("old wrong present publish date count : {}".format(count))
#     return news_id_publish_time


def get_publish_date_from_sources_politifact(db, is_fake):
    if is_fake:
        news_collection = db.fake_news_collection
        news_format_collection = db.fake_news_format
    else:
        news_collection = db.real_news_collection
        news_format_collection = db.real_news_format

    news_id_fact_statement_date_dict = dict()

    news_id_source_date_dict = dict()

    for news_format in news_format_collection.find({"news_source": "politifact"}):
        news_id = news_format["id"]

        news_id_int = int(news_id.replace("politifact", ""))

        news = news_collection.find_one({"id": news_id_int})

        publish_date = get_formatted_news_publish_date(news)

        try:
            if publish_date:
                publish_date = next(publish_date)

                if publish_date:
                    news_id_source_date_dict[news_id] = publish_date
        except StopIteration:
            pass

        news_id_fact_statement_date_dict[news_id] = news["statement_date"]

    return news_id_fact_statement_date_dict, news_id_source_date_dict


def get_formatted_news_publish_date(fake_news):
    try:
        id = fake_news['id']
        source_html = fake_news['sources']
        sources_soup = BeautifulSoup(source_html)
        sources = sources_soup.find_all('p')
        if not sources:
            sources = sources_soup.find_all('div')
        statement = ''
        url = ''

        ## Using the first source that contains href as the fake news source if source is not removed
        ## This is not always true

        date_matches = None
        for i in range(len(sources)):
            if sources[i].find('a') is not None:
                statement_tmp = sources[i].text

                date_matches = datefinder.find_dates(statement_tmp)
                statements = re.findall(r'\"(.+?){\,,.}\"', statement_tmp)
                if len(statements) == 0:
                    statement = sources[i].a.text

                    # TODO: Verify this logic is proper
                    splits = statement_tmp.split(',')
                    for split in splits:
                        if len(statement) < len(split):
                            statement = split

                    # TODO: Why encoding is required - encoding considers quotes of string also - understand why?
                    # statement = statement.encode('utf-8')
                else:
                    # TODO: Why encoding is required - encoding considers quotes of string also - understand why?
                    # statement = statements[0].encode('utf-8')
                    statement = statements[0]
                    pass

                # TODO: Check if it is proper
                statement = str(statement).translate(str.maketrans('', '', string.punctuation))

                # statement_new = statement.translate(str.maketrans('', '', string.punctuation))  # move punctuations

                url = sources[i].a['href']
                break

        # TODO: Check if the condition is proper
        if statement == '' or len(statement.split(' ')) <= 3:
            return None

        return date_matches

    except:
        return None


def get_politifact_tweet_filter_dates(db, is_fake):
    news_id_fact_statement_date_dict, news_id_source_date_dict = get_publish_date_from_sources_politifact(
        db, is_fake=is_fake)
    news_id_publish_time_dict = get_news_articles_published_time(db, is_fake=is_fake)

    news_id_filter_date_dict = get_dataset_publication_date(news_id_publish_time_dict, news_id_fact_statement_date_dict,
                                                            news_id_source_date_dict)

    return news_id_filter_date_dict


def get_replies_from_dataset(dataset_dir, news_source, label, out_dir):
    dataset_file = "{}/{}_{}_news_complete_dataset.json".format(dataset_dir, news_source, label)
    dataset = get_news_articles(dataset_file)

    reply_id_content_dict = dict()

    for news in dataset:
        for tweet in news["tweets"]:
            get_reply_of_replies(tweet["reply"], reply_id_content_dict)

    pickle.dump(reply_id_content_dict,
                open("{}/{}_{}_reply_id_content_dict.pkl".format(out_dir, news_source, label), "wb"))


def dump_all_botometer_results(db):
    screen_name_botometer_score_dict = dict()

    for user_score in db.twitter_user_botometer_results.find():
        screen_name_botometer_score_dict[user_score["screen_name"]] = user_score["result"]

    pickle.dump(screen_name_botometer_score_dict, open("all_user_botometer_scores.pkl", "wb"))


def dump_all_user_profile_info(db, is_fake, label):
    user_id_profile_info = dict()

    all_users_ids = pickle.load(open("all_prop_graph_{}_user.pkl".format(label), "rb"))

    if is_fake:
        user_profile_collection = db.fake_twitter_user_profile
    else:
        user_profile_collection = db.real_twitter_user_profile

    for user_id in tqdm(all_users_ids):
        user_object = user_profile_collection.find_one({"user_id": user_id}, {"profile_info.statuses_count": 1,
                                                                              "profile_info.friends_count": 1,
                                                                              "profile_info.followers_count": 1,
                                                                              "profile_info.created_at": 1})
        if user_object is None:
            user_object = db.twitter_user_profile.find_one({"user_id": user_id}, {"profile_info.statuses_count": 1,
                                                                                  "profile_info.friends_count": 1,
                                                                                  "profile_info.followers_count": 1,
                                                                                  "profile_info.created_at": 1})
        if user_object and "profile_info" in user_object:
            user_id_profile_info[user_id] = user_object["profile_info"]

    print("No. of users found : {}".format(len(user_id_profile_info)))

    pickle.dump(user_id_profile_info, open("all_{}_user_profile_info.pkl".format(label), "wb"))


def get_user_aggregate_features(db, is_fake, user_names):
    dump_folder = "/home/dmahudes/fake_user_profiles"

    if is_fake:
        label_user_collection = db.fake_twitter_user_profile
    else:
        label_user_collection = db.real_twitter_user_profile

    user_profile_collection = db.twitter_user_profile

    # np.random.shuffle(user_ids)

    for user_name in tqdm(user_names):

        user_object = label_user_collection.find_one({"screen_name": user_name}, {"screen_name": 1, "user_id": 1,
                                                                                  "profile_info": 1, "_id": 0})
        if user_object is None:
            user_object = user_profile_collection.find_one({"user_id": user_name}, {"screen_name": 1, "user_id": 1,
                                                                                    "profile_info": 1, "_id": 0})

        if user_object is None:
            print('user {} not found'.format(user_name))
        else:
            json.dump(user_object, open("{}/{}.json".format(dump_folder, user_name), "w"))


def remove_escape_characters(text_content):
    text_content = text_content.replace(',', ' ')
    text_content = text_content.replace('\n', ' ')
    text_content = text_content.replace('\t', ' ')
    words = text_content.split(" ")
    return " ".join(words[:1000])


def get_missing_rst_news_content():
    news_source = "gossipcop"

    file = "/Users/deepak/Downloads/{}_content_no_ignore.tsv".format(news_source)
    # rst_folder = "/Users/deepak/Desktop/DMML/GitRepo/FakeNewsPropagation/data/baseline_features/rst/raw_parsed_data/{}".format(
    #     news_source)
    #
    # out_folder = "data/baseline_features/rst/raw_parsed_data/{}_kai".format(news_source)

    fake_news_ids = list()

    real_news_ids = list()

    all_news_folder = "data/baseline_data_kai/all_{}".format(news_source)

    kai_data_folder = "/Users/deepak/Desktop/DMML/GitRepo/FakeNewsPropagation/data/baseline_data_kai/kai_{}".format(
        news_source)

    missing_files = set()
    with open(file, encoding="UTF-8") as file:
        reader = csv.reader(file, delimiter='\t', )
        next(reader)

        for news in reader:

            if news[1] == '1':
                fake_news_ids.append(news[0])
            else:
                real_news_ids.append(news[0])

            expected_file = "{}/{}.txt.brackets".format(all_news_folder, news[0])
            out_file = "{}/{}.txt.brackets".format(kai_data_folder, news[0])

            file = Path(expected_file)
            todofile = Path("data/baseline_data_kai/{}_missed/{}.json".format(news_source, news[0]))
            if file.is_file():
                shutil.copy(expected_file, out_file)
            elif todofile.is_file():
                pass
            else:
                missing_files.add(expected_file)
                with open("data/baseline_data_kai/{}_missed/{}.json".format(news_source, news[0]), "w",
                          encoding="UTF-8") as out_file:
                    out_file.write(remove_escape_characters(news[2]))
            # file = Path(expected_file)
            # if file.is_file():
            #     with open("{}/{}.txt".format(out_folder, news[0]), "w", encoding="UTF-8") as out_file:
            #         out_file.write(remove_escape_characters(news[2]))
            # else:
            #     missing_files.add(news[0])

    pickle.dump(fake_news_ids,
                open("data/baseline_data_kai/{}_{}_sample_news_ordered_ids.pkl".format(news_source, "fake"), "wb"))
    pickle.dump(real_news_ids,
                open("data/baseline_data_kai/{}_{}_sample_news_ordered_ids.pkl".format(news_source, "real"), "wb"))

    print("No. of missing files : {}".format(len(missing_files)))


def get_files_for_liwc_parsing():
    news_source = "gossipcop"

    file = "/Users/deepak/Downloads/{}_content_no_ignore.tsv".format(news_source)

    fake_data_file = open("data/baseline_data_kai/liwc/raw_data/{}_fake_liwc_data.csv".format(news_source), "w",
                          encoding="UTF-8")

    real_data_file = open("data/baseline_data_kai/liwc/raw_data/{}_real_liwc_data.csv".format(news_source), "w",
                          encoding="UTF-8")

    fake_csv_writer = csv.writer(fake_data_file)
    real_csv_writer = csv.writer(real_data_file)

    with open(file, encoding="UTF-8") as file:
        reader = csv.reader(file, delimiter='\t', )
        next(reader)

        for news in reader:
            csv_row = [news[0], remove_escape_characters(news[2])]

            if news[1] == '1':
                fake_csv_writer.writerow(csv_row)
            else:
                real_csv_writer.writerow(csv_row)

    fake_data_file.close()
    real_data_file.close()


def get_users_in_network(prop_graph: tweet_node, edge_type=None):
    q = queue.Queue()

    q.put(prop_graph)

    users_list = set()

    while q.qsize() != 0:
        node = q.get()

        if node.user_id is not None:
            users_list.add(node.user_id)

        if edge_type == RETWEET_EDGE:
            children = node.retweet_children
        elif edge_type == REPLY_EDGE:
            children = node.reply_children
        else:
            children = node.children

        for child in children:
            q.put(child)

    return users_list


def get_node_ids_in_network_by_type(prop_graph: tweet_node, edge_type=None, node_type=None):
    q = queue.Queue()

    q.put(prop_graph)

    node_ids_set = set()

    while q.qsize() != 0:
        node = q.get()

        if node.tweet_id is not None and node.node_type == node_type:
            node_ids_set.add(node.tweet_id)

        if edge_type == RETWEET_EDGE:
            children = node.retweet_children
        elif edge_type == REPLY_EDGE:
            children = node.reply_children
        else:
            children = node.children

        for child in children:
            q.put(child)

    return node_ids_set


def get_tweets_ids_in_prop_network(prop_graph: tweet_node):
    tweet_ids = set()

    for child in prop_graph.children:
        tweet_ids.add(child.tweet_id)

    return tweet_ids


def prop_network_stats(news_source):
    fake_prop_graph, real_prop_graph = get_propagation_graphs("data/saved_new_no_filter", news_source)

    tweet_ids = set()
    retweet_ids = set()
    reply_ids = set()
    user_ids = set()

    for prop_graph in fake_prop_graph:
        tweet_ids.update(get_tweets_ids_in_prop_network(prop_graph))
        retweet_ids.update(get_node_ids_in_network_by_type(prop_graph, RETWEET_EDGE, RETWEET_NODE))
        reply_ids.update(get_node_ids_in_network_by_type(prop_graph, REPLY_EDGE, REPLY_NODE))
        user_ids.update(get_users_in_network(prop_graph))

    for prop_graph in real_prop_graph:
        tweet_ids.update(get_tweets_ids_in_prop_network(prop_graph))
        retweet_ids.update(get_node_ids_in_network_by_type(prop_graph, RETWEET_EDGE, RETWEET_NODE))
        reply_ids.update(get_node_ids_in_network_by_type(prop_graph, REPLY_EDGE, REPLY_NODE))
        user_ids.update(get_users_in_network(prop_graph))

    print("News source : {}".format(news_source))
    print("No. of tweets : {}".format(len(tweet_ids)))
    print("No. of retweet ids : {}".format(len(retweet_ids)))
    print("No. of reply ids : {}".format(len(reply_ids)))
    print("Nol. of user : {}".format(len(user_ids)))


if __name__ == "__main__":
    config = load_configuration("project.config")
    db = get_database_connection(config)

    # prop_network_stats("politifact")
    # prop_network_stats("gossipcop")

    # get_files_for_liwc_parsing()

    news_source = "politifact"
    dump_LIWC_Representation("data/baseline_data_kai/liwc/liwc_results/{}_fake_liwc_data.txt".format(news_source),
                             "data/baseline_data_kai/liwc/extracted_featuers/{}_fake_liwc_features.csv".format(news_source))

    dump_LIWC_Representation("data/baseline_data_kai/liwc/liwc_results/{}_real_liwc_data.txt".format(news_source),
                             "data/baseline_data_kai/liwc/extracted_featuers/{}_real_liwc_features.csv".format(news_source))

    # get_missing_rst_news_content()
    # get_user_aggregate_features(db, is_fake=True,
    #                             user_names=["News1Lightning", "OfeliasHeaven", "jimbradyispapa", "CraigRozniecki",
    #                                         "yojudenz",
    #                                         "GinaLawriw", "GossipCop", "GossipCopIntern", "findsugarmummy",
    #                                         "DJDavidNewsroom"])
    # dump_all_user_profile_info(db, is_fake=True, label="fake")
    # dump_all_user_profile_info(db, is_fake=False, label="real")

    exit(1)

    # get_replies_from_dataset("data/engagement_data_latest","politifact","fake","data/pre_process_data")
    # get_replies_from_dataset("data/engagement_data_latest", "politifact", "real", "data/pre_process_data")

    get_replies_from_dataset("data/engagement_data_latest", "gossipcop", "fake", "data/pre_process_data")
    get_replies_from_dataset("data/engagement_data_latest", "gossipcop", "real", "data/pre_process_data")

    # news_id_filter_date_dict = get_politifact_tweet_filter_dates(db, is_fake=True)
    #
    # print(len(news_id_filter_date_dict))
    #
    # news_id_fact_statement_date_dict, news_id_source_date_dict = get_publish_date_from_sources_politifact(db,
    #                                                                                                       is_fake=False)
    # news_id_publish_time_dict = get_news_articles_published_time(db, is_fake=False)
    #
    # # news_id_publish_time = get_news_articles_published_time(
    # #     "data/engagement_data/politifact_fake_news_dataset_format.json")
    #
    # news_id_filter_date_dict = get_dataset_publication_date(news_id_publish_time_dict, news_id_fact_statement_date_dict,
    #                                                         news_id_source_date_dict)
    #
    # print("Source news id len : {}".format(len(news_id_source_date_dict)))
    # print("Statement news id len : {}".format(len(news_id_fact_statement_date_dict)))
    # print("publish news ids len  : {}".format(len(news_id_publish_time_dict)))
    # print("News id propagation network filter date len : {}".format(len(news_id_filter_date_dict)))
    #
    # exit(1)

    # dump_social_network_user_id_single_file("data/social_network_data/gossipcop_user_ids_friends_network.txt",
    #                         "/Users/deepak/Desktop/social_network_single_files/user_ids_files" )
    #
    # dump_social_network_user_name_single_file("data/social_network_data/gossipcop_user_names_friends_network.txt",
    #                                         "/Users/deepak/Desktop/social_network_single_files/user_names_files")

    # dump_user_friends_data(db, "data/format/politifact_prop_user_names.json",
    #                        "data/social_network_data/politifact_user_names_friends_network.txt")
    #
    # dump_user_friends_data(db, "data/format/gossipcop_prop_user_names.json",
    #                        "data/social_network_data/gossipcop_user_names_friends_network.txt")

    # dump_user_id_friends_data(db, "data/format/politifact_user_id_user_name_dict.json",
    #                           "data/social_network_data/politifact_user_ids_friends_network.txt")
    #
    # dump_user_id_friends_data(db, "data/format/gossipcop_user_id_user_name_dict.json",
    #                           "data/social_network_data/gossipcop_user_ids_friends_network.txt")

    # dump_user_friends_data(db, "data/format/politifact_prop_user_names.json",
    #                        "data/social_network_data/politifact_user_names_friends_network.txt")

    # get_user_to_fetch("data/format/politifact_prop_user_names.json",
    #                   "data/format/politifact_user_id_user_name_dict.json",
    #                   db)

    # dump_friends_file_as_lines("/home/dmahudes/FakeNewsPropagation/data/politifact_real_user_friends_ids_complete.txt",
    #                     "/home/dmahudes/FakeNewsPropagation/data/format/politifact_real_user_friends_ids_complete_format.txt")

    # dump_social_network_to_db(db, "/Users/deepak/Desktop/twint_collect/data 2")
    # dump_social_network_to_db(db, "/Users/deepak/Desktop/twint_collect/data")
    # dump_social_network_to_db(db, "/Users/deepak/Desktop/twint_collect/home/ubuntu/social_network_crawl/data")
    # dump_social_network_to_db(db,
    #                           "/home/dmahudes/FakeNewsPropagation/data/network_data/home/ubuntu/social_network_crawl/data")
