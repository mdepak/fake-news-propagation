import json
import mmap
import os
import re
import string
import sys
import traceback
from datetime import datetime

import datefinder
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from pymongo import UpdateOne
from tqdm import tqdm
import newspaper

from pre_process_util import load_configuration, get_database_connection


def get_web_archieve_results(search_url):
    archieve_url = "http://web.archive.org/cdx/search/cdx?url={}&output=json".format(search_url)

    response = requests.get(archieve_url)
    response_json = json.loads(response.data)

    response_json = response_json[1:]

    return response_json


def get_website_from_arhieve():
    pass


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

        news_id_int = int(news_id.replace("politifact",""))

        news = news_collection.find_one({"id":news_id_int})

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


if __name__ == "__main__":
    config = load_configuration("project.config")
    db = get_database_connection(config)

    news_id_filter_date_dict = get_politifact_tweet_filter_dates(db, is_fake=True)

    print(len(news_id_filter_date_dict))

    news_id_fact_statement_date_dict, news_id_source_date_dict = get_publish_date_from_sources_politifact(db, is_fake=False)
    news_id_publish_time_dict = get_news_articles_published_time(db, is_fake=False)

    # news_id_publish_time = get_news_articles_published_time(
    #     "data/engagement_data/politifact_fake_news_dataset_format.json")

    news_id_filter_date_dict = get_dataset_publication_date(news_id_publish_time_dict, news_id_fact_statement_date_dict,
                                                            news_id_source_date_dict)

    print("Source news id len : {}".format(len(news_id_source_date_dict)))
    print("Statement news id len : {}".format(len(news_id_fact_statement_date_dict)))
    print("publish news ids len  : {}".format(len(news_id_publish_time_dict)))
    print("News id propagation network filter date len : {}".format(len(news_id_filter_date_dict)))

    exit(1)

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
