import json
import mmap
import os
import re
import sys
import traceback

import pandas as pd
from pymongo import UpdateOne

from preprocess_data import load_configuration, get_database_connection


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


def get_friends_names(friends_file):
    with open(friends_file, encoding="UTF-8") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines[1:]


if __name__ == "__main__":
    config = load_configuration("project.config")
    db = get_database_connection(config)
    get_user_to_fetch("data/format/politifact_prop_user_names.json", "data/format/politifact_user_id_user_name_dict.json",
                      db)

    # dump_friends_file_as_lines("/home/dmahudes/FakeNewsPropagation/data/politifact_real_user_friends_ids_complete.txt",
    #                     "/home/dmahudes/FakeNewsPropagation/data/format/politifact_real_user_friends_ids_complete_format.txt")

    # dump_social_network_to_db(db, "/Users/deepak/Desktop/twint_collect/data 2")
    # dump_social_network_to_db(db, "/Users/deepak/Desktop/twint_collect/data")
    # dump_social_network_to_db(db, "/Users/deepak/Desktop/twint_collect/home/ubuntu/social_network_crawl/data")
    # dump_social_network_to_db(db,
    #                           "/home/dmahudes/FakeNewsPropagation/data/network_data/home/ubuntu/social_network_crawl/data")
