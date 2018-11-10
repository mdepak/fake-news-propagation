import json
import mmap
import os
import re
import sys
import traceback

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


if __name__ == "__main__":
    config = load_configuration("project.config")
    db = get_database_connection(config)

    # dump_user_friends_data(db, "data/format/politifact_prop_user_names.json",
    #                        "data/social_network_data/politifact_user_names_friends_network.txt")
    #
    # dump_user_friends_data(db, "data/format/gossipcop_prop_user_names.json",
    #                        "data/social_network_data/gossipcop_user_names_friends_network.txt")

    dump_user_id_friends_data(db, "data/format/politifact_user_id_user_name_dict.json",
                              "data/social_network_data/politifact_user_ids_friends_network.txt")

    dump_user_id_friends_data(db, "data/format/gossipcop_user_id_user_name_dict.json",
                              "data/social_network_data/gossipcop_user_ids_friends_network.txt")

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
