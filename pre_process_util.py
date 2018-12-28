import configparser
import json

from pymongo import MongoClient


def load_configuration(config_file):
    """Gets the configuration file and returns the dictionary of configuration"""
    filename = config_file
    config = configparser.ConfigParser()
    config.read(filename)

    return config


def get_database_connection(config):
    host = config['MongoDB']['host']
    port = int(config['MongoDB']['port'])
    db_name = config['MongoDB']['database_name']

    client = MongoClient(host, port)
    db = client[db_name]
    return db


def get_news_articles(data_file):
    with open(data_file) as file:
        for line in file:
            yield json.loads(line)
