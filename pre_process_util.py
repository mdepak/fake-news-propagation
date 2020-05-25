import json


def get_news_articles(data_file):
    with open(data_file) as file:
        for line in file:
            yield json.loads(line)
