import datetime
import time


class tweet_node:

    def __init__(self, tweet_id, text, created_time, user_name, user_id, news_id, node_type):
        self.tweet_id = tweet_id
        self.text = text
        self.created_time = created_time
        self.user_name = user_name
        self.user_id = user_id

        self.news_id = news_id
        self.no_likes = None
        self.no_retweets = None
        self.no_replies = None

        self.retweet_children = []

        self.reply_children = []

        self.children = set()

        self.sentiment = None
        self.stance = None
        self.topic_vector = None

        self.original_object = None

        self.parent_node = None

        self.node_type = node_type

    def __eq__(self, other):
        return self.tweet_id == other.tweet_id

    def __hash__(self):
        return hash(self.tweet_id)

    def set_node_type(self, node_type):
        self.node_type = node_type

    def set_parent_node(self, parent_node):
        self.parent_node = parent_node

    def add_retweet_child(self, child_node):
        self.retweet_children.append(child_node)
        self.children.add(child_node)

    def add_reply_child(self, child_node):
        self.reply_children.append(child_node)
        self.children.add(child_node)

    def get_contents(self):
        return {"tweet_id": str(self.tweet_id),
                "text": self.text,
                "created_time": self.created_time,
                "user_name": self.user_name,
                "user_id": self.user_id,
                "news_id": self.news_id
                }


def twitter_datetime_str_to_object(date_str):
    time_struct = time.strptime(date_str, "%a %b %d %H:%M:%S +0000 %Y")
    date = datetime.datetime.fromtimestamp(time.mktime(time_struct))
    return int(date.strftime('%s'))
