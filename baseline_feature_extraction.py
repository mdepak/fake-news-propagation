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
    for news_id, text_content in news_id_text_content_dict.items():
        if len(text_content) <5:
            count+=1

    print("News ids without text content : {}".format(count))


if __name__ == "__main__":
    # get_news_ids_used_for_propagation_network("politifact")

    config = load_configuration("project.config")
    db = get_database_connection(config)

    fake_news_text_contents, real_news_text_contents = get_news_article_text_content_used_for_propagation_network(db,
                                                                                                                  "data/saved_new_no_filter",
                                                                                                                  "politifact")
    # count_news_articles_without_text_content(fake_news_text_contents)
    # count_news_articles_without_text_content(real_news_text_contents)


    dump_text_content("data/baseline_data", "politifact_fake", fake_news_text_contents)
    dump_text_content("data/baseline_data", "politifact_real", real_news_text_contents)
