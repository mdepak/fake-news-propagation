from analysis_util import get_propagation_graphs, create_dir


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


def get_news_article_text_content_used_for_propagation_network(news_source):
    fake_prop_graph, real_prop_graph = get_propagation_graphs(news_source)
    count = 0

    fake_news_text_contents = dict()
    for graph in fake_prop_graph:
        fake_news_text_contents[graph.tweet_id] = graph.text
        if len(graph.text) < 3:
            print(graph.tweet_id)
            count += 1

    print("Fake graphs without text : {}".format(count))

    count = 0
    real_news_text_contents = dict()
    for graph in real_prop_graph:
        real_news_text_contents[graph.tweet_id] = graph.text
        if len(graph.text) < 3:
            print(graph.tweet_id)
            count += 1

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


if __name__ == "__main__":
    # get_news_ids_used_for_propagation_network("politifact")

    fake_news_text_contents, real_news_text_contents = get_news_article_text_content_used_for_propagation_network(
        "politifact")
    dump_text_content("baseline_data", "politifact_fake", fake_news_text_contents)
    dump_text_content("baseline_data", "politifact_real", real_news_text_contents)
