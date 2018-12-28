import pickle

import numpy as np

from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
from nltk import TweetTokenizer
from torch.autograd import Variable


def get_batches(batch_size, params):
    total_len = len(params)
    for batch_i in range(int(np.ceil(total_len / batch_size))):
        start_i = batch_i * batch_size

        yield params[start_i:start_i + batch_size]


def get_elmo_sentence_embeddings(documents):
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 1, dropout=0)

    # use batch_to_ids to convert sentences to character ids
    # sentences = [['First', 'sentence', '.'], ['Another', '.']]

    batches_lat_embeddings = []

    device = torch.device('cuda')

    batch_size = 128

    elmo = elmo.to(device)

    for batch_idx, doc_batch in enumerate(get_batches(batch_size, documents)):
        character_ids = batch_to_ids(doc_batch)
        character_ids = character_ids.to(device)
        #

        embeddings = elmo(character_ids)

        # sentence_embeddings = torch.sum(embeddings['elmo_representations'][0], dim=1)
        layer_1_rep = get_weights_from_layers(masks=embeddings["mask"], elmo_rep=embeddings['elmo_representations'][0])

        batches_lat_embeddings.append(layer_1_rep)

        print("batch idx : {} completed...".format(batch_idx), flush=True)

    return np.concatenate(batches_lat_embeddings, axis=0)

    # layer_2_rep = get_weights_from_layers(masks = embeddings["mask"],elmo_rep = embeddings['elmo_representations'][1])
    #
    # return np.concatenate([layer_1_rep, layer_2_rep], axis=1)


def get_weights_from_layers(masks, elmo_rep):
    batch_size = masks.shape[0]
    max_seq_len = masks.shape[1]

    # masks = masks.unsqueeze(1)

    elmo_rep = elmo_rep.view(elmo_rep.shape[0] * elmo_rep.shape[1], 1024)

    # mask_weighted_rep = torch.matmul(masks, elmo_rep)
    masks = masks.view(masks.shape[0] * masks.shape[1])

    masks = masks.view(-1, 1).repeat(1, 1024)
    # mask_weighted_rep = torch.matmul(masks.float(), elmo_rep)

    mask_weighted_rep = masks.float() * elmo_rep
    mask_weighted_rep = mask_weighted_rep.view(batch_size, max_seq_len, 1024)

    sentence_embeddings = torch.sum(mask_weighted_rep, dim=1)

    return Variable(sentence_embeddings).data.cpu().numpy()


def dump_elmo_features(data_dir, news_source, label, out_dir):
    reply_id_content_dict = pickle.load(
        open("{}/{}_{}_reply_id_content_dict.pkl".format(data_dir, news_source, label), "rb"))

    reply_contents = []

    reply_arr_idx_dict = dict()

    idx = 0

    tokenizer = TweetTokenizer(strip_handles=True)
    for reply_id, content in reply_id_content_dict.items():
        reply_arr_idx_dict[reply_id] = idx
        reply_contents.append(tokenizer.tokenize(content))
        idx += 1

    sentence_lat_embeddings = get_elmo_sentence_embeddings(reply_contents)

    pickle.dump(sentence_lat_embeddings,
                open("{}/{}_{}_elmo_lat_embeddings.pkl".format(out_dir, news_source, label), "wb"))
    pickle.dump(reply_arr_idx_dict,
                open("{}/{}_{}_reply_id_latent_mat_index.pkl".format(out_dir, news_source, label), "wb"))


if __name__ == "__main__":
    # sentences = [['First', 'sentence', '.'], []]
    # sentence_lat_embeddings = get_elmo_sentence_embeddings(sentences)

    print("============  Dumping fake data ============")
    dump_elmo_features("data/pre_process_data", "politifact", "fake", "data/pre_process_data/elmo_features")

    print("============  Dumping real data ============")
    dump_elmo_features("data/pre_process_data", "politifact", "real", "data/pre_process_data/elmo_features")
