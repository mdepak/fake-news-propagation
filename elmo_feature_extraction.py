import numpy as np

from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
from torch.autograd import Variable


def dump_elmo_embeddings(documents):
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 1, dropout=0)

    # use batch_to_ids to convert sentences to character ids
    # sentences = [['First', 'sentence', '.'], ['Another', '.']]
    character_ids = batch_to_ids(documents)

    embeddings = elmo(character_ids)

    # sentence_embeddings = torch.sum(embeddings['elmo_representations'][0], dim=1)
    layer_1_rep = get_weights_from_layers(masks=embeddings["mask"], elmo_rep=embeddings['elmo_representations'][0])

    return layer_1_rep

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


if __name__ == "__main__":
    sentences = [['First', 'sentence', '.'], ['Another', '.']]
    sentence_lat_embeddings = dump_elmo_embeddings(sentences)
