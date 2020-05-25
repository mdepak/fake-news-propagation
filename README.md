# Fake News Propagation
Code for ICWSM 2020 paper "`Hierarchical Propagation Networks for Fake News Detection: Investigation and Exploitation`" (https://arxiv.org/abs/1903.09196)

### Dataset
Hierarchical propagation networks are constructed using the news samples from the FakeNewsNet dataset. 
To adhere to Twitter's privacy policy, the user information is anonymized and tweet contents are not shared.

The dataset is formatted in the `networkx` graph JSON format
 and available at `data` directory, categorized based on the news source (Politifact/GossipCop) and label.

Each node in the graph contains the random tweet id, exact epoch timestamp, node type, random user id attributes.
Additionally, the nodes in the retweet networks(macro networks) have bot score attribute and
the nodes in the reply chain network(micro-network) have the sentiment of the tweet content.

For the results mentioned in the paper, randomly down-sampled news ids provided in the `data/sample_ids` is used.


### To Run:
To use this dataset, un-zip the file `nx_network_data.zip` file in `data` directory and one can use an example in `load_dataset.py` to load the dataset.

To extract features and run the model, install the dependencies in `requirements.txt` and use basic_model.py's main function. Function `get_classificaton_results_tpnf_by_time`
in `basic_model` can be used to prune the graphs by time and work on the pruned dataset.

### References/Citation
If you use this dataset/code, please cite the following papers:

`@article{shu2019hierarchical,
  title={Hierarchical propagation networks for fake news detection: Investigation and exploitation},
  author={Shu, Kai and Mahudeswaran, Deepak and Wang, Suhang and Liu, Huan},
  journal={arXiv preprint arXiv:1903.09196},
  year={2019}
}`

`@article{shu2018fakenewsnet,
  title={FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media},
  author={Shu, Kai and  Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  journal={arXiv preprint arXiv:1809.01286},
  year={2018}
}`

(C) 2020 Arizona Board of Regents on Behalf of ASU
