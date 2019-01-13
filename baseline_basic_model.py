import numpy as np
import pickle

from analysis_util import get_propagation_graphs, equal_samples
from basic_model import get_basic_model_results, dump_random_forest_feature_importance
from construct_sample_features import get_train_test_split
from structure_temp_analysis import ScienceCascadeFeatureHelper


def get_science_dataset_array(news_source):
    fake_prop_graph, real_prop_graph = get_propagation_graphs("data/saved_new_no_filter", news_source)
    fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)
    feature_helper = ScienceCascadeFeatureHelper()
    include_micro = False
    include_macro = True

    fake_features = feature_helper.get_features_array(fake_prop_graph, micro_features=include_micro,
                                                      macro_features=include_macro, news_source=news_source,
                                                      label="fake")
    real_features = feature_helper.get_features_array(real_prop_graph, micro_features=include_micro,
                                                      macro_features=include_macro, news_source=news_source,
                                                      label="real")

    return np.concatenate([fake_features, real_features])



def get_castillo_features(news_source, castillo_feature_folder="data/castillo/saved_features"):
    features =  pickle.load(open("{}/{}_castillo_features.pkl".format(castillo_feature_folder, news_source), "rb"))
    features = np.nan_to_num(features)
    return features


def get_tpnf_features(news_source, feature_folder="data/train_test_data"):
    return pickle.load(open("{}/{}_micro_macro_struct_temp_linguistic.pkl".format(feature_folder, news_source), "rb"))


def get_liwc_features(news_source, feature_folder = "data/baseline_features/liwc_features"):
    fake_features = np.loadtxt("{}/{}_fake_liwc.csv".format(feature_folder, news_source), delimiter=',')
    real_features = np.loadtxt("{}/{}_real_liwc.csv".format(feature_folder, news_source), delimiter=',')
    feature_array = np.concatenate([fake_features, real_features])

    return feature_array

def get_rst_features(news_source, rst_feature_folder = "data/baseline_features/rst/raw_parsed_data"):
    fake_features = np.loadtxt("{}/{}_fake_rst_features.csv".format(rst_feature_folder, news_source), delimiter='\t')
    real_features = np.loadtxt("{}/{}_real_rst_features.csv".format(rst_feature_folder, news_source), delimiter='\t')
    feature_array = np.concatenate([fake_features, real_features])

    return feature_array


def get_sample_feature_array(news_source, tpnf=False, castillo=False, liwc=False, rst=False, stfn  = False):
    feature_arrays = []

    if tpnf:
        feature_arrays.append(get_tpnf_features(news_source))

    if castillo:
        feature_arrays.append(get_castillo_features(news_source))

    if liwc:
        feature_arrays.append(get_liwc_features(news_source))

    if rst:
        feature_arrays.append(get_rst_features(news_source))

    if stfn:
        feature_arrays.append(get_science_dataset_array(news_source))

    all_feature_array = np.concatenate(feature_arrays, axis=1)

    print("Baseline feature array")
    print(all_feature_array.shape, flush=True)

    return all_feature_array


def get_baselines_classificaton_result(news_source, tpnf=False, castillo=False, liwc=False, rst=False, stfn = False):
    sample_feature_array = get_sample_feature_array(news_source, tpnf, castillo, liwc, rst , stfn)

    print("Sample feature array dimensions")
    print(sample_feature_array.shape, flush=True)

    num_samples = int(len(sample_feature_array) / 2)
    target_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)

    X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)
    get_basic_model_results(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    get_baselines_classificaton_result("gossipcop", tpnf=True, castillo=False, liwc=False, rst=False, stfn=True)

    # feature_array = get_castillo_features("politifact")
    # num_samples = int(feature_array.shape[0]/2)
    # np.savetxt("fake_castillo_features.csv", feature_array[:num_samples], delimiter=",")
    # np.savetxt("real_castillo_features.csv", feature_array[num_samples+1:], delimiter=",")
    #
    # dump_random_forest_feature_importance(feature_array)