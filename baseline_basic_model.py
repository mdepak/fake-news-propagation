import pickle

import numpy as np

from analysis_util import get_propagation_graphs, equal_samples
from basic_model import get_basic_model_results
from construct_sample_features import get_train_test_split, get_TPNF_dataset, get_dataset_feature_names
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

    sample_features = np.concatenate([fake_features, real_features])

    pickle.dump(sample_features,  open("data/stfn/{}_stfn_features.pkl".format(news_source), "wb"))
    return sample_features


def get_castillo_features(news_source, castillo_feature_folder="data/castillo/saved_features"):
    features = pickle.load(open("{}/{}_castillo_features.pkl".format(castillo_feature_folder, news_source), "rb"))
    features = np.nan_to_num(features)
    return features


def get_tpnf_features(news_source, feature_folder="data/train_test_data"):
    return pickle.load(open("{}/{}_micro_macro_struct_temp_linguistic.pkl".format(feature_folder, news_source), "rb"))


def get_liwc_features(news_source, feature_folder="data/baseline_features/liwc_features"):
    fake_features = np.loadtxt("{}/{}_fake_liwc.csv".format(feature_folder, news_source), delimiter=',')
    real_features = np.loadtxt("{}/{}_real_liwc.csv".format(feature_folder, news_source), delimiter=',')
    feature_array = np.concatenate([fake_features, real_features])

    return feature_array


def get_rst_features(news_source, rst_feature_folder="data/baseline_features/rst_both/raw_parsed_data"):
    fake_features = np.loadtxt("{}/{}_fake_rst_features.csv".format(rst_feature_folder, news_source), delimiter='\t')
    real_features = np.loadtxt("{}/{}_real_rst_features.csv".format(rst_feature_folder, news_source), delimiter='\t')
    feature_array = np.concatenate([fake_features, real_features])

    return feature_array


def get_sample_feature_array(news_source, tpnf=False, castillo=False, liwc=False, rst=False, stfn=False):
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


def get_baselines_classificaton_result(news_source, tpnf=False, castillo=False, liwc=False, rst=False, stfn=False):
    sample_feature_array = get_sample_feature_array(news_source, tpnf, castillo, liwc, rst, stfn)

    print("Sample feature array dimensions")
    print(sample_feature_array.shape, flush=True)

    num_samples = int(len(sample_feature_array) / 2)
    target_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)

    X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)
    get_basic_model_results(X_train, X_test, y_train, y_test)


def get_baseline_modification_classificaton_result(news_source, data_dir = "data/train_test_data"):
    include_micro = True
    include_macro = True

    include_structural = True
    include_temporal = True
    include_linguistic = True

    science_features = get_science_dataset_array(news_source)
    science_features = science_features[:, [3,4]]
    print("stfn features :", science_features.shape)

    sample_feature_array = get_TPNF_dataset(data_dir, news_source, include_micro, include_macro, include_structural,
                                            include_temporal, include_linguistic, use_cache=True)

    sample_feature_array = sample_feature_array[:, :-1]
    feature_names, short_feature_names = get_dataset_feature_names(include_micro, include_macro, include_structural,
                                                                   include_temporal, include_linguistic)

    print("tpnf features :", sample_feature_array.shape)

    sample_feature_array = np.concatenate([sample_feature_array, science_features], axis=1)

    print("overall features dim  : ", sample_feature_array.shape)

    print("Sample feature array dimensions")
    print(sample_feature_array.shape, flush=True)

    num_samples = int(len(sample_feature_array) / 2)
    target_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)

    X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)
    get_basic_model_results(X_train, X_test, y_train, y_test)


def get_domain_adaptation_classification_results(source_news_source, target_news_source, tpnf=False, castillo=False,
                                                 liwc=False, rst=False, stfn=False):
    train_sample_feature_array = get_sample_feature_array(source_news_source, tpnf, castillo, liwc, rst, stfn)
    test_sample_feature_array = get_sample_feature_array(target_news_source, tpnf, castillo, liwc, rst, stfn)

    print("Source Domain : {}".format(source_news_source))
    print("Target Domain : {}".format(target_news_source))

    print("source :  ", train_sample_feature_array.shape)
    print("target :  ", test_sample_feature_array.shape)

    train_num_samples = int(len(train_sample_feature_array) / 2)
    test_num_samples = int(len(test_sample_feature_array) / 2)

    train_target_labels = np.concatenate([np.ones(train_num_samples), np.zeros(train_num_samples)], axis=0)
    test_target_labels = np.concatenate([np.ones(test_num_samples), np.zeros(test_num_samples)], axis=0)

    S_X_train, S_X_test, S_y_train, S_y_test = get_train_test_split(train_sample_feature_array, train_target_labels)
    T_X_train, T_X_test, T_y_train, T_y_test = get_train_test_split(test_sample_feature_array, test_target_labels)

    # get_basic_model_results(train_sample_feature_array, test_sample_feature_array, train_target_labels,
    # test_target_labels)

    get_basic_model_results(S_X_train, T_X_test, S_y_train, T_y_test)


if __name__ == "__main__":
    get_baselines_classificaton_result("gossipcop", tpnf=True, castillo=False, liwc=False, rst=False, stfn=False)

    # get_baselines_classificaton_result("gossipcop", tpnf=False, castillo=False, liwc=False, rst=False, stfn=True)

    # get_baseline_modification_classificaton_result("gossipcop")

    # get_domain_adaptation_classification_results("gossipcop", "politifact", tpnf=True, stfn=False, liwc=False, rst=False)

    # feature_array = get_castillo_features("politifact")
    # num_samples = int(feature_array.shape[0]/2)
    # np.savetxt("fake_castillo_features.csv", feature_array[:num_samples], delimiter=",")
    # np.savetxt("real_castillo_features.csv", feature _array[num_samples+1:], delimiter=",")
    #
    # dump_random_forest_feature_importance(feature_array)
