# import pickle
# from os import listdir
# from os.path import isfile, join
# from pathlib import Path
#
# from pymongo import MongoClient
# from datetime import datetime
# import networkx as nx
# import numpy as np
# from random import shuffle
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
# import re
# from tqdm import tqdm
# # from sklearn.ensemble import RandomForestClassifier
# # import json
# # import random
# # from nltk.tokenize import RegexpTokenizer
# # from stop_words import get_stop_words
# # from nltk.stem.porter import PorterStemmer
# # from gensim import corpora
# # import gensim
# # from sklearn.model_selection import cross_validate
# # from sklearn.dummy import DummyClassifier
# # from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# # from sklearn.model_selection import train_test_split
# # from sklearn import preprocessing
#
# from analysis_util import get_propagation_graphs, equal_samples, get_numpy_array, create_dir
# from misc_process import get_reply_of_replies
# from pre_process_util import get_news_articles, load_configuration, get_database_connection
# from structure_temp_analysis import StructureFeatureHelper
#
# all_reply_id_sentiment_score_dict = pickle.load(open("{}/all_reply_id_sentiment_result.pkl"
#                                                      .format("data/pre_process_data/vader_sentiment"), "rb"))
#
#
# # def content_featureAgg(tweets):
# #     # Current version tweets content are almost the same, not distinguishable
# #
# #     return []
# #
# # def networkFeatureAgg(users,user_followers_coll,user_followees_coll):
# #
# #     user_index = dict()
# #     for i in range(len(users)):
# #         user_index[users[i]]=i
# #
# #     edge_list = set()
# #     for au in tqdm(users):
# #         user_name = au
# #         all_follower_tmp = list(user_followers_coll.find({'user_name': user_name}))
# #         if len(all_follower_tmp)!=0:
# #             all_followers = all_follower_tmp[0]['followers']
# #             for aft in all_followers:
# #                 if aft['screen_name'] in user_index:
# #                     edge_list.add((user_name,aft['screen_name']))
# #
# #         all_followee_tmp = list(user_followees_coll.find({'user_name':user_name}))
# #         if len(all_followee_tmp)!=0:
# #             all_followees = all_followee_tmp[0]['followees']
# #             for aft in all_followees:
# #                 if aft['screen_name'] in user_index:
# #                     edge_list.add((aft['screen_name'],user_name))
# #     G=nx.Graph()
# #     G.add_edges_from(edge_list)
# #     node_num = G.number_of_nodes()
# #     link_num = G.number_of_edges()
# #     if node_num==0:
# #         density=0
# #         cc=0
# #         avg_degree=0
# #     else:
# #         density = link_num/(float(node_num)*float(node_num))
# #         cc = nx.average_clustering(G)
# #         degrees = G.degree()
# #         avg_degree = sum(degrees.values())/len(degrees.values())
# #     return [node_num,link_num,density,cc,avg_degree]
# #
# # def getSocialEngagements(db,datasource):
# #     f_out = open('./'+datasource+'/SocialFeats.txt','w+')
# #     if datasource=='BuzzFeed':
# #         user_profiles_coll = db['TwitterUserProfile']
# #     else:
# #         user_profiles_coll = db['TwitterUserProfile2']
# #     if datasource=='BuzzFeed':
# #         user_followers_coll = db['TwitterUserFollowers']
# #     else:
# #         user_followers_coll = db['TwitterUserFollowers2']
# #     if datasource=='BuzzFeed':
# #         user_followees_coll = db['TwitterUserFollowees']
# #     else:
# #         user_followees_coll = db['TwitterUserFollowees2']
# #     news_tweets = dict()
# #     news_users = dict()
# #     # Fake News / Real News
# #
# #     if datasource=='BuzzFeed':
# #         dir_path = './Crawler/BuzzFeedCrawler/RealTwitterResult'
# #     else:
# #         dir_path = './Crawler/PolitiFact/PolitiFactTwitterResult'
# #     org_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
# #     for of in org_files:
# #         ID = of[:of.index('.json')]
# #         file_name = dir_path+'/'+of
# #         tweets = []
# #         users = []
# #         with open(file_name) as f_engagements:
# #             for line in f_engagements:
# #                 line = line.strip()
# #                 tweet_json  = json.loads(line)
# #                 tweets.append(tweet_json['text'])
# #                 users.append(tweet_json['username'])
# #         news_tweets[ID]=tweets
# #         news_users[ID]=users
# #
# #     for k, tweets in news_tweets.items():
# #         users = news_users[k]
# #         if len(users)>150:
# #             users = users[:150]
# #         user_features = userFeatureAgg(users, user_profiles_coll)
# #         content_features = content_featureAgg(tweets)
# #         network_features = networkFeatureAgg(users,user_followers_coll,user_followees_coll)
# #
# #         all_feats=[]
# #         all_feats.extend(user_features)
# #         all_feats.extend(content_features)
# #         all_feats.extend(network_features)
# #         f_out.write(k+'\t'+'\t'.join(str(f) for f in all_feats)+'\n')
# #         print k+'\t'+'\t'.join(str(f) for f in all_feats)
# #     f_out.close()
# #
# # def getSocialEngagementsEarly(db,datasource,delta):
# #     early_users = dict()
# #     with open('./'+datasource+'/Early/User_'+delta+'.txt') as f_users:
# #         for line in f_users:
# #             line = line.strip()
# #             early_users[line]=1
# #
# #     f_out = open('./'+datasource+'/Early/SocialFeatsReal'+delta+'.txt','w+')
# #     if datasource=='BuzzFeed':
# #         user_profiles_coll = db['TwitterUserProfile']
# #     else:
# #         user_profiles_coll = db['TwitterUserProfile2']
# #     if datasource=='BuzzFeed':
# #         user_followers_coll = db['TwitterUserFollowers']
# #     else:
# #         user_followers_coll = db['TwitterUserFollowers2']
# #     if datasource=='BuzzFeed':
# #         user_followees_coll = db['TwitterUserFollowees']
# #     else:
# #         user_followees_coll = db['TwitterUserFollowees2']
# #     news_tweets = dict()
# #     news_users = dict()
# #     # Fake News / Real News
# #
# #     if datasource=='BuzzFeed':
# #         dir_path = './Crawler/BuzzFeedCrawler/TwitterResult'
# #     else:
# #         dir_path = './Crawler/PolitiFact/RealTwitterResult'
# #     org_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
# #     for of in org_files:
# #         ID = of[:of.index('.json')]
# #         file_name = dir_path+'/'+of
# #         tweets = []
# #         users = []
# #         with open(file_name) as f_engagements:
# #             for line in f_engagements:
# #                 line = line.strip()
# #                 tweet_json  = json.loads(line)
# #                 if tweet_json['username'] not in early_users:
# #                     continue
# #                 tweets.append(tweet_json['text'])
# #                 users.append(tweet_json['username'])
# #         news_tweets[ID]=tweets
# #         news_users[ID]=users
# #
# #     for k, tweets in news_tweets.items():
# #         users = news_users[k]
# #         if len(users)>150:
# #             users = users[:150]
# #         user_features = userFeatureAgg(users, user_profiles_coll)
# #         content_features = content_featureAgg(tweets)
# #         network_features = networkFeatureAgg(users,user_followers_coll,user_followees_coll)
# #
# #         all_feats=[]
# #         all_feats.extend(user_features)
# #         all_feats.extend(content_features)
# #         all_feats.extend(network_features)
# #         f_out.write(k+'\t'+'\t'.join(str(f) for f in all_feats)+'\n')
# #         print k+'\t'+'\t'.join(str(f) for f in all_feats)
# #     f_out.close()
# #
# # def userFeature(user, user_profiles_coll):
# #     if list(user_profiles_coll.find({'screen_name':user})) ==[]:
# #         return [0,0,0,0]
# #     tmp = list(user_profiles_coll.find({'screen_name':user}))[0]
# #     pnum = tmp['statuses_count']
# #     fnum = tmp['friends_count']
# #     fonum = tmp['followers_count']
# #     create_time = tmp['created_at']
# #     verified = tmp['verified']
# #     if verified==False:
# #         verified=0
# #     else:
# #         verified=1
# #     date_create = datetime.strptime(create_time, '%a %b %d %H:%M:%S +0000 %Y')
# #     today = datetime.now()
# #     dregister =(today-date_create).days
# #     return [pnum,fnum,fonum,dregister,verified]
# #
# # def content_feature(tweet):
# #     topic_feature = []
# #     url_num = len([m for m in re.finditer('http', tweet)])
# #     question_flag = 0
# #     if '?' in tweet:
# #         question_flag=1
# #     mention_num = len([m for m in re.finditer('@', tweet)])
# #     retweet_count=0
# #     try:
# #         retweet_count = float(tweet.split(':::')[1])
# #     except:
# #         pass
# #
# #     return [url_num,question_flag,mention_num,retweet_count]
# #
# # def getTopicFeature(tweets, num_topic):
# #     doc_set  = []
# #     for entry in tweets:
# #         try:
# #             doc_set.append(entry.split(':::')[0])
# #         except:
# #             pass
# #
# #     texts = []
# #     tokenizer = RegexpTokenizer(r'\w+')
# #     en_stop = get_stop_words('en')
# #     p_stemmer = PorterStemmer()
# #     for i in doc_set:
# #         # clean and tokenize document string
# #         raw = i.lower()
# #         # Filter http
# #         raw = raw.replace('http','')
# #         tokens = tokenizer.tokenize(raw)
# #         # remove stop words from tokensk
# #         stopped_tokens = [i for i in tokens if not i in en_stop]
# #         # stem tokens
# #         stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
# #         # add tokens to list
# #         texts.append(stemmed_tokens)
# #
# #     dictionary = corpora.Dictionary(texts)
# #     # convert tokenized documents into a document-term matrix
# #     corpus = [dictionary.doc2bow(text) for text in texts]
# #     # generate LDA model
# #     ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topic, id2word=dictionary, passes=20)
# #
# #     topic_distribution = []
# #     for c in corpus:
# #
# #         dis = ldamodel[c]
# #         tmp_dis = [0 for i in range(num_topic)]
# #         for d in dis:
# #             tmp_dis[d[0]]=d[1]
# #         topic_distribution.append(tmp_dis)
# #     return topic_distribution
# #
# # def TweetLevelFeaturs(db):
# #     f_out = open('./'+datasource+'/TweetLevelFeatsReal.txt','w+')
# #     if datasource=='BuzzFeed':
# #         user_profiles_coll = db['TwitterUserProfile']
# #     else:
# #         user_profiles_coll = db['TwitterUserProfile1']
# #         # Fake News / Real News
# #     dir_path = './Crawler/BuzzFeedCrawler/RealTwitterResult'
# #     org_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
# #     news_tweets = dict()
# #     news_users = dict()
# #     for of in org_files:
# #         ID = of[:of.index('.json')]
# #         file_name = dir_path+'/'+of
# #         tweets = []
# #         users = []
# #         with open(file_name) as f_engagements:
# #             for line in f_engagements:
# #                 line = line.strip()
# #                 tweet_json  = json.loads(line)
# #                 tweets.append(tweet_json['text']+':::'+str(tweet_json['retweets'])+':::'+str(tweet_json['id']))
# #                 users.append(tweet_json['username'])
# #         news_tweets[ID]=tweets
# #         news_users[ID]=users
# #
# #     for k, tweets in news_tweets.items():
# #         users = news_users[k]
# #         if 'Real' in k:
# #             tw_label='1' ### Using 1 as high credibility
# #         else:
# #             tw_label='-1'
# #
# #         Topic_feats = getTopicFeature(tweets,10)
# #
# #         for i in range(len(users)):
# #             user = users[i]
# #             tweet = tweets[i]
# #             tid = tweet.split(':::')[2]
# #             user_features = userFeature(user,user_profiles_coll)
# #             content_features = content_feature(tweet)
# #             all_feats=[]
# #             all_feats.extend(user_features)
# #             all_feats.extend(content_features)
# #             all_feats.extend(Topic_feats[i])
# #             f_out.write(tid+'\t'+tw_label+'\t'+'\t'.join(str(f) for f in all_feats)+'\n')
# #             print tid+'\t'+tw_label+'\t'+'\t'.join(str(f) for f in all_feats)
# #     f_out.close()
# #
# # def Castillo11(datasource,delta):
# #     all_news = []
# #     with open('./'+datasource+'/News.txt') as f_news:
# #         for line in f_news:
# #             all_news.append(line.strip())
# #
# #     all_X = []
# #     all_y = []
# #     with open('./'+datasource+'/Early/SocialFeats'+delta+'.txt') as f_fake_social:
# #         for line in f_fake_social:
# #             line = line.strip()
# #             ID = line.split('\t')[0]
# #             if ID in all_news:
# #                 feats = [float(x) for x in line.split('\t')[1:]]
# #                 all_X.append(feats)
# #                 all_y.append(1)
# #     with open('./'+datasource+'/Early/SocialFeatsReal'+delta+'.txt') as f_real_social:
# #         for line in f_real_social:
# #             line = line.strip()
# #             ID = line.split('\t')[0]
# #             if ID in all_news:
# #                 feats = [float(x) for x in line.split('\t')[1:]]
# #                 all_X.append(feats)
# #                 all_y.append(0)
# #     X = np.array(all_X)
# #     y = np.array(all_y)
# #     arry = range(X.shape[0])
# #     shuffle(arry)
# #     X = X[arry, :]
# #     y = y[arry]
# #     clf = SVC(kernel='linear', class_weight='balanced')
# #     # clf = RandomForestClassifier()
# #     scoring = ['accuracy','precision', 'recall', 'f1']
# #     print '***'+delta+'***'
# #     res = cross_validate(estimator=clf, X=X, y=y, cv=5, verbose=0, n_jobs=-1, scoring=scoring)
# #     print '\t'.join([str(x) for x in res['test_accuracy']])
# #     # print '\t'.join([str(x) for x in res['test_precision']])
# #     # print '\t'.join([str(x) for x in res['test_recall']])
# #     print '\t'.join([str(x) for x in res['test_f1']])
# #
# #     # res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=0, n_jobs=-1, scoring='accuracy')
# #     # res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
# #     # print res
# #
# #     # res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='precision')
# #     # res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
# #     # print res
# #     #
# #     # res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='recall')
# #     # res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
# #     # print res
# #
# #     # res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=0, n_jobs=-1, scoring='f1')
# #     # res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
# #     # print res
# #
# # def Castillo11_2(datasource):
# #     all_news = []
# #     with open('./'+datasource+'/News.txt') as f_news:
# #         for line in f_news:
# #             all_news.append(line.strip())
# #
# #     all_X = []
# #     all_y = []
# #     with open('./'+datasource+'/SocialFeats.txt') as f_fake_social:
# #         for line in f_fake_social:
# #             line = line.strip()
# #             ID = line.split('\t')[0]
# #             if ID in all_news:
# #                 feats = [float(x) for x in line.split('\t')[1:]]
# #                 all_X.append(feats)
# #                 all_y.append(1)
# #     with open('./'+datasource+'/SocialFeatsReal.txt') as f_real_social:
# #         for line in f_real_social:
# #             line = line.strip()
# #             ID = line.split('\t')[0]
# #             if ID in all_news:
# #                 feats = [float(x) for x in line.split('\t')[1:]]
# #                 all_X.append(feats)
# #                 all_y.append(0)
# #     X = np.array(all_X)
# #     y = np.array(all_y)
# #     arry = range(X.shape[0])
# #     shuffle(arry)
# #     X = X[arry, :]
# #     y = y[arry]
# #
# #     # X = preprocessing.normalize(X)
# #     # clf = RandomForestClassifier()
# #
# #     train_sizes = [0.2,0.4,0.6,0.8]
# #     for ts in train_sizes:
# #         acc = []
# #         prec = []
# #         recall = []
# #         f1 = []
# #         for i in range(3):
# #             clf = SVC(kernel='linear', class_weight='balanced')
# #             X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = ts)
# #             clf.fit(X_train,y_train)
# #             y_pred = clf.predict(X_test)
# #             acc.append(accuracy_score(y_test, y_pred))
# #             prec.append(precision_score(y_test, y_pred))
# #             recall.append(recall_score(y_test, y_pred))
# #             f1.append(f1_score(y_test, y_pred))
# #
# #         print "", sum(acc)/len(acc)
# #         print "", sum(prec)/len(prec)
# #         print "", sum(recall)/len(recall)
# #         print "", sum(f1)/len(f1)
# #         print ""
# #
# # def balanced_subsample(x,y,id,subsample_size=1.0):
# #
# #     class_xs = []
# #     min_elems = None
# #
# #     for yi in np.unique(y):
# #         elems = x[(y == yi)]
# #         class_xs.append((yi, elems))
# #         if min_elems == None or elems.shape[0] < min_elems:
# #             min_elems = elems.shape[0]
# #
# #     use_elems = min_elems
# #     if subsample_size < 1:
# #         use_elems = int(min_elems*subsample_size)
# #
# #     xs = []
# #     ys = []
# #
# #     for ci,this_xs in class_xs:
# #         if len(this_xs) > use_elems:
# #             np.random.shuffle(this_xs)
# #
# #         x_ = this_xs[:use_elems]
# #         y_ = np.empty(use_elems)
# #         y_.fill(ci)
# #
# #         xs.append(x_)
# #         ys.append(y_)
# #
# #     xs = np.concatenate(xs)
# #     ys = np.concatenate(ys)
# #
# #     return xs,ys
# #
# # def TweetLevelPredict():
# #     all_X = []
# #     all_y = []
# #     all_tid = []
# #     with open('./BuzzFeed/TweetLevelFeats.txt') as f_fake_social:
# #         for line in f_fake_social:
# #             line = line.strip()
# #             tid = line.split('\t')[0]
# #             label = line.split('\t')[1]
# #             feats = [float(x) for x in line.split('\t')[2:]]
# #             all_X.append(feats)
# #             all_y.append(label)
# #             all_tid.append(tid)
# #     with open('./BuzzFeed/TweetLevelFeatsReal.txt') as f_real_social:
# #         for line in f_real_social:
# #             line = line.strip()
# #             label = line.split('\t')[1]
# #             tid = line.split('\t')[0]
# #             feats = [float(x) for x in line.split('\t')[2:]]
# #             all_X.append(feats)
# #             all_y.append(label)
# #             all_tid.append(tid)
# #     X = np.array(all_X)
# #     y = np.array(all_y)
# #     tid = np.array(all_tid)
# #     Xs,ys = balanced_subsample(X,y,0.01)
# #     arry = range(Xs.shape[0])
# #     shuffle(arry)
# #     Xs = Xs[arry, :]
# #     ys= ys[arry]
# #
# #     # clf = RandomForestClassifier(max_depth=2,random_state=0)
# #     clf = SVC(kernel='linear', class_weight='balanced',probability=True)
# #     # res = cross_val_score(estimator=clf, X=Xs, y=ys, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
# #     # res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
# #     # print res
# #     clf.fit(Xs,ys)
# #     y_predict = clf.predict(X)
# #     print 'Accuracy '
# #
# # def Dummy(datasource):
# #     all_news = []
# #     with open('./'+datasource+'/News.txt') as f_news:
# #         for line in f_news:
# #             all_news.append(line.strip())
# #
# #     all_X = []
# #     all_y = []
# #     with open('./'+datasource+'/SocialFeats.txt') as f_fake_social:
# #         for line in f_fake_social:
# #             line = line.strip()
# #             ID = line.split('\t')[0]
# #             if ID in all_news:
# #                 feats = [float(x) for x in line.split('\t')[1:]]
# #                 all_X.append(feats)
# #                 all_y.append(1)
# #     with open('./'+datasource+'/SocialFeatsReal.txt') as f_real_social:
# #         for line in f_real_social:
# #             line = line.strip()
# #             ID = line.split('\t')[0]
# #             if ID in all_news:
# #                 feats = [float(x) for x in line.split('\t')[1:]]
# #                 all_X.append(feats)
# #                 all_y.append(0)
# #     X = np.array(all_X)
# #     y = np.array(all_y)
# #     arry = range(X.shape[0])
# #     shuffle(arry)
# #     X = X[arry, :]
# #     y = y[arry]
# #     clf = DummyClassifier(constant=1)
# #     scoring = ['accuracy','precision', 'recall', 'f1']
# #     res = cross_validate(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring=scoring)
# #
# #
# #     print '\t'.join([str(x) for x in res['test_accuracy']])
# #     print '\t'.join([str(x) for x in res['test_precision']])
# #     print '\t'.join([str(x) for x in res['test_recall']])
# #     print '\t'.join([str(x) for x in res['test_f1']])
# #
# #     res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')
# #     res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
# #     print res
# #
# #     res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='precision')
# #     res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
# #     print res
# #
# #     res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='recall')
# #     res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
# #     print res
# #
# #     res = cross_val_score(estimator=clf, X=X, y=y, cv=5, verbose=1, n_jobs=-1, scoring='f1')
# #     res = "%0.3f +/- %0.3f" % ( np.mean(res), np.std(res))
# #     print res
#
#
# def get_message_based_features(reply_id_content_dict):
#     num_words = []
#     num_urls = []
#     question_mark_nums = []
#     num_mentions = []
#
#     for reply_id, content in reply_id_content_dict.items():
#         url_num = len([m for m in re.finditer('http', content)])
#         question_flag = 0
#         if '?' in content:
#             question_flag = 1
#         mention_num = len([m for m in re.finditer('@', content)])
#         num_word = len(content.split())
#
#         num_words.append(num_word)
#         num_urls.append(url_num)
#         question_mark_nums.append(question_flag)
#         num_mentions.append(mention_num)
#
#     try:
#         mean_num_words = np.mean(num_words)
#     except:
#         mean_num_words = 0
#
#     try:
#         mean_num_urls = np.mean(num_urls)
#     except:
#         mean_num_urls = 0
#
#     try:
#         mean_question_mark_nums = np.mean(question_mark_nums)
#     except:
#         mean_question_mark_nums = 0
#
#     try:
#         mean_num_mentions = np.mean(num_mentions)
#     except:
#         mean_num_mentions = 0
#
#     return [mean_num_words, mean_num_urls, mean_question_mark_nums, mean_num_mentions]
#
#
# def get_content_topic_based_features(reply_id_content_dict):
#     positive_words = []
#     negative_words = []
#     neutral_words = []
#     sentiment_scores = []
#     reply_lengths = []
#
#     for reply_id, content in reply_id_content_dict.items():
#         if reply_id in reply_id_content_dict:
#             sentiment_info = all_reply_id_sentiment_score_dict[reply_id]
#             positive_words.append(sentiment_info["pos"])
#             negative_words.append(sentiment_info["neg"])
#             neutral_words.append(sentiment_info["neu"])
#             sentiment_scores.append(sentiment_info["compound"])
#             reply_lengths.append(len(content))
#
#     try:
#         mean_positive_words = np.mean(positive_words)
#     except:
#         mean_positive_words = 0
#
#     try:
#         mean_negative_words = np.mean(negative_words)
#     except:
#         mean_negative_words = 0
#
#     try:
#         mean_neutral_words = np.mean(neutral_words)
#     except:
#         mean_neutral_words = 0
#
#     try:
#         mean_sentiment_score = np.mean(sentiment_scores)
#     except:
#         mean_sentiment_score = 0
#
#     try:
#         mean_reply_length = np.mean(reply_lengths)
#     except:
#         mean_reply_length = 0
#
#     return [len(reply_id_content_dict), mean_positive_words, mean_negative_words, mean_neutral_words,
#             mean_sentiment_score, mean_reply_length]
#
#
# def get_user_aggregate_features(db, is_fake, user_ids):
#     posts_num = []
#     friends_num = []
#     followers_num = []
#     days_register = []
#
#     if is_fake:
#         label_user_collection = db.fake_twitter_user_profile
#     else:
#         label_user_collection = db.real_twitter_user_profile
#
#     user_profile_collection = db.twitter_user_profile
#
#     # np.random.shuffle(user_ids)
#
#     for user_id in tqdm(user_ids):
#
#         user_object = label_user_collection.find_one({"user_id": user_id}, {"profile_info.statuses_count": 1,
#                                                                             "profile_info.friends_count": 1,
#                                                                             "profile_info.followers_count": 1,
#                                                                             "profile_info.created_at": 1})
#         if user_object is None:
#             user_object = user_profile_collection.find_one({"user_id": user_id}, {"profile_info.statuses_count": 1,
#                                                                                   "profile_info.friends_count": 1,
#                                                                                   "profile_info.followers_count": 1,
#                                                                                   "profile_info.created_at": 1})
#
#         if user_object is None:
#             print('user {} not found'.format(user_id))
#         else:
#             if "profile_info" in user_object:
#                 pnum = user_object["profile_info"]['statuses_count']
#                 fnum = user_object["profile_info"]['friends_count']
#                 fonum = user_object["profile_info"]['followers_count']
#                 create_time = user_object["profile_info"]['created_at']
#                 date_create = datetime.strptime(create_time, '%a %b %d %H:%M:%S +0000 %Y')
#                 today = datetime.now()
#                 dregister = (today - date_create).days
#                 posts_num.append(pnum)
#                 friends_num.append(fnum)
#                 followers_num.append(fonum)
#                 days_register.append(dregister)
#
#     try:
#         avg_posts_num = sum(posts_num) / len(posts_num)
#     except:
#         avg_posts_num = 0
#     try:
#         avg_friends_num = sum(friends_num) / len(friends_num)
#     except:
#         avg_friends_num = 0
#     try:
#         avg_followers_num = sum(followers_num) / len(followers_num)
#     except:
#         avg_followers_num = 0
#     try:
#         avg_days_register = sum(days_register) / len(days_register)
#     except:
#         avg_days_register = 0
#
#     return [avg_posts_num, avg_friends_num, avg_followers_num, avg_days_register]
#
#
# def get_castillo_features(db, news_source, raw_data_dir, label, prop_graphs):
#     raw_data = pickle.load(open("{}/{}_{}_castillo_raw_data.pkl".format(raw_data_dir, news_source, label), "rb"))
#
#     all_features = []
#
#     for news in raw_data:
#         sample_feature = []
#         sample_feature.extend(get_user_aggregate_features(db, label == "fake", news["user_ids"]))
#         sample_feature.extend(get_message_based_features(news["reply_id_content_dict"]))
#         sample_feature.extend(get_content_topic_based_features(news["reply_id_content_dict"]))
#
#         all_features.append(sample_feature)
#
#     structure_feature_helper = StructureFeatureHelper()
#     structure_features = structure_feature_helper.get_features_array(prop_graphs, micro_features=False,
#                                                                      macro_features=True)
#
#     other_features = get_numpy_array(all_features)
#     structure_features = get_numpy_array(structure_features)[:, [0, 1, 2]]
#     print("Other features shape")
#     print(other_features.shape)
#
#     print("Structure features shape")
#     print(structure_features.shape)
#     return np.concatenate([other_features, structure_features], axis=1)
#
#
# def dump_castillo_features(db, news_source, raw_data_dir, feature_out_dir, prop_graphs_dir):
#     fake_prop_graph, real_prop_graph = get_propagation_graphs(prop_graphs_dir, news_source)
#     fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)
#
#     create_dir(feature_out_dir)
#
#     fake_castillo_features = get_castillo_features(db, news_source, raw_data_dir, "fake", fake_prop_graph)
#     real_castillo_features = get_castillo_features(db, news_source, raw_data_dir, "real", real_prop_graph)
#
#     all_castillo_features = np.concatenate([fake_castillo_features, real_castillo_features])
#
#     print("All castillo features")
#     print(all_castillo_features.shape, flush=True)
#
#     pickle.dump(all_castillo_features, open("{}/{}_castillo_features.pkl".format(feature_out_dir, news_source), "wb"))
#
#
# def get_raw_feature_for_news(news):
#     data = {}
#
#     user_ids = set()
#
#     reply_id_content_dict = dict()
#
#     for tweet in news["tweets"]:
#         user_ids.add(tweet["user_id"])
#         get_reply_of_replies(tweet["reply"], reply_id_content_dict)
#
#     data["id"] = news["id"]
#     data["user_ids"] = list(user_ids)
#     data["reply_id_content_dict"] = reply_id_content_dict
#
#     return data
#
#
# def get_castillo_data(data_dir, prop_graphs, news_source, label):
#     prop_graphs_ids = []
#     for news_graph in prop_graphs:
#         prop_graphs_ids.append(news_graph.tweet_id)
#
#     castillo_raw_data = [None] * len(prop_graphs_ids)
#
#     prop_graphs_ids_set = set(prop_graphs_ids)
#
#     file_path = "{}/{}_{}_news_complete_dataset.json".format(data_dir, news_source, label)
#
#     for news in get_news_articles(file_path):
#         news_id = news["id"]
#         if news_id in prop_graphs_ids_set:
#             news_id_index = prop_graphs_ids.index(news_id)
#             castillo_raw_data[news_id_index] = get_raw_feature_for_news(news)
#
#     return castillo_raw_data
#
#
# def get_castillo_raw_data(data_dir, prop_graphs_dir, out_dir, news_source):
#     fake_prop_graph, real_prop_graph = get_propagation_graphs(prop_graphs_dir, news_source)
#
#     fake_prop_graph, real_prop_graph = equal_samples(fake_prop_graph, real_prop_graph)
#
#     create_dir(out_dir)
#
#     fake_castillo_raw_data = get_castillo_data(data_dir, fake_prop_graph, news_source, "fake")
#     real_castillo_raw_data = get_castillo_data(data_dir, real_prop_graph, news_source, "real")
#
#     pickle.dump(fake_castillo_raw_data, open("{}/{}_fake_castillo_raw_data.pkl".format(out_dir, news_source), "wb"))
#     pickle.dump(real_castillo_raw_data, open("{}/{}_real_castillo_raw_data.pkl".format(out_dir, news_source), "wb"))
#
#
# def get_castillo_feature_array(news_source, castillo_feature_dir):
#     file_path = "{}/{}_real_castillo_raw_data.pkl".format(castillo_feature_dir, news_source)
#     file_obj = Path(file_path)
#
#     if file_obj.exists():
#         return pickle.load(open(file_path, "wb"))
#
#     return None
#
#
#
# if __name__ == '__main__':
#     config = load_configuration("project.config")
#     db = get_database_connection(config)
#     news_source = "politifact"
#
#     for news_source in ["politifact", "gossipcop"]:
#         # get_castillo_raw_data("data/engagement_data_latest", "data/saved_new_no_filter", "data/castillo/raw_data",
#         #                       news_source)
#         #
#         # print("Raw data dumped", flush=True)
#
#         dump_castillo_features(db, news_source, "data/castillo/raw_data", "data/castillo/saved_features",
#                                "data/saved_new_no_filter")
#
#         print("Castillo features for {} dumped".format(news_source), flush=True)
