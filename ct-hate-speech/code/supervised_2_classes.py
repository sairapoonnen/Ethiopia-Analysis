import logging
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from collections import Counter
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
import pickle
import os
import random
import re
import json
import sys, getopt
warnings.simplefilter(action='ignore', category=FutureWarning)

def transform_to_dataset(tagged_articles):
    """
    Create X and y aray with article text and article category
    :param tagged_sentences: list of list of tuples (term_i, tag_i)
    :return: 
    """
    X, y = [], [] 
    for article in tagged_articles:
        X.append(article["text"])
        y.append(article["category"])
    return X, y


def naive_bayes_clf(X_train, X_test, y_train, alpha, use_grid_search = False, grid_values = {}):
    if not use_grid_search:
        naive_clf = naive_bayes.MultinomialNB(alpha=alpha)
        naive_clf.fit(X_train,y_train)# predict the labels on validation dataset
        predictions_NB = naive_clf.predict(X_test)# Use accuracy_score function to get the accuracy
        save_model("naive_bayes_2_classes",naive_clf)
        metrics = print_metrics("Naive Bayes Classifier", y_test, predictions_NB)
    else:
        if not grid_values:
            print("Provide grid inputs")
            return
        else:
            naive_clf = naive_bayes.MultinomialNB()
            grid_clf = GridSearchCV(naive_clf, param_grid=grid_values)
            grid_clf.fit(X_train, y_train)
            predictions_NB = grid_clf.predict(X_test)
            metrics = print_metrics("Naive Bayes Classifier (Grid search)", y_test, predictions_NB)
            logging.info("Grid NB Best Parameters: {}".format(grid_clf.best_params_))
            logging.info("\n"+ pd.DataFrame.from_dict(grid_clf.cv_results_)[["params", "mean_test_score"]].to_csv(sep=' ', index=False))
            save_model("naive_bayes_2_class_250", grid_clf)
    return metrics

def svm_clf(X_train, X_test, y_train, C, gamma, use_grid_search = False, grid_values = {}):
    if not use_grid_search:
        svm_clf = svm.SVC(C=C, gamma=gamma)
        svm_clf.fit(X_train, y_train)
        predictions_SVM = svm_clf.predict(X_test)# Use accuracy_score function to get the accuracy
        save_model("svm_2_classes", svm_clf)
        metrics = print_metrics("SVM Classifier", y_test, predictions_SVM)
    
    else:
        if not grid_values:
            print("Provide grid inputs")
            return
        else:
            svm_clf = svm.SVC()
            grid_clf = GridSearchCV(svm_clf, param_grid=grid_values)
            grid_clf.fit(X_train, y_train)
            predictions_SVM = grid_clf.predict(X_test)
            metrics = print_metrics("SVM Classifier (Grid search)", y_test, predictions_SVM)
            logging.info("Grid SVM Best Parameters: {}".format(grid_clf.best_params_))
            logging.info("\n"+ pd.DataFrame.from_dict(grid_clf.cv_results_)[["params", "mean_test_score"]].to_csv(sep=' ', index=False))
    return metrics


def xg_boost_clf(X_train, X_test, y_train, y_test, params):
    num_rounds = 20
    xgb_model = xgb.XGBClassifier(
            learning_rate=params['eta'],
                            max_depth = params['max_depth'],
                            n_estimators =params['n_estimators'],
                              scale_pos_weight=params['scale_pos_weight'],
                              )
    xgb_model.fit(X_train, y_train)

    xgb_predict=xgb_model.predict(X_test)
    metrics = print_metrics("XG Boost Classifier", y_test, xgb_predict)
    return metrics



def rf_boosting(X_train, Y_train, X_test, Y_test, Y_pred):
    max_n_ests = 51
    fig = plt.figure(figsize=(20,20))
    plt.plot([x for x in range(1, max_n_ests)], [accuracy_score(Y_test, Y_pred)] * (max_n_ests - 1))
    
    average_tst_error = []
    for j in [5, 10, 15, 20, 25]:
        clf_stump=DecisionTreeClassifier(max_features=None,max_depth=j)
        # print(j)
        avg = []
        for i in np.arange(1,max_n_ests):
            bstlfy=AdaBoostClassifier(base_estimator=clf_stump,n_estimators=i)
            bstlfy=bstlfy.fit(X_train,Y_train)
            bst_tr_err= Y_train == bstlfy.predict(X_train)
            bst_tst_err= accuracy_score(Y_test, bstlfy.predict(X_test))
            avg.append(bst_tst_err)

        print(avg)
        plt.plot([x for x in range(1, max_n_ests)], avg)
    
    plt.legend(['Boosting w/ max depth 5', 'Boosting w/ max depth 10', 'Boosting w/ max depth 15', 'Boosting w/ max depth 20', 'Boosting w/ max depth 25'], loc='bottom right')

    fig.suptitle('Test sample accuracy vs Number of estimators')
    plt.xlabel('Number of estimators')
    plt.ylabel('Test sample accuracy')
    # fig.savefig('images/rf/boosting.png')
    plt.show()


def rf_bagging(X_train, X_test, y_train, y_test):
    max_n_ests = 51
    # fig = plt.figure(figsize=(20,20))
    # plt.plot([x for x in range(1, max_n_ests)], [accuracy_score(Y_test, Y_pred)] * (max_n_ests - 1))
    
    clf_stump=DecisionTreeClassifier(max_features=None,max_depth=10)
    # print(j)
    bstlfy=BaggingClassifier(base_estimator=clf_stump,n_estimators=max_n_ests)
    bstlfy=bstlfy.fit(X_train,y_train)
    bst_tr_predict= bstlfy.predict(X_test)
    metrics = print_metrics("Decision Tree Bagging Classifier", y_test, bst_tr_predict)
    return metrics

def wrf_classifier(X,y):
    # wrf = RandomForestClassifier(n_estimators=10, class_weight='balanced')
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    # scores = cross_val_score(wrf, X, y, scoring='recall', cv=cv, n_jobs=-1)
    # summarize performance
    # print('Mean ROC AUC: %.3f' % np.mean(scores))
    model = EasyEnsembleClassifier(n_estimators=10)
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='recall', cv=cv, n_jobs=-1)
    # summarize performance
    print('Mean recall: %.3f' % np.mean(scores))
    
def brf_classifier(X_train, X_test, y_train, y_test):
    brf = BalancedRandomForestClassifier(n_estimators=100,max_depth=20)
    brf.fit(X_train, y_train)
    y_pred = brf.predict(X_test)
    metrics = print_metrics("BRF Classifier", y_test, y_pred)
    return metrics
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    # summarize performance
    # print('Mean ROC AUC: %.3f' % mean(scores))
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    # evaluate model
    # scores = cross_val_score(brf, X,y , scoring='recall', cv=cv, n_jobs=-1)
    # summarize performance
    # print('Mean ROC AUC: %.3f' % np.mean(scores))


def get_tfidf_vectorizer(corpus, max_features=500):
    logging.info("Data vectorized using Tfidf Vectorizer")
    Tfidf_vect = TfidfVectorizer(max_features=max_features)
    Tfidf_vect.fit(corpus)
    return Tfidf_vect

def get_count_vectorizer(corpus, n_grams, max_features = 500):
    logging.info("Data vectorized using Count Vectorizer with {} ngrams".format(n_grams))
    Count_vect = CountVectorizer(ngram_range=(n_grams, n_grams), max_features = max_features)
    Count_vect.fit(corpus)
    return Count_vect


def print_metrics(clf_name, y_test, predicted):
    report = classification_report(y_test, predicted, output_dict=True)
    logging.info("classifier: {}".format(clf_name))
    logging.info("\n" + classification_report(y_test, predicted))
    return report

def save_model(clf_name, clf):
    with open(os.path.join(dir_path,"classifiers/{}.pkl".format(clf_name)), "wb") as clf_f:
        pickle.dump(clf, clf_f)

def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    logging.info("File loaded: {}".format(file_path))
    return data


if __name__ == "__main__":
    dir_path = "/home/ckmj/Documents/Ethiopia-Analysis/ct-hate-speech"
    CUSTOM_SEED=42
    np.random.seed(CUSTOM_SEED)
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hm:x:y:c:v:n:",["model=","X=", "y=", "corpus=", "vectorizer=","ngrams="])
    except getopt.GetoptError:
        print('python3 {} -m <nb|svm> -x <X data file> -y <y data file> -c <corpus file> -v <tfidf|ngrams> -n <1|2> -a <tfidf>'.format(sys.argv[0]))
        sys.exit(2)
    ngrams = None
    for opt, arg in opts:
        if opt == '-h':
            print('python3 {} -m <nb|svm> -x <X data file> -y <y data file> -c <corpus file> -v <tfidf|ngrams> -n <1|2>'.format(sys.argv[0]))
            sys.exit()
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-x", "--X"):
            X_path = "data/" + arg
        elif opt in ("-y", "--y"):
            y_path = "data/" + arg
        elif opt in ("-c", "--corpus"):
            corpus_path = "data/" + arg
        elif opt in ("-v", "--vectorizer"):
            vectorizer_selection = arg
        elif opt in ("-n", "--ngrams"):
            ngrams = int(arg)
        
    if ngrams == None: ngrams = 1


    #all data that i am using has been segmented, stop words and no burmese text removed.


    if vectorizer_selection == "ngrams":
        logging.basicConfig(level=logging.DEBUG, filename=os.path.join(dir_path,"code/logs/{}_{}_{}_{}".format(model, vectorizer_selection, str(ngrams), str(X_path[5:-5]))), filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")
        print("logs/{}_{}_{}_{}".format(model, vectorizer_selection, str(ngrams), str(X_path[5:-5])))
    elif vectorizer_selection == "tfidf":
        logging.basicConfig(level=logging.DEBUG, filename=os.path.join(dir_path,"code/logs/{}_{}_{}".format(model, vectorizer_selection, str(X_path[5:-5]))), filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")
        print("logs/{}_{}_{}".format(model, vectorizer_selection, str(X_path[5:-5])))
    
    corpus = None
    try:
        corpus = load_json_file(os.path.join(dir_path, corpus_path))
    except:
        pass
        
    X = load_json_file(os.path.join(dir_path, X_path))
    y = load_json_file(os.path.join(dir_path, y_path))
    
    if corpus is None:
        corpus = X
    
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80,test_size=0.20)



    # logging.info("Corpus: {}".format(corpus_path))
    # logging.info("X data : {}".format(X_path))
    # logging.info("y data: {}".format(y_path))
    logging.info("Data loaded")   
    logging.info("Num data points: {}".format(len(X)))

    logging.info("Training data size : {}".format(len(y_train)))
    logging.info("Testing data size: {}".format(len(y_test)))

    #This is not one-hot encoding but it replaces the name of the class with a number. So the number of columns will still be 1.
    
    Encoder = LabelEncoder()
    
    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)
    logging.info("Y encoded")

    y = Encoder.fit_transform(y)
    # print(Counter(y_train))
    if vectorizer_selection == "tfidf":
        vectorizer = get_tfidf_vectorizer(corpus = corpus, max_features = 5000)
    elif vectorizer_selection == "ngrams":
        if ngrams != None:
            vectorizer = get_count_vectorizer(corpus = corpus, n_grams = ngrams, max_features = 5000)
        else:
            vectorizer = get_count_vectorizer(corpus = corpus, n_grams = 1, max_features = 5000)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    X = vectorizer.transform(X)
    logging.info("Feature names : {} ...".format(vectorizer.get_feature_names()[:20]))
    logging.info("Vectorized X train dimensions: {}".format(np.shape(X_train_vectorized)))
    logging.info("Vectorized X test dimensions: {}".format(np.shape(X_test_vectorized)))

    # resampled_majority_class_sizes = [1289, 1468, 1768, 2068, 2368, 2668, 2894]   #ratio of minority class to majority class
    resampled_majority_class_sizes = [1]   #ratio of minority class to majority class
    precisions = []
    recalls = []
    accuracies = []
    for majority_class_size in resampled_majority_class_sizes:
        # nmus = NearMiss(sampling_strategy= {1:majority_class_size})
        # X_train_vectorized_resampled, y_train_resampled = nmus.fit_resample(X_train_vectorized, y_train)
        X_train_vectorized_resampled, y_train_resampled = (X_train_vectorized, y_train) 
        logging.info("After resampling: \n \t X data size  : {} y data size : {}".format(np.shape(X_train_vectorized_resampled), np.shape(y_train_resampled)))


        # Different values were tested using GridSearch and the following values yielded the best results.
        svm_grid_values = {
                "C" : (0.01, 0.1, 1, 10, 100),
                "gamma":(10, 1, 0.1, 0.01),
                "kernel": ("linear", "rbf")
                }

        nb_grid_values = {
                "alpha":(0.001, 0.01, 0.1, 0.2, 0.3,0.4)
                }

        if model == "svm":
            metrics = svm_clf(X_train_vectorized_resampled, X_test_vectorized, y_train_resampled, 1,2, True, svm_grid_values)
        elif model == "nb":
            metrics = naive_bayes_clf(X_train_vectorized_resampled, X_test_vectorized, y_train_resampled, 0.2, True, nb_grid_values)
        elif model == "xgb":
            #use tfidf for xgb
            params = {
                'max_depth': 20,  # the maximum depth of each tree
                'eta': 0.0001,  # the training step for each iteration
                'objective': 'binary:logistic',  # error evaluation for multiclass training
                'n_estimators': 100,
                'scale_pos_weight': 20,
                'eval_metric': 'error',
                'gamma':10
            }  # the number of classes that exist in this datset
            metrics = xg_boost_clf(X_train_vectorized_resampled, X_test_vectorized, y_train_resampled, y_test, params)
        elif model == "bag":
            metrics = rf_bagging(X_train_vectorized_resampled, X_test_vectorized, y_train_resampled, y_test)
        elif model == "brf":
            metrics = brf_classifier(X_train_vectorized_resampled, X_test_vectorized, y_train_resampled, y_test)
        precisions.append(metrics["1"]["precision"])
        recalls.append(metrics["1"]["recall"])
        accuracies.append(metrics["accuracy"])

    print(precisions)
    print(recalls)
    print(accuracies)
    
    '''
    plt.title("Majority class undersampling trends in Perf metrics BRF (segmented unicode)", fontsize=10)
    plt.plot([173, 273, 473, 873, 1673, 3273, 4343], precisions, linewidth=2, label="precision")
    plt.plot([173, 273, 473, 873, 1673, 3273, 4343], recalls, linewidth=2, label = "recall")
    plt.plot([173, 273, 473, 873, 1673, 3273, 4343], accuracies, linewidth=2, label = "accuracy")
    plt.gca().set_xlabel("Training Size(Majority class size undersampled)", fontsize=10)
    plt.gca().set_ylabel("Test data performance measures", fontsize=10)
    plt.legend()
    plt.savefig(os.path.join(dir_path, "graphs/test_1.png"))
    plt.show()
    '''
