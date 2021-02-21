# from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, RandomForestRegressor
# from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import confusion_matrix,classification_report
# from imblearn.over_sampling import SMOTE
# from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
import json
# from sklearn.svm import SVC
import numpy as np
import pickle
# import xgboost as xgb

def print_metrics(clf_name, y_test, predicted):
    report = classification_report(y_test, predicted, output_dict=True)
    print("classifier: {}".format(clf_name))
    print("\n" + classification_report(y_test, predicted))
    return report


with open("../../data/X_filtered.json", "r") as X_file:
    X = json.load(X_file)

with open("../../data/y_filtered.json", "r") as y_file:
    y = json.load(y_file)

counts = Counter(y)
print("Original data counts : " + str(counts))

# vect = TfidfVectorizer(max_features=5000)
# vect.fit(X)

vect = CountVectorizer(ngram_range=(2,3), max_features = 5000)
vect.fit(X)

# with open("hate-speech-api/clfs/oversampled-{}.pkl".format("ngrams2_3-vect"), "wb") as vect_f:
    # pickle.dump(vect, vect_f, protocol=2)
# vect = CountVectorizer(ngram_range=(2,2), max_features = 5000)
# vect.fit(X)

Encoder = LabelEncoder()

y = Encoder.fit_transform(y)
X = vect.transform(X)
y_counter = Counter(y)
# resampled_minority_class_sizes = np.arange(y_counter[1], y_counter[0], 500)
resampled_minority_class_sizes =  [y_counter[0]]


precisions = []
recalls = []
accuracies = []

models = {
    # "EasyEnsembleClf": EasyEnsembleClassifier(n_estimators=10),
    # "BalancedRFClf": BalancedRandomForestClassifier(n_estimators=100,max_depth=20),
    "RFClf" : RandomForestClassifier(n_estimators=100, max_depth=20)
    # "BaggingClf"   : BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=10),n_estimators=50),
    # "SVMClf"        : svm.SVC(C=1, gamma=0.1),
    # "MNBClf"        : naive_bayes.MultinomialNB(alpha=0.1),
    # "XGBoostClf"    : xgb.XGBClassifier(learning_rate=0.0001, max_depth = 20, n_estimators = 100, scale_pos_weight=20)
}

for minority_class_size in resampled_minority_class_sizes:
    # oversample = SMOTE(sampling_strategy={1: minority_class_size})
    # oversample = SMOTE(sampling_strategy={1: minority_class_size})

    # X, y = oversample.fit_resample(X, y)
    # with open("X_ngrams_2_3.bin", "wb") as X_out:
        # pickle.dump(X, X_out)
    # with open("y_ngrams_2_3.bin", "wb") as y_out:
        # pickle.dump(y, y_out)
    
    with open("../../data/X_ngrams_2_3.bin", "rb") as X_in:
        X =pickle.load(X_in)
    with open("../../data/y_ngrams_2_3.bin", "rb") as y_in:
        y= pickle.load(y_in)
    y_counter = Counter(y)
    print("After oversampling: " + str(y_counter))
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80,test_size=0.20)

    clf_name = "RFClf"
    clf = models[clf_name]
    clf.fit(X_train, y_train)
    prds = clf.predict(X_test)
    metrics = print_metrics(clf_name, y_test, prds)
    # precisions.append(metrics["1"]["precision"])
    # recalls.append(metrics["1"]["recall"])
    # accuracies.append(metrics["accuracy"])
    
# plt.title("Minority class oversampling trends in Perf metrics {} (segmented unicode filtered)".format(clf_name), fontsize=10)
# plt.plot(resampled_minority_class_sizes, precisions, linewidth=2, label="precision")
# plt.plot(resampled_minority_class_sizes, recalls, linewidth=2, label = "recall")
# plt.plot(resampled_minority_class_sizes, accuracies, linewidth=2, label = "accuracy")
# plt.gca().set_xlabel("Training Size(Minority class sizes oversampled)", fontsize=10)
# plt.gca().set_ylabel("Test data performance measures", fontsize=10)
# plt.legend()
# plt.savefig("oversampling_seg_uni_fil_{}.png".format(clf_name))
with open("clfs/oversampled-{}-ngrams2_3.pkl".format(clf_name), "wb") as clf_f:
    pickle.dump(clf, clf_f, protocol=2)
# plt.show() 






