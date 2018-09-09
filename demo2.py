import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn import preprocessing
from lightfm import LightFM
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score
import time
from lightfm.evaluation import reciprocal_rank
import scipy.sparse as sparse
import pickle
from sklearn.model_selection import train_test_split

def build_user_dict(users):
    result = []
    for user in users.itertuples():
        user_feature_raw = np.argwhere(np.array(user[3:50]) == 1)
        user_feature = np.reshape(user_feature_raw, len(user_feature_raw))
        result.append((user[51], user_feature + 2))
    return result

items = pd.read_csv('items.txt', sep=';', error_bad_lines=False, header=None)
users = pd.read_csv('usersDescription.txt', sep=';', header=None)
ratings = pd.read_csv('ratings.txt', sep=';', header=None)

from lightfm.data import Dataset

dataset = Dataset(user_identity_features=True, item_identity_features=True)
dataset.fit(users=(users[50].unique()), items=(items[0]),
            item_features=list(range(2, 10)),
            user_features=list(range(2, 50))
            )

items_features_raw = list(
    (item[1], (np.argwhere(np.array(item[3:]) == 1)[0] + 2).tolist()) for item in items.itertuples())
items_features = dataset.build_item_features(items_features_raw)
users_features_raw = build_user_dict(users)
users_features = dataset.build_user_features(users_features_raw)

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

ratings2 = ratings[ratings[2] > 0]
ratings2 = ratings2.drop_duplicates(subset=[1, 2, 3])
train, test = train_test_split(ratings2, test_size=0.1)
print(train.shape)
print(test.shape)

(train_interactions, train_weights) = dataset.build_interactions(train[[3, 1]].values)
(test_interactions, test_weights) = dataset.build_interactions(test[[3, 1]].values)

from lightfm import LightFM

model = LightFM(loss='warp', random_state=0)
model.fit(train_interactions, user_features=users_features, item_features=items_features, epochs=200, num_threads=1)

from lightfm.evaluation import recall_at_k
from lightfm.evaluation import precision_at_k

print("Train recall@7: %.2f" % recall_at_k(model, train_interactions, k=7, user_features=users_features,
                                           item_features=items_features).mean())
print("Test recall@7: %.2f" % recall_at_k(model, test_interactions, train_interactions, k=7,
                                          user_features=users_features, item_features=items_features).mean())
print("Train precision@7: %.2f" % precision_at_k(model, train_interactions, k=7, user_features=users_features,
                                              item_features=items_features).mean())
print("Test precision@7: %.2f" % precision_at_k(model, test_interactions, train_interactions, k=7,
                                             user_features=users_features, item_features=items_features).mean())
print("Train reciprocal rank: %.2f" % reciprocal_rank(model, train_interactions, user_features=users_features,
                                                      item_features=items_features).mean())
print("Test reciprocal rank: %.2f" % reciprocal_rank(model, test_interactions, train_interactions,
                                                     user_features=users_features,
                                                     item_features=items_features).mean())

def get_predict(user, model, list_item):
    print("User %d" % user)
    scores = model.predict(user - 1, list_item, item_features=items_features, user_features=users_features)
    print("Liked items from history:")
    known_result = train[train[3] == user][1].values
    print(known_result)
    print("Favorite items ( haven't seen ):")
    # print(ratings2[ratings2[3] == 1][1].values)
    print(test[test[3] == user][1].values)
    print("Predict results")
    predict_result = np.argsort(-scores) + 1
    removed_known_results = np.setdiff1d(predict_result, known_result)
    print(removed_known_results[:7])

for user in np.random.choice(test[3], 5).tolist():
    get_predict(user, model, np.arange(18))
