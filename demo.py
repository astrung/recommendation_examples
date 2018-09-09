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

items = pd.read_csv('items.txt', sep=';', error_bad_lines=False, header=None)
users = pd.read_csv('usersDescription.txt', sep=';', header=None)
ratings = pd.read_csv('ratings.txt', sep=';', header=None)

from lightfm.data import Dataset

dataset = Dataset(user_identity_features=False, item_identity_features=True)
dataset.fit(users=(users[50].unique()), items=(items[0]))

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

ratings2 = ratings[ratings[2] > 0]
ratings2 = ratings2.drop_duplicates(subset=[1,2,3])
train, test = train_test_split(ratings2, test_size=0.1)
print(train.shape)
print(test.shape)

(train_interactions, train_weights) = dataset.build_interactions(train[[3, 1]].values)
(test_interactions, test_weights) = dataset.build_interactions(test[[3, 1]].values)

# arr = sparse.coo_matrix(np.tile(list(range(2,10)), (len(items), 1)))
# items['features'] = arr.toarray().tolist()
# # item_features = dataset.build_item_features()
# # items2 = items.to_dict('records')

from lightfm import LightFM

model = LightFM(loss='warp', random_state=0)
model.fit(train_interactions, epochs=100, num_threads=1)

from lightfm.evaluation import recall_at_k
from lightfm.evaluation import precision_at_k

print("Train recall@7: %.2f" % recall_at_k(model, train_interactions, k=7).mean())
print("Test recall@7: %.2f" % recall_at_k(model, test_interactions, train_interactions, k=7).mean())
print("Train precision@7: %.2f" % precision_at_k(model, train_interactions, k=7).mean())
print("Test precision@7: %.2f" % precision_at_k(model, test_interactions, train_interactions, k=7).mean())
print("Train reciprocal rank: %.2f" % reciprocal_rank(model, train_interactions).mean())
print("Test reciprocal rank: %.2f" % reciprocal_rank(model, test_interactions, train_interactions).mean())

def get_predict(user, model, list_item):
    print("=========================")
    print("User %d" % user)
    scores = model.predict(user - 1, list_item)
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
    print("=========================")

for user in np.random.choice(test[3], 5).tolist():
    get_predict(user, model, np.arange(18))