"""
Core implementation borrowed from: https://github.com/chyikwei/recommend
"""

from six.moves import xrange
import numpy as np
from numpy.random import RandomState
import scipy.sparse as sparse


def make_ratings(n_users, n_items, min_rating_per_user, max_rating_per_user,
                 rating_choices, seed=None, shuffle=True):
    """
    Randomly generate a (user_id, item_id, rating) array
    :returns:: ndarray with shape (n_samples, 3)
    """
    if not (isinstance(rating_choices, list) or
            isinstance(rating_choices, tuple)):
        raise ValueError("'rating_choices' must be a list or tuple")
    if min_rating_per_user < 0 or min_rating_per_user >= n_items:
        raise ValueError("invalid 'min_rating_per_user' invalid")
    if (min_rating_per_user > max_rating_per_user) or \
       (max_rating_per_user >= n_items):
        raise ValueError("invalid 'max_rating_per_user' invalid")

    rs = RandomState(seed=seed)
    user_arrs = []
    for user_id in xrange(n_users):
        item_count = rs.randint(min_rating_per_user, max_rating_per_user)
        item_ids = rs.choice(n_items, item_count, replace=False)
        ratings = rs.choice(rating_choices, item_count)
        arr = np.stack(
            [np.repeat(user_id, item_count), item_ids, ratings], axis=1)
        user_arrs.append(arr)

    ratings = np.array(np.vstack(user_arrs))
    ratings[:, 2] = ratings[:, 2].astype('float')
    if shuffle:
        rs.shuffle(ratings)
    return ratings

def build_user_item_matrix(n_users, n_items, ratings, classes=None, the_class=None):
    """
    Build user-item matrix
    :returns: sparse matrix with shape (n_users, n_items)
    """
    #if classes != None and the_class != None:
    #    indices = np.where(classes == the_class)
    data = ratings[:, 2].ravel()
    row_ind = ratings[:, 0].ravel()
    col_ind = ratings[:, 1].ravel()
    shape = (n_users, n_items)
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
