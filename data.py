from numba import njit
import abc
from numba.typed import List

from random import seed as set_seed
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from polara import get_movielens_data
from polara.preprocessing.dataframes import reindex, leave_one_out

from sampler import prime_sampler_state, sample_unseen
from utils import transform_indices


def get_dataset(verbose=False, path=None, splitting='temporal'):
    if path != 'None':
        mldata = pd.read_csv(path)
    else: 
        mldata = get_movielens_data(include_time=True).rename(columns={'movieid': 'itemid'})

    if splitting == 'temporal':
        test_timepoint = mldata['timestamp'].quantile(
            q=0.95, interpolation='nearest'
        )
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            'userid not in @test_data_.userid.unique() and timestamp < @test_timepoint'
        )
        training, data_index = transform_indices(train_data_.copy(), 'userid', 'itemid')

        test_data = reindex(test_data_, data_index['items'])
        if verbose:
            print(test_data.nunique())
        testset_, holdout_ = leave_one_out(
            test_data, target='timestamp', sample_top=True, random_state=0
        )
        testset_valid_, holdout_valid_ = leave_one_out(
            testset_, target='timestamp', sample_top=True, random_state=0
        )

    elif splitting == 'leave-one-out':
        mldata, data_index = transform_indices(mldata.copy(), 'userid', 'itemid')
        training, holdout_ = leave_one_out(
        mldata, target='timestamp', sample_top=True, random_state=0
        )
        training_valid_, holdout_valid_ = leave_one_out(
            training, target='timestamp', sample_top=True, random_state=0
        )

        testset_valid_ = training_valid_.copy()
        testset_ = training.copy()
        training = training_valid_.copy()

    else:
        raise ValueError

    userid = data_index['users'].name
    test_users = pd.Index(
        # ensure test users are the same across testing data
        np.intersect1d(
            testset_valid_[userid].unique(),
            holdout_valid_[userid].unique()
        )
    )
    testset_valid = (
        testset_valid_
        # reindex warm-start users for convenience
        .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
        .query(f'{userid} >= 0')
        .sort_values('userid')
    )
    holdout_valid = (
        holdout_valid_
        # reindex warm-start users for convenience
        .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
        .query(f'{userid} >= 0')
        .sort_values('userid')
    )

    testset_ = (
        testset_
        # reindex warm-start users for convenience
        .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
        .query(f'{userid} >= 0')
        .sort_values('userid')
    )
    holdout_ = (
        holdout_
        # reindex warm-start users for convenience
        .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
        .query(f'{userid} >= 0')
        .sort_values('userid')
    )

    if verbose:
        print(testset_valid.nunique())
        print(holdout_valid.shape)
    assert holdout_valid.set_index('userid')['timestamp'].ge(
        testset_valid
        .groupby('userid')
        ['timestamp'].max()
    ).all()

    data_description = dict(
        users = data_index['users'].name,
        items = data_index['items'].name,
        order = 'timestamp',
        n_users = len(data_index['users']),
        n_items = len(data_index['items']),
    )

    if verbose:
        print(data_description)
    dict_popularities = training.groupby('itemid')['userid'].count().reset_index().rename(columns={'userid': 'count'}).sort_values('itemid')['count'].values
    if verbose:
        print(dict_popularities)

    return training, data_description, dict_popularities, testset_valid, testset_, holdout_valid, holdout_

def no_sample(user_items, maxlen, pad_token):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.empty((maxlen, 1))

    n_user_items = min(len(user_items) - 1, maxlen)

    seq[-n_user_items:] = user_items[-n_user_items-1:-1]
    pos[-n_user_items:] = user_items[-n_user_items:]

    return seq, pos, neg

def sample_dross(all_items, user_items, maxlen, pad_token, n_neg_samples, random_state):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)

    n_user_items = min(len(user_items) - 1, maxlen)

    seq[-n_user_items:] = user_items[-n_user_items-1:-1]
    pos[-n_user_items:] = user_items[-n_user_items:]
    
    neg = random_state.choice(all_items[~np.isin(all_items, user_items)], n_neg_samples, replace=False)

    return seq, pos, neg

def sample_with_rep(user_items, maxlen, pad_token, n_neg_samples, itemnum, random_state):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.full((n_neg_samples, maxlen), pad_token, dtype=np.int32)

    n_user_items = min(len(user_items) - 1, maxlen)

    seq[-n_user_items:] = user_items[-n_user_items-1:-1]
    pos[-n_user_items:] = user_items[-n_user_items:]
    neg[:, -n_user_items:] = random_state.randint(0, itemnum, (n_neg_samples, n_user_items))

    return seq, pos, neg

@njit
def sample_without_rep(user_items, maxlen, pad_token, n_neg_samples, itemnum, seed):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.full((maxlen, n_neg_samples), pad_token, dtype=np.int32)

    hist_items_counter = 1
    nxt = user_items[-1]
    idx = maxlen - 1

    set_seed(seed)

    ts_ = list(set(user_items))

    for i in user_items[-2::-1]:
        seq[idx] = i
        pos[idx] = nxt

        state = prime_sampler_state(itemnum, ts_)
        remaining = itemnum - len(ts_)
        
        sample_unseen(n_neg_samples, state, remaining, neg[idx])

        nxt = i
        idx -= 1
        hist_items_counter += 1
        if idx == -1:
            break
        
    neg = np.swapaxes(neg, 0, 1)
    return seq, pos, neg


class BaseAugmenter(abc.ABC):
    def __init__(self, aug_parameters):
        pass

    def fit(self, data):
        pass

    def get_augmentation(self, idx):
        pass

class SemanticAugmenter(BaseAugmenter):
    def __init__(self, aug_parameters, maxlen, pad_token, random_state):
        super().__init__(aug_parameters)
        self.params = aug_parameters
        self.sliding_window = self.params.get("sliding_window", False)
        self.maxlen = maxlen
        self.pad_token = pad_token
        self.random_state = random_state

    def fit(self, data):
        self.orig_sequences = data
        self.orig_idx = []
        if self.sliding_window:
            res_sequence = []
            for k, (uid, sequence) in enumerate(data):
                sub_sequence = sequence[-self.sliding_window.max_len:]
                for i in range(2, len(sub_sequence)-1):
                    res_sequence.append(
                        (uid, sub_sequence[:i])
                    )
                    if i == len(sub_sequence) - 2:
                        self.orig_idx.append(k)
                    else:
                        self.orig_idx.append(-1)
                
            self.aug_sequences = res_sequence
            self.orig_idx = np.array(self.orig_idx)
        else:
            self.aug_sequences = data
            self.orig_idx = np.arange(len(data))
        
        self.prepare_same_target_seq()

    def prepare_same_target_seq(self):
        targets = np.array([seq[-1] for _, seq in self.orig_sequences])
        complementary_targets = np.array([seq[-1] for _, seq in self.aug_sequences]) 
        print("Preparing same target seq")
        self.same_target_idx = []
        for index, item_id in enumerate(targets):
            all_index_same_id = np.where(complementary_targets == item_id)[0]
            delete_index = np.argwhere( self.orig_idx[all_index_same_id] == index )
            all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)
            self.same_target_idx.append(all_index_same_id_wo_self)
        print("Finish same target")
    
    def get_augmentation(self, idx):
        pos_aug = np.full(self.maxlen, self.pad_token, dtype=np.int32)
        same_target_idx = self.same_target_idx[idx]
        if len(same_target_idx) >= 1:
            aug_seq_id = self.random_state.choice(same_target_idx, 1)[0]
        else:
            aug_seq_id = idx
        
        user_items = self.aug_sequences[aug_seq_id][1]
        n_user_items = min(len(user_items) - 1, self.maxlen)
        pos_aug[-n_user_items:] = user_items[-n_user_items-1:-1]

        return pos_aug

class CoocAugmenter(BaseAugmenter):
    def __init__(self, aug_parameters):
        super().__init__(aug_parameters)

    def fit(self, data):
        pass
    
    def get_augmentation(self, idx):
        pass


class SequentialDataset(Dataset):
    def __init__(self, user_train, usernum, itemnum, maxlen, seed, n_neg_samples=1, sampling='without_rep', augmentations_config = None, pad_token=None):
        super().__init__()
        self.user_train = user_train

        #PREV
        #self.valid_users = [user for user in range(usernum) if len(user_train.get(user, [])) > 1]
        #self.usernum = len(self.valid_users)

        #CUR
        self.valid_users = user_train
        self.usernum = len(user_train)
        self.augmentations_config = augmentations_config
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        #if self.augmentations_config is not None and "target_semantic" in self.augmentations_config:
        #    self.prepare_same_target_seq()
        if self.augmentations_config is not None and "target_semantic" in self.augmentations_config:
            self.augmenter = SemanticAugmenter(
                self.augmentations_config, 
                maxlen=maxlen, 
                pad_token=pad_token,
                random_state=self.random_state
            )
            self.augmenter.fit(self.valid_users)

        self.itemnum = itemnum
        self.maxlen = maxlen
        self.n_neg_samples = n_neg_samples
        self.sampling = sampling
        
        if self.sampling == 'dross':
            self.all_items = np.arange(self.itemnum, dtype=np.int32)

        self.pad_token = pad_token

    def __len__(self):
        return self.usernum
    
    def __getitem__(self, idx):
        #PREV
        #user = self.valid_users[idx]
        #user_items = List()
        #[user_items.append(x) for x in self.user_train[user]]

        #CUR
        user, user_items = self.valid_users[idx]

        if self.sampling == 'with_rep':
            seq, pos, neg = sample_with_rep(user_items, self.maxlen, self.pad_token, self.n_neg_samples, self.itemnum, self.random_state)
        elif self.sampling == 'without_rep':
            seq, pos, neg = sample_without_rep(user_items, self.maxlen, self.pad_token, self.n_neg_samples, self.itemnum, self.random_state.randint(np.iinfo(int).min, np.iinfo(int).max))
        elif self.sampling == 'dross':
            seq, pos, neg = sample_dross(self.all_items, user_items, self.maxlen, self.pad_token, self.n_neg_samples, self.random_state)
        elif self.sampling == 'no_sampling':
            seq, pos, neg = no_sample(user_items, self.maxlen, self.pad_token)
        else:
            raise NotImplementedError()
        
        pos_aug = self.prepare_pos_aug(idx)

        return user, seq, pos, neg, pos_aug

    def get_pos_aug(self, idx):
        pos_aug = self.prepare_pos_aug(idx)
        return pos_aug
    
    def prepare_pos_aug(self, idx):
        """pos_aug = np.full(self.maxlen, self.pad_token, dtype=np.int32)
        if self.augmentations_config is not None and "target_semantic" in self.augmentations_config:
            same_target_idx = self.same_target_idx[idx]
            if len(same_target_idx) >= 1:
                aug_seq_id = self.random_state.choice(same_target_idx, 1)[0]
            else:
                aug_seq_id = idx
            
            user_items = self.valid_users[aug_seq_id][1]
            n_user_items = min(len(user_items) - 1, self.maxlen)
            pos_aug[-n_user_items:] = user_items[-n_user_items-1:-1]

        return pos_aug"""

        if self.augmentations_config is not None and "target_semantic" in self.augmentations_config:
            return self.augmenter.get_augmentation(idx)
        else:
            return np.full(self.maxlen, self.pad_token, dtype=np.int32)
    
    def prepare_same_target_seq(self):
        targets = np.array([seq[-1] for _, seq in self.valid_users])
        print("Preparing same target seq")
        self.same_target_idx = []
        for index, item_id in enumerate(targets):
            all_index_same_id = np.where(targets == item_id)[0]
            delete_index = np.argwhere(all_index_same_id == index)
            all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)
            self.same_target_idx.append(all_index_same_id_wo_self)
        print("Finish same target")

def augment_sliding_window(sequences, max_len):
    res_sequence = []
    for uid, sequence in sequences.items():
        sub_sequence = sequence[-max_len:]
        for i in range(2, len(sub_sequence)-1):
            res_sequence.append(
                (uid, sub_sequence[:i])
            )
    return res_sequence

def prepare_sequences(sequences, validation = False):
    res_sequence = []
    num_min = 1 if validation else 2
    for uid, sequence in sequences.items():
        if len(sequence) < num_min:
            continue
        res_sequence.append((uid, sequence))
    return res_sequence

def data_to_sequences(data, data_description, augmentations_config = None, validation = False):
    userid = data_description['users']
    itemid = data_description['items']
    sequences = (
        data.sort_values([userid, data_description['order']])
        .groupby(userid, sort=False)[itemid].apply(list)
    )

    #if augmentations_config is not None and "sliding_window" in augmentations_config:
    #    sequences = augment_sliding_window(sequences, augmentations_config["sliding_window"]["max_len"])
    #else:
    #    sequences = prepare_sequences(sequences, validation)
    
    sequences = prepare_sequences(sequences, validation)

    return sequences

if __name__ == '__main__':
    get_dataset()
