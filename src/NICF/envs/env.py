#!/usr/bin/python
# encoding: utf-8

import numpy as np
import ipdb
import numpy as np
import ipdb
from .. import logger
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from ..util import *
import math
from collections import Counter
import copy as cp

# TODO:
class env(object):
    def __init__(self, args, dls):
        logger.log("initialize environment")
        self.T = args.T
        self.rates = {}
        self.items = {}
        self.users = {}
        self.utypes = {}
        self.utype_kind = {}
        self.ideal_list = {}
        self.args = args
        self.dls = dls

        p_data = dls.pre_training_data

        item_id2iid = {}

        def _load_data(data):
            for index, row in data.iterrows():
                uid = row.user_id
                item_id = row.item_id
                rating = row.rating
                if uid not in self.rates:
                    self.rates[uid] = {}
            
                if item_id not in item_id2iid:
                    item_id2iid[item_id] = len(item_id2iid) + 1

                iid = item_id2iid[item_id]

                self.rates[uid][iid] = rating
                
                self.items[iid] = rating # maintain api
        
        _load_data(p_data)

        for online_id in dls.online_id_stat:
            id_data = dls.grouped_data.get_group(online_id)
            _load_data(id_data)        

        # with open(path_join(self.args.data_path, "env.dat"), "r") as f:
        #     for line in f:
        #         line = line.strip("\n").split("\t")
        #         iid = list(map(lambda x:x.split(":"),line[1:]))
        #         self.rates[int(line[0])] = {int(i[0]):int(i[1]) for i in iid}
        #         for i in iid: self.items[int(i[0])]=int(i[1]) # 只用来统计 self.item_num 用

        logger.log("user number: " + str(len(self.rates) + 1))
        logger.log("item number: " + str(len(self.items) + 1))
        logger.log("user type"
                   " number: " + str(len(self.utype_kind) + 1))
        
        # self.setup_train_test()
        self.setup_train_test_()

    @property
    def user_num(self):
        return len(self.rates) + 1

    @property
    def item_num(self):
        return len(self.items) + 1

    @property
    def utype_num(self):
        return len(self.utypes) + 1

    def setup_train_test(self):
        users = list(range(1, self.user_num))
        np.random.shuffle(users)
        self.training, self.validation, self.evaluation = np.split(np.asarray(users), [int(.85 * self.user_num - 1),
                                                                                       int(.9 * self.user_num - 1)])

    def setup_train_test_(self):
        self.training = np.array(self.dls.pre_training_data["user_id"].drop_duplicates())

        self.evaluation = np.array(self.dls.online_id_stat)

    def reset(self):
        self.reset_with_users(np.random.choice(self.training))

    def reset_with_users(self, uid):
        self.state = [(uid,1), []]
        self.short = {}
        return self.state

    def step(self, action):
        """
        self.rates: 
            dict: uid -> rating_dict
                rating_dict:item_id -> rating
        
        self.state:
            list:
                0:tuple (uid, xx)
                1:list
        """
        if action in self.rates[self.state[0][0]] and (not action in self.short):
            rate = self.rates[self.state[0][0]][action]
            if rate >= 4:
                reward = 1
            else:
                if self.dls.datan in ["KuaiRec"]:
                    reward = rate
                else:
                    reward = 0
        else:
            rate = 0
            reward = 0

        if len(self.state[1]) < self.T - 1:
            done = False
        else:
            done = True
        self.short[action] = 1
        t = self.state[1] + [[action, reward, done]]
        info = {"precision": self.precision(t),
                "recall": self.recall(t, self.state[0][0]),
                "rate":rate}
        self.state[1].append([action, reward, done, info])
        return self.state, reward, done, info

    def step_policy(self,policy):
        policy = policy[:self.args.T]
        rewards = []
        for action in policy:
            if action in self.rates[self.state[0][0]]:
                rewards.append(self.rates[self.state[0][0]][action])
            else:
                rewards.append(0)
        t = [[a, rewards[i], False] for i,a in enumerate(policy)]
        info = {"precision": self.precision(t),
                "recall": self.recall(t, self.state[0][0])}
        self.state[1].extend(t)
        return self.state,rewards,True,info


    def ndcg(self, episode, uid):
        if len(self.rates[uid]) > len(episode):
            return self.dcg_at_k(list(map(lambda x: x[1], episode)),
                                 len(episode),
                                 method=1) / self.dcg_at_k(sorted(list(self.rates[uid].values()),reverse=True),
                                                           len(episode),
                                                           method=1)
        else:
            return self.dcg_at_k(list(map(lambda x: x[1], episode)),
                                 len(episode),
                                 method=1) / self.dcg_at_k(
                list(self.rates[uid].values()) + [0] * (len(episode) - len(self.rates[uid])),
                len(episode), method=1)

    def dcg_at_k(self, r, k, method=1):
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')

    def alpha_dcg(self, item_list, k=10, alpha=0.5, *args):
        items = []
        G = []
        for i, item in enumerate(item_list[:k]):
            items += item
            G.append(sum(map(lambda x: math.pow(alpha, x - 1), dict(Counter(items)).values())) / math.log(i + 2, 2))
        return sum(G)

    def precision(self, episode):
        return sum([i[1] for i in episode])

    def recall(self, episode, uid):
        if self.dls.datan in ["KuaiRec"]:
            satisfield_num = (np.array(list(self.rates[uid].values())) >= 1).sum()
        else:
            satisfield_num = (np.array(list(self.rates[uid].values())) >= 4).sum()
        # if satisfield_num == 0:
        #     return 0

        # return sum([i[1] for i in episode]) / len(self.rates[uid])
        return sum([i[1] for i in episode]) / satisfield_num
