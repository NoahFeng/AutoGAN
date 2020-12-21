# -*- coding: utf-8 -*-
# @Date    : 2019-09-29
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_search.building_blocks_search import CONV_TYPE, NORM_TYPE, UP_TYPE, SHORT_CUT_TYPE, SKIP_TYPE


class Controller(nn.Module):
    def __init__(self, args, cur_stage):
        """
        init
        :param args:
        :param cur_stage: varies from 0 to ...
        """
        super(Controller, self).__init__()
        self.hid_size = args.hid_size
        self.cur_stage = cur_stage
        self.lstm = torch.nn.LSTMCell(self.hid_size, self.hid_size)
        if cur_stage:
            # SKIP_TYPE的说明：当处在cur_stage阶段，说明当前cell之前共有cur_stage个cell，最多会接入cur_stage个跳连，
            # 前面每一个cell接入情况有len(SKIP_TYPE)种，那么cur_stage个cell共有len(SKIP_TYPE)**cur_stage种可能。
            self.tokens = [len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE), len(SHORT_CUT_TYPE), len(SKIP_TYPE)**cur_stage]
        else:
            self.tokens = [len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE), len(SHORT_CUT_TYPE)]
        # embedding向量个数为：当前cell中所有组件的总选项数目。 embedding向量的特征维度为：hid_size
        self.encoder = nn.Embedding(sum(self.tokens), self.hid_size)
        self.decoders = nn.ModuleList([nn.Linear(self.hid_size, token) for token in self.tokens])

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hid_size, requires_grad=False).cuda()

    def forward(self, x, hidden, index):
        if index == 0:
            embed = x
        else:
            # 第x个选项对应的embedding  x属于[0, sum(self.tokens) - 1]
            embed = self.encoder(x)
        hx, cx = self.lstm(embed, hidden)

        # decode
        logit = self.decoders[index](hx)

        return logit, (hx, cx)

    def sample(self, batch_size, with_hidden=False, prev_hiddens=None, prev_archs=None):
        # 这里的batch_size指的是搜索架构有batch_size个。
        x = self.initHidden(batch_size)

        if prev_hiddens:
            assert prev_archs
            prev_hxs, prev_cxs = prev_hiddens
            # 从prev_archs中随机选择batch_size个架构  即从之前已经搜索好的架构中，选出来若干个作为基础，再搜索当前cell
            selected_idx = np.random.choice(len(prev_archs), batch_size)  # TODO: replace=False
            selected_idx = [int(x) for x in selected_idx]

            selected_archs = []
            selected_hxs = []
            selected_cxs = []

            for s_idx in selected_idx:
                selected_archs.append(prev_archs[s_idx].unsqueeze(0))
                selected_hxs.append(prev_hxs[s_idx].unsqueeze(0))
                selected_cxs.append(prev_cxs[s_idx].unsqueeze(0))
            selected_archs = torch.cat(selected_archs, 0)
            hidden = (torch.cat(selected_hxs, 0), torch.cat(selected_cxs, 0))
        else:
            hidden = (self.initHidden(batch_size), self.initHidden(batch_size))
        entropies = []
        actions = []
        selected_log_probs = []
        # 当前cell下所需搜索的组件，共len(self.decoders)个组件
        for decode_idx in range(len(self.decoders)):
            # 经过LSTM cell之后，得到的当前组件的logits以及hidden state
            logit, hidden = self.forward(x, hidden, decode_idx)
            # 计算当前组件的各选项的概率
            prob = F.softmax(logit, dim=-1)     # bs * logit_dim
            log_prob = F.log_softmax(logit, dim=-1)
            entropies.append(-(log_prob * prob).sum(1, keepdim=True))  # bs * 1
            # 以多项式采样，从当前组件的选项方案中采样出1种选项(变量action的取值为[0, 当前组件选项方案数 - 1]) 即RL中的action
            action = prob.multinomial(1)  # batch_size * 1
            actions.append(action)
            # 获取已采样到的选项对应的log_prob
            selected_log_prob = log_prob.gather(1, action.data)  # batch_size * 1
            selected_log_probs.append(selected_log_prob)

            # x即当前组件的已选中选项在embedding层的下标
            x = action.view(batch_size)+sum(self.tokens[:decode_idx])
            x = x.requires_grad_(False)

        archs = torch.cat(actions, -1)  # batch_size * len(self.decoders)
        selected_log_probs = torch.cat(selected_log_probs, -1)  # batch_size * len(self.decoders)
        entropies = torch.cat(entropies, 0)  # bs * 1

        if prev_hiddens:
            # 将之前已搜好的基础架构和当前轮次搜到的cell架构组合起来
            archs = torch.cat([selected_archs, archs], -1)

        if with_hidden:
            return archs, selected_log_probs, entropies, hidden

        return archs, selected_log_probs, entropies
