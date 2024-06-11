import torch
from torch.utils.data import Dataset
import os
from transformers import XLMRobertaTokenizerFast
import random
import pickle

THREAD_DATA_FILE = 'thread_data_all.pkl'

class PropDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: XLMRobertaTokenizerFast, template: str, batch_size: int, train=True, selection='bfs', top_k=10) -> None:

        self.train = train
        if self.train:
            self.thread_true = []
            self.thread_false = []

        self.threadDict = {}
        with open(os.path.join(data_path, THREAD_DATA_FILE), 'rb') as f:
            while 1:
                try:
                    thread = pickle.load(f)
                    if thread['post_feature'].shape[0] > 1024:
                        continue
                    self.threadDict[thread['id_']] = thread
                    if self.train:
                        if thread['label'] == '0':
                            self.thread_false.append(thread['id_'])
                        elif thread['label'] == '1':
                            self.thread_true.append(thread['id_'])

                except EOFError:
                    break
        
        self.thread_id = list(self.threadDict.keys())
        
        print('load: ', len(self.thread_id), ' threads')

        self.tokenizer = tokenizer
        self.template = template
        self.template_ids = None
        self.template_mask = None

        self.batch_size = batch_size
        
        self.selection = selection
        self.top_k = top_k
        
        self.get_template()

        random.seed(42)

    def create_batches(self, thread_id, batch_size):

        batches = []

        if not self.train:
            num_samples = len(thread_id)
            
            start_idx = 0
            for end_idx in range(batch_size, num_samples + batch_size, batch_size):
                end_idx = min(num_samples, end_idx)

                batch = thread_id[start_idx:end_idx]
                batches.append(batch)

                start_idx = end_idx
        
        elif self.train:
            maxCount = max(len(self.thread_true), len(self.thread_false))

            while maxCount - len(self.thread_true):
                a = self.thread_true[random.randint(0, len(self.thread_true)-1)]
                self.thread_true.append(a)

            while maxCount - len(self.thread_false):
                a = self.thread_false[random.randint(0, len(self.thread_false)-1)]
                self.thread_false.append(a)
            
            assert len(self.thread_true) == len(self.thread_false)

            random.shuffle(self.thread_true)
            random.shuffle(self.thread_false)

            num_samples = maxCount
            
            start_idx = 0
            for end_idx in range(batch_size, num_samples + batch_size, batch_size):
                end_idx = min(num_samples, end_idx)

                batch_false = self.thread_false[start_idx:end_idx]
                batch_true = self.thread_true[start_idx:end_idx]
                batch = batch_false + batch_true
                batches.append(batch)

                start_idx = end_idx
        
        return batches

    def get_template(self):
        if self.template_ids is None or self.template_mask is None:
            self.template = self.tokenizer(self.template, return_tensors='pt', add_special_tokens=False)
            self.template_ids = self.template['input_ids']
            self.template_mask = self.template['attention_mask']

            self.mask_pos = torch.where(self.template_ids[0] == self.tokenizer.mask_token_id)

        return self.template_ids, self.template_mask

    def get_next_thread(self):
        
        self.batches = self.create_batches(self.thread_id, self.batch_size)

        for batch in self.batches:

            yield self.collate_fn(batch)

    def collate_fn(self, batch):
        
        rootIds = []
        rootMaskList = []
        rootIndexs = []
        absTimeList = []
        relPosList = []
        ranking_indices = []
        labelList = []
        postFeatureList = []
        edge_index_FD = [[], []]

        counter = 0
        for item in batch:
            thread = self.threadDict[item]

            # <------------- root input_ids ------------->
            root_ids = thread['root_ids']
            root_ids = root_ids.squeeze()
            rootIds.append(root_ids.tolist())

            # <------------- root attention_mask ------------->
            root_mask = thread['root_mask']
            root_mask = root_mask.squeeze()
            rootMaskList.append(root_mask.tolist())

            # <------------- edge index FD ------------->
            edge_index = [[], []]
            edge_index[0] = [int(index) + counter for index in thread['edge_index'][0]]
            edge_index[1] = [int(index) + counter for index in thread['edge_index'][1]]

            edge_index_FD[0].extend(edge_index[0])
            edge_index_FD[1].extend(edge_index[1])

            # <------------- absolute time ------------->
            abs_time = thread['abs_time']
            absTimeList.extend(abs_time)

            # <------------- relative position ------------->
            rel_pos = thread['rel_pos']
            relPosList.extend(rel_pos)

            # <------------- ranking ------------->
            ranking = thread['ranking'][self.selection][:self.top_k]
            ranking = [int(index) for index in ranking]
            ranking = torch.tensor(ranking)
            ranking_indices.append(ranking)

            # <------------- post preprocess feature ------------->
            post_feature = thread['post_feature']
            postFeatureList.append(post_feature)

            # <------------- root index ------------->
            root_index = counter
            rootIndexs.append(root_index)

            # <------------- label ------------->
            label = thread['label']
            labelList.append(0) if label == '1' else labelList.append(1)

            counter += len(post_feature)
        
        rootFeature = self.tokenizer.pad({'input_ids': rootIds, 'attention_mask': rootMaskList}, return_tensors='pt', return_attention_mask=True)

        edge_index_BD = [edge_index_FD[1], edge_index_FD[0]]

        postFeatureList = torch.cat(postFeatureList, dim=0)

        return rootFeature['input_ids'], rootFeature['attention_mask'], \
                edge_index_FD, edge_index_BD, \
                torch.LongTensor(absTimeList), torch.LongTensor(relPosList), \
                ranking_indices, postFeatureList, torch.LongTensor(rootIndexs), torch.LongTensor(labelList)

    def num_batch(self):
        if self.train:
            maxCount = max(len(self.thread_true), len(self.thread_false))

            return maxCount // self.batch_size + 1
        else:
            return len(self.thread_id) // self.batch_size + 1