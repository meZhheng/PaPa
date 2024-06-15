from transformers import XLMRobertaForMaskedLM, XLMRobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.TimeAwareGCN import TimeAwareGCN
from modules.TimeAwareEncoder import TimeAwareEncoder
import os
        
class Roberta_Prototypical_BiGCN:
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        extendConfig,
        **kwargs,
    ):  
        print("Initializing Pretrained Language Encoder...")
        model = super().from_pretrained(model_name_or_path, **kwargs)

        print("Initializing Soft Prompts...")
        model.initialize_soft_prompts(n_tokens=extendConfig.n_tokens)

        print("Initializing Prototypical Classifier...")
        model.initialize_prototype(cl_nn_class=extendConfig.cl_nn_class, cl_nn_size=extendConfig.cl_nn_size)

        print("Initializing Propagation Learning Module...")
        model.initialize_propagation_module(extendConfig)

        return model

    def initialize_soft_prompts(self, n_tokens: int) -> None:

        self.n_tokens = n_tokens

        init_prompt_value = self.embeddings.word_embeddings.weight[:self.n_tokens].clone().detach()

        self.soft_prompt = nn.Embedding(self.n_tokens, self.config.hidden_size)

        self.soft_prompt.weight = nn.Parameter(init_prompt_value, requires_grad=True)
    
    def initialize_prototype(self, cl_nn_class, cl_nn_size) -> None:
        self.cl_nn_class = cl_nn_class

        self.cl_nn = nn.Linear(self.config.hidden_size, cl_nn_size, bias=False)
        w = torch.empty((cl_nn_class, cl_nn_size))
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=True)
    
    def initialize_propagation_module(self, config) -> None:
        self.mask_pos = config.mask_pos
        self.mode = config.ablation

        relPosEncoder = TimeAwareEncoder(config.emb_dim, config.train_aware_encoder)
        absTimeEncoder = TimeAwareEncoder(config.emb_dim, config.train_aware_encoder)

        if self.mode == 'rm_tae':
            relPosEncoder = None
            absTimeEncoder = None

        self.BiGCN = TimeAwareGCN(config, relPosEncoder=relPosEncoder, absTimeEncoder=absTimeEncoder)
    
    def responseRanking(self, feature, rootIndex, ranking_indices):
        if feature is None:
            return None, None

        tmp = []
        for i, root_idx in enumerate(rootIndex):

            rank_ids = ranking_indices[i]
            if root_idx == rootIndex[-1]:
                batch = feature[root_idx:]
            else:
                batch = feature[root_idx:rootIndex[i+1]]

            batch = torch.index_select(feature, dim=0, index=rank_ids.cuda())

            tmp.append(batch)
        
        propagation_emb = []
        propagation_mask = []
        for item in tmp:
            if len(item) == self.n_tokens:
                propagation_emb.append(item)
                propagation_mask.append(torch.ones((self.n_tokens)))
            else:
                propagation_emb.append(torch.cat((item, torch.zeros((self.n_tokens - len(item), item.shape[1])).cuda()), dim=0))
                propagation_mask.append(torch.cat((torch.ones((len(item))), torch.zeros((self.n_tokens - len(item)))), dim=0))

        propagation_emb = torch.stack(propagation_emb, dim=0)
        propagation_mask = torch.stack(propagation_mask, dim=0)

        return propagation_emb, propagation_mask.cuda()

    def cat_input_embedding(self, template_ids, learnable_emb, propagation_emb, input_ids) -> torch.Tensor:
        inputs_embeds = self.embeddings.word_embeddings(input_ids)

        if template_ids is None and learnable_emb is None:
            return torch.cat((propagation_emb, inputs_embeds), dim=1)
        
        elif template_ids is None:
            learnable_emb = learnable_emb.repeat(inputs_embeds.size(0), 1, 1)
            propagation_emb = torch.add(propagation_emb, learnable_emb)
            return torch.cat((propagation_emb, inputs_embeds), dim=1)
        
        elif learnable_emb is None:

            template_emb = self.embeddings.word_embeddings(template_ids)
            return torch.cat((template_emb, propagation_emb, inputs_embeds), dim=1)
        
        elif propagation_emb is None:
            learnable_emb = learnable_emb.repeat(inputs_embeds.size(0), 1, 1)
            template_emb = self.embeddings.word_embeddings(template_ids)
            return torch.cat((template_emb, learnable_emb, inputs_embeds), dim=1)
        
        else:
            learnable_emb = learnable_emb.repeat(inputs_embeds.size(0), 1, 1)
            template_emb = self.embeddings.word_embeddings(template_ids)
            propagation_emb = torch.add(propagation_emb, learnable_emb)

            return torch.cat((template_emb, propagation_emb, inputs_embeds), dim=1)

    def cat_attention_mask(self, template_mask, propagation_mask, attention_mask):
        
        if template_mask is None:
            return torch.cat((propagation_mask, attention_mask), dim=1)
        
        elif propagation_mask is None:
            learnable_mask = torch.ones((attention_mask.size(0), self.n_tokens)).cuda()

            return torch.cat((template_mask, learnable_mask, attention_mask), dim=1)
        
        else:
            template_mask = template_mask.expand(attention_mask.size(0), -1)
            return torch.cat((template_mask, propagation_mask, attention_mask), dim=1)
        
    def save_model(self, path, name):
        self.save_pretrained(os.path.join(path, name))
    
    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1, 0))
    
    def cl_loss_format(self, v_ins):
        # instance-prototype loss
        sim_mat = torch.exp(self.sim(v_ins, self.proto))
        num = sim_mat.shape[1]

        loss_pro = 0.
        for i in range(num):
            pos_score = torch.diag(sim_mat[:, i, :])
            neg_score = (sim_mat[:, i, :].sum(1) - pos_score)
            logits_ = pos_score / (pos_score + neg_score)
            loss_pro += - torch.log(logits_).sum()

        loss_pro = loss_pro / (num * self.cl_nn_class * self.cl_nn_class)

        # instance-instance loss
        loss_ins = 0.
        for i in range(v_ins.shape[0]):
            sim_instance = torch.exp(self.sim(v_ins, v_ins[i]))
            pos_ins = sim_instance[i]
            neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
            loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins)).sum()
        loss_ins = loss_ins / (num * self.cl_nn_class * num * self.cl_nn_class)
        loss = loss_pro + loss_ins

        return loss

    def cal_logits(self, v_pro):
        sim_mat_agg = torch.exp(self.sim(v_pro, self.proto))
        logits = F.softmax(sim_mat_agg, dim=1)
        return logits
        
    def forward(self, root_ids, root_mask, template_ids, template_mask,
        edge_index_TD, edge_index_BU, roots, labels, abs_time, rel_pos, ranking_indices, post_feature
    ):  
        template_ids = template_ids.expand(root_ids.size(0), -1)
        template_mask = template_mask.expand(root_mask.size(0), -1)
        
        # <------------- Concat Prompt Learning Template ------------->
        propagation_emb = None
        propagation_mask = None
        learnable_emb = None

        if self.mode == 'all':
            propagation_emb = self.BiGCN(post_feature, edge_index_TD, edge_index_BU, roots, abs_time, rel_pos)
            learnable_emb = self.soft_prompt.weight

        elif self.mode == 'rm_hard':
            propagation_emb = self.BiGCN(post_feature, edge_index_TD, edge_index_BU, roots, abs_time, rel_pos)
            learnable_emb = self.soft_prompt.weight

            template_ids = None
            template_mask = None

        elif self.mode == 'rm_soft':
            propagation_emb = self.BiGCN(post_feature, edge_index_TD, edge_index_BU, roots, abs_time, rel_pos)
        
        elif self.mode == 'rm_prompt':
            propagation_emb = self.BiGCN(post_feature, edge_index_TD, edge_index_BU, roots, abs_time, rel_pos)

            template_ids = None
            template_mask = None
        
        elif self.mode == 'rm_prop':
            learnable_emb = self.soft_prompt.weight
        
        propagation_emb, propagation_mask = self.responseRanking(propagation_emb, roots, ranking_indices)

        inputs_embeds = self.cat_input_embedding(template_ids, learnable_emb, propagation_emb, root_ids)
        attention_mask = self.cat_attention_mask(template_mask, propagation_mask, root_mask)

        # <------------- Pretrained Language Encoder ------------->
        output = super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = output.last_hidden_state
        mask_output = sequence_output[torch.arange(sequence_output.size(0)), self.mask_pos]

        # <------------- Prototypical Classifier ------------->
        if self.training:  # and self.cl_mode:
            embeds = [[] for _ in range(2)]

            for j in range(len(mask_output)):
                label = labels[j]
                embeds[label].append(mask_output[j])
            embeds = [torch.stack(e) for e in embeds]
            embeds = torch.stack(embeds)
            x = self.cl_nn(embeds)

            loss = self.cl_loss_format(x)

            return loss
        
        elif self.eval:
            logits = self.cal_logits(self.cl_nn(mask_output))

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            predictions = torch.argmax(logits, dim=1)

            return predictions, labels, loss
    
class RobertaPromptTuningLM(Roberta_Prototypical_BiGCN, XLMRobertaModel):
    def __init__(self, config):
        super().__init__(config)

    