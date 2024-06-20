import transformers

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.optim.adamw import AdamW
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

import logging
from tqdm import tqdm
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class Trainer(transformers.Trainer):
    def __init__(self, model, train_dataset, test_dataset, config):
        
        self.model = model
        self.config = config

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.template_ids, self.template_mask = train_dataset.get_template()
        self.template_ids = self.template_ids.cuda()
        self.template_mask = self.template_mask.cuda()

        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.writer = SummaryWriter(os.path.join(self.config.logging_dir, TIMESTAMP))
        
        self.optimizer = None
        self.lr_scheduler = None

    def create_optimizer_and_scheduler(self, num_training_steps):
        
        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.config.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + len('encoder.layer.'):].split('.')[0])
                        except:
                            print('err', n)
                            raise Exception("")
                        if layer_num >= self.config.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)

                elif 'embeddings' in n:
                    print('no ', n)
                else:
                    print('yes', n)
                    params[n] = p

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=num_training_steps
            )

    def train(self):
        num_training_steps = self.config.num_train_epochs * self.train_dataset.num_batch() // self.config.gradient_accumulation_steps
        self.create_optimizer_and_scheduler(num_training_steps)

        global_step = 0

        optimizer = self.optimizer
        scheduler = self.lr_scheduler
        model = self.model
        model.train()
        model.zero_grad()

        loss = torch.tensor(0.0).to(f'cuda:{model.device_ids[0]}')

        for epoch in range(int(self.config.num_train_epochs)):

            epoch_loss = torch.tensor(0.0).cuda()

            epoch_iterator = tqdm(self.train_dataset.get_next_thread())
            for step, batch in enumerate(epoch_iterator):

                loss = self.step(model, batch)
                loss = torch.mean(loss)
                loss.backward()

                global_step += 1
                if global_step % self.config.gradient_accumulation_steps == 0:

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss
    
                epoch_iterator.set_description(f"Epoch: {epoch+1} Step:{step+1}|{global_step} Lr: {scheduler.get_last_lr()[0]:.4e} Loss: {epoch_loss/(step+1):.4f}")
            
            self.writer.add_scalar('Training/Loss', epoch_loss/(step+1), epoch)

        self.eval(model, epoch)

    def step(self, model, batch):
        
        root_ids, root_mask, edge_index_FD, edge_index_BD, abs_time, rel_pos, ranking_indices, post_feature, rootIndexs, labels = batch

        root_ids = root_ids.cuda()
        root_mask = root_mask.cuda()
        rootIndexs = rootIndexs.cuda()
        labels = labels.cuda()
        abs_time = abs_time.cuda()
        rel_pos = rel_pos.cuda()
        post_feature = post_feature.cuda()

        return model(
            root_ids, root_mask, 
            self.template_ids, self.template_mask,
            edge_index_FD, edge_index_BD, 
            rootIndexs, labels, abs_time, rel_pos, ranking_indices, post_feature
        )

    def eval(self, model, epoch):

        model.eval()
        eval_loss = 0
        with torch.no_grad():
            
            acc = torch.tensor(0.0)
            label_lst = []
            prediction_lst = []
            for batch in self.test_dataset.get_next_thread():

                predictions, labels, eval_loss_ = self.step(model, batch)
                prediction_lst.extend(predictions)
                label_lst.extend(labels)
                eval_loss += eval_loss_
            
            f1_score_pos, f1_score_neg, macro_f1_score, acc = self.calculate_metrics(prediction_lst, label_lst)
            eval_loss /= len(self.test_dataset.batches)

            self.writer.add_scalar('Eval/Loss', eval_loss, epoch)
            self.writer.add_scalar('Eval/f1_pos', f1_score_pos, epoch)
            self.writer.add_scalar('Eval/f1_neg', f1_score_neg, epoch)
            self.writer.add_scalar('Eval/f1_macro', macro_f1_score, epoch)
            self.writer.add_scalar('Eval/Acc', acc, epoch)
        
        model.train()

    @staticmethod
    def calculate_metrics(predictions, labels):
        predictions = torch.stack(predictions)
        labels = torch.stack(labels)

        TP = torch.sum((predictions == 1) & (labels == 1)).item()
        FP = torch.sum((predictions == 1) & (labels == 0)).item()
        FN = torch.sum((predictions == 0) & (labels == 1)).item()
        TN = torch.sum((predictions == 0) & (labels == 0)).item()

        precision_pos = TP / (TP + FP + 1e-8)
        recall_pos = TP / (TP + FN + 1e-8)
        f1_score_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos + 1e-8)

        precision_neg = TN / (TN + FN + 1e-8)
        recall_neg = TN / (TN + FP + 1e-8)
        f1_score_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg + 1e-8)

        macro_f1_score = (f1_score_pos + f1_score_neg) / 2.0

        # Calculate accuracy
        acc = (TP + TN) / labels.numel()

        return f1_score_pos, f1_score_neg, macro_f1_score, acc