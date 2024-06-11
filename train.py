from transformers import RobertaConfig,  AutoTokenizer
from transformers import TrainingArguments, GlueDataTrainingArguments
from transformers import HfArgumentParser
from transformers import set_seed

from dataloader import PropDataset

from dataclasses import dataclass, field

import torch.nn as nn
import torch
import copy
import logging
import os

from model import RobertaPromptTuningLM
from trainer import Trainer

from config import Config

@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    cache_dir: str = field(
        metadata={"help": "Path to trained model"}
    )

    cl_nn_size: int = field(
        default=64,
    )

    cl_nn_class: int = field(
        default=2,
    )

    n_tokens: int = field(
        default=10
    )

    prompt_pos = None

    mask_pos: int = field(
        default=None
    )

@dataclass
class ExtendedDataArguments(GlueDataTrainingArguments):

    raw_data_dir: str = field(
        default="../Dataset",
        metadata={"help": "Path to raw data"}
    )

@dataclass
class ExtendedTrainingArguments(TrainingArguments):

    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "If evaluate model when training"}
    )

    save_at_last: bool = field(
        default=False
    )

    train_batch_size: int = field(
        default=24,
        metadata={"help": "Batch size"}
    )

    start_eval_steps: int = field(
        default=500,
    )
    
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

def main(config):
  
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    
    dataset = PropDataset(config.data_train, tokenizer=tokenizer, template=config.template, batch_size=config.batch_size, selection=config.ranking, top_k=config.n_tokens)
    test_dataset = PropDataset(config.data_test, tokenizer=tokenizer, template=config.template, batch_size=config.batch_size_test, train=False, selection=config.ranking, top_k=config.n_tokens)

    config.mask_pos = dataset.mask_pos

    # Create config
    model_config = RobertaConfig.from_pretrained(
        config.model_name_or_path,
        finetuning_task=config.task_name,
        cache_dir=config.cache_dir,
    )

    model = RobertaPromptTuningLM.from_pretrained(
        config.model_name_or_path,
        extendConfig=config,
        config=model_config,
        cache_dir=config.cache_dir,
    )

    model = nn.DataParallel(model.cuda())

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        test_dataset=test_dataset,
        config=config
    )
    
    trainer.train()

    if config.save_at_last:
        trainer.save_model(config.cache_dir)

if __name__ == '__main__':
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in vars(config)["gpu_idx"])

    set_seed(42)
    torch.manual_seed(42)

    main(config)