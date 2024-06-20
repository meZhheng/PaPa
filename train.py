from transformers import RobertaConfig,  AutoTokenizer
from transformers import set_seed

from dataloader import PropDataset

import torch.nn as nn
import torch
import os
import random

from model import RobertaPromptTuningLM
from trainer import Trainer

from config import Config

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

    set_seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    main(config)