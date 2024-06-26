class Config():

	def __init__(self):
		# Global settings
		self.model_name_or_path = '../xlm-roberta-base'
		self.task_name = 'PaPa'
		self.cache_dir = './checkpoints'
		self.logging_dir = './logs'
		self.save_at_last = False

		self.seed = 42

		self.ablation = 'rm_soft'			# Optional[all, rm_soft, rm_hard, rm_prompt, rm_prop, rm_tae, rm_pro]

		# GPU settings 
		self.gpu = True
		self.gpu_idx = [0] 					# Do note that number of GPU < batch_size 
		self.main_gpu = [0]

		# Training
		self.gradient_accumulation_steps = 1
		self.num_train_epochs = 30
		self.batch_size = 4
		self.batch_size_test = 8
		self.num_classes = 2

		# Time Aware Encoder settings
		self.emb_dim = 768
		self.train_aware_encoder = True

		# Data paths
		self.data_train = "data/Twitter" 			# Optional[Twitter15, Terrorist, Twitter, Weibo]
		self.data_test = "data/TwitterCovid"		# Optional[Twitter16, Gossip, TwitterCovid, WeiboCovid]
		
		# Model parameters settings
		self.d_model = 768
		self.dropout_rate = 0.3

		# Learning rate
		self.warmup_steps = 725					# Optional[4885 for PHEME, 2485 for Weibo, 725 for Twitter]
		self.learning_rate = 1e-5
		self.adam_beta1 = 0.90
		self.adam_beta2 = 0.98
		self.adam_epsilon = 1e-8

		# Prompt Learning Framework
		self.fix_layers = 0
		self.n_tokens = 50
		self.template = '<s> Here is a piece of news with <mask> information.'
		self.ranking = 'bfs'

		# Prototypical Classifier
		self.cl_nn_class = 2
		self.cl_nn_size = 64

		# Bi-direction GCN
		self.gcn_hidden_dim = 768
		self.gcn_dropout = 0.0
		self.gcn_edge_dropout = 0.1

		# label verbalizer
		self.label_mapping = {
			'0': 'confirm',
			'1': 'rumor'
		}
		
	def __repr__(self):
		return str(vars(self))
