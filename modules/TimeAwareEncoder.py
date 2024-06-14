import torch 
import torch.nn as nn

class TimeAwareEncoder(nn.Module):

	def __init__(self, d_emb_dim, requires_grad):
		super(TimeAwareEncoder, self).__init__()

		self.d_emb_dim = d_emb_dim
		self.requires_grad = requires_grad

		# <------------- Defining the position embedding -------------> 
		self.aware_weights = nn.Parameter(torch.randn((self.d_emb_dim, 1)), requires_grad=self.requires_grad)

	def forward(self, pos):

		aware_encoding = torch.cos(self.aware_weights * pos)

		return aware_encoding.t()