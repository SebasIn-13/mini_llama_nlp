
import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
	def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
		super(LlamaZeroShotClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# Zero-shot classification does not require updating llama paramters.
		for param in self.llama.parameters():
			param.requires_grad = False
		assert len(label_names) == self.num_labels
		self.tokenizer = tokenizer
		self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]


	def forward(self, input_ids):
		# compute the completion probability of each label string
		logits, _ = self.llama(input_ids)
		log_probabilities = F.log_softmax(logits, dim=-1)
		label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
		for i, label_token_ids in enumerate(self.label_name_ids):
			total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
			label_probabilities[:, i] = total_log_prob[:, 0]
		return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
	def __init__(self, config):
		super(LlamaEmbeddingClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# If we use pretrain mode, we freeze Llama parameters.
		for param in self.llama.parameters():
			if config.option == 'pretrain':
				param.requires_grad = False
			elif config.option == 'finetune':
				param.requires_grad = True

		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

	def forward(self, input_ids):
		'''
		1) Find the hidden state after the final token of the input sequence
		2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
		   overfitting.
		2) Pass this through the classifier head (self.classifier_head), which will return
		   logits (unnormalized probabilities) over all classes.
		3) Take the log-softmax of the logits and return log-probabilities over all classes.
		'''
		# DONE
		#hiddenstates
		logits, hidden_states = self.llama(input_ids)
		# final token (Debido a que hay distintos tamaños y se aplica padding hay que encontrar el último token por batch)
		
		# Se obtiene el tamaño del bache
		batch_size = hidden_states.size(0)

		# 2. Se obtiene el largo de cada secuencia sin padding
		non_pad_mask = input_ids.ne(0) 
		sequence_lengths = non_pad_mask.sum(dim=1) 

		# 3. Se obtiene el índice del último token para cada secuencia.
		last_token_indices = sequence_lengths - 1  

		# 4. Se crea los índices del batch 
		batch_indices = torch.arange(batch_size)

		# 5. Se obtienen los estados ocultos en las posiciones del batch y su índice válido
		final_hidden_states = hidden_states[batch_indices, last_token_indices]
		
		#dropout
		dropout_output = self.dropout(final_hidden_states)
		
		#classifier head
		classifier_output = self.classifier_head(dropout_output)
		
		#log-softmax
		log_probabilities = F.log_softmax(classifier_output, dim=-1)
		return log_probabilities
