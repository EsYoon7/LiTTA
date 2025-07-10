import random
import os
import logging
from copy import deepcopy
from datetime import datetime
import math
import pickle
import string
import json
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import Wav2Vec2ForCTC
from speechbrain.pretrained import EncoderDecoderASR
import nemo.collections.asr as nemo_asr

from typing import Optional, Tuple, Union
import ipdb

def set_seed(seed):
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


def get_logger(args):
	logger = logging.getLogger("main")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(message)s')

	time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	file_handler = logging.FileHandler(os.path.join(args.log_dir, f"log_{time_string}.txt"))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


def get_model(args):
	if args.asr in ["facebook/wav2vec2-base-960h"]: # CTC-based models
		model = Wav2Vec2ForCTC.from_pretrained(args.asr).requires_grad_(True).eval()
		if 'cuda' in args.device:
			model = model.cuda()
	elif args.asr in ["speechbrain/asr-crdnn-rnnlm-librispeech", "speechbrain/asr-crdnn-transformerlm-librispeech", "speechbrain/asr-transformer-transformerlm-librispeech", "speechbrain/asr-conformersmall-transformerlm-librispeech"]: # attention-based models
		model = EncoderDecoderASR.from_hparams(args.asr, run_opts={"device": args.device}).requires_grad_(True).eval()
	elif args.asr in ["pretrained_models/stt_en_conformer_ctc_small.nemo", "pretrained_models/stt_en_conformer_ctc_small_ls.nemo"]: # conformers
		model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.asr).to(args.device).requires_grad_(True).eval()
	elif args.asr in ["pretrained_models/stt_en_conformer_transducer_small.nemo", "pretrained_models/stt_en_conformer_transducer_large.nemo"]: # transducers
		model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.asr).to(args.device).requires_grad_(True).eval()
	return model


def collect_params(model, train_params):
	if isinstance(model, Wav2Vec2ForCTC):
		return collect_params_ctc(model, train_params)
	elif isinstance(model, EncoderDecoderASR):
		return collect_params_attn(model, train_params)
	elif isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
		return collect_params_conf(model, train_params)
	elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
		return collect_params_trans(model, train_params)


def collect_params_ctc(model, train_params):
	params, names = [], []
	for nm, m in model.named_modules():
		if "all" in train_params:
			for np, p in m.named_parameters():
				p.requires_grad = True
				if not f"{nm}.{np}" in names:
					params.append(p)
					names.append(f"{nm}.{np}")
		if "feature" in train_params:
			if len(str(nm).split('.')) > 1:
				if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
					for np, p in m.named_parameters():
						p.requires_grad = True
						if not f"{nm}.{np}" in names:
							params.append(p)
							names.append(f"{nm}.{np}")
		if "LN" in train_params:
			if isinstance(m, nn.LayerNorm):
				for np, p in m.named_parameters():
					if np in ['weight', 'bias']:  
						p.requires_grad = True
						if not f"{nm}.{np}" in names:
							params.append(p)
							names.append(f"{nm}.{np}")
	return params, names


def collect_params_attn(model, train_params):
	params, names = [], []
	for nm, m in model.named_modules():
		for np, p in m.named_parameters():
			collect = False
			if "all" in train_params:
				collect = True
			if 'enc' in train_params and 'encoder' in str(nm):
				collect = True
			if 'dec' in train_params and 'decoder' in str(nm):
				collect = True
			if 'LN' in train_params and isinstance(m, nn.LayerNorm):
				collect = True

			if collect:
				p.requires_grad = True
				params.append(p)
				names.append(f"{nm}.{np}")
	return params, names


def collect_params_conf(model, train_params):
	params, names = [], []
	for nm, m in model.named_modules():
		for np, p in m.named_parameters():
			collect = False 
			if "all" in train_params:
				collect = True
			if 'LN' in train_params and isinstance(m, nn.LayerNorm):
				collect = True
			if 'BN' in train_params and isinstance(m, nn.BatchNorm1d):
				collect = True
			if 'enc' in train_params and 'encoder' in nm:
				collect = True
			if 'dec' in train_params and 'decoder' in nm:
				collect = True

			if collect:
				p.requires_grad = True
				params.append(p)
				names.append(f"{nm}.{np}")
	return params, names


def collect_params_trans(model, train_params):
	params, names = [], []
	for nm, m in model.named_modules():
		for np, p in m.named_parameters():
			collect = False 
			if "all" in train_params:
				collect = True
			if 'LN' in train_params and isinstance(m, nn.LayerNorm):
				collect = True
			if 'BN' in train_params and isinstance(m, nn.BatchNorm1d):
				collect = True
			if 'enc' in train_params and 'encoder' in nm:
				collect = True
			if 'dec' in train_params and 'decoder' in nm:
				collect = True
			if 'joint' in train_params and 'joint' in nm:
				collect = True

			if collect:
				p.requires_grad = True
				params.append(p)
				names.append(f"{nm}.{np}")
	return params, names


def freeze_norm_stats(model):
	for m in model.modules():
		if isinstance(m, nn.BatchNorm1d):
			m.track_running_stats = False


def set_rnn_to_train(model):
	for m in model.modules():
		if isinstance(m, torch.nn.modules.rnn.RNNBase):
			m.train()
			m.dropout = 0
	return model


def get_optimizer(args, params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None):
	opt = getattr(torch.optim, opt_name)
	if opt_name == 'Adam':       
		optimizer = opt(params, lr=lr, betas=(beta, 0.999), weight_decay=weight_decay)
	else: 
		optimizer = opt(params, lr=lr, weight_decay=weight_decay)
	
	if scheduler is not None: 
		return optimizer, eval(scheduler)(optimizer, T_max=args.t_max, eta_min=args.lr_min)
	return optimizer, None


def copy_model_and_optimizer(model, optimizer, scheduler):
	"""Copy the model and optimizer states for resetting after adaptation."""
	model_state = deepcopy(model.state_dict())
	optimizer_state = deepcopy(optimizer.state_dict())
	if scheduler is not None:
		scheduler_state = deepcopy(scheduler.state_dict())
		return model_state, optimizer_state, scheduler_state
	else:
		return model_state, optimizer_state, None


def load_model_and_optimizer(model, optimizer, scheduler, model_state, optimizer_state, scheduler_state):
	"""Restore the model and optimizer states from copies."""
	model.load_state_dict(model_state, strict=True)
	optimizer.load_state_dict(optimizer_state)
	if scheduler is not None:
		scheduler.load_state_dict(scheduler_state)
		return model, optimizer, scheduler
	else: 
		return model, optimizer, None


def get_blank_index(args, model, processor):
	if isinstance(model, Wav2Vec2ForCTC):
		blank_index = 0
	elif isinstance(model, EncoderDecoderASR):
		blank_index = model.mods.decoder.blank_index
	elif isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
		blank_index = 128
	elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
		blank_index = processor.blank
	return blank_index


def softmax_entropy(x, dim=-1):
	return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def renyi_entropy(x, alpha, dim=-1):
	# x: (B, L, D)
	if alpha == 1:
		return torch.mean(softmax_entropy(x, dim))
	if alpha == 'inf' or alpha == float('inf'):
		entropy, _ = torch.max(x, dim)
		return -torch.mean(torch.log(entropy))
	entropy = torch.log(torch.pow(x.softmax(dim), alpha).sum(dim)) # entropy: B, L
	entropy = entropy / (1 - alpha)
	return torch.mean(entropy)

def renyi_entropy_raw(x, alpha, dim=-1):
	# x: (B, L, D)
	if alpha == 1:
		return torch.mean(softmax_entropy(x, dim))
	if alpha == 'inf' or alpha == float('inf'):
		entropy, _ = torch.max(x, dim)
		return -torch.mean(torch.log(entropy))
	entropy = torch.log(torch.pow(x.softmax(dim), alpha).sum(dim)) # entropy: B, L
	entropy = entropy / (1 - alpha)
	return entropy

def marginal_renyi_entropy(x, alpha, dim=-1):
	# x: (B, L, D)
	if alpha == 1:
		x = x.mean(0)
		return torch.mean(softmax_entropy(x, dim))
	if alpha == 'inf' or alpha == float('inf'):
		x = x.mean(0)
		entropy, _ = torch.max(x, dim)
		return -torch.mean(torch.log(entropy))
	x_softmax = x.softmax(dim)
	x_marginal = x_softmax.mean(0)
	entropy = torch.log(torch.pow(x_marginal, alpha).sum(dim)) # entropy: B, L
	entropy = entropy / (1 - alpha)
	return torch.mean(entropy)

def non_saturating_loss(x, dim=-1):
	max_idx = torch.argmax(x, dim=dim, keepdim=True)
	one_hots = torch.zeros_like(x).scatter(dim, max_idx, 1).to(x.device)
	return - torch.mean(one_hots * x) + torch.log(((1 - one_hots) * torch.exp(x)).sum(dim=dim)).mean()


def mcc_loss(x, class_num, reweight=False, dim=-1):
	mcc_loss = 0
	for x_split in x: # (B, L, D) -> (L, D)
		x_split = x_split.unsqueeze(0)
		p = x_split.softmax(dim) # (1, L, D)
		p = p.squeeze(0) # (L, D)

		if reweight: # (1, L, D) * (L, 1)
			target_entropy_weight = softmax_entropy(x_split, dim=-1).detach().squeeze(0) # instance-wise entropy (1, L, D)
			target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (1, L)
			target_entropy_weight = x_split.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
			cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)
		else:
			cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) -> (D, D)

		cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
		mcc_loss += (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
	mcc_loss /= len(x)
	return mcc_loss


def js_divergence(p1, p2):
	total_m = 0.5 * (p1 + p2)
	loss = 0.5 * F.kl_div(torch.log(p1), total_m, reduction="batchmean") + 0.5 * F.kl_div(torch.log(p2), total_m, reduction="batchmean")
	return loss


def log_softmax(x, axis):
	x_max = np.amax(x, axis=axis, keepdims=True)
	if x_max.ndim > 0:
		x_max[~np.isfinite(x_max)] = 0
	elif not np.isfinite(x_max):
		x_max = 0
	tmp = x - x_max
	exp_tmp = np.exp(tmp)
	with np.errstate(divide="ignore"):
		s = np.sum(exp_tmp, axis=axis, keepdims=True)
		out: np.ndarray = np.log(s)
	out = tmp - out
	return out

# =============================================================================================================
def compute_mask_indices(
	shape: Tuple[int, int],
	mask_prob: float,
	mask_length: int,
	attention_mask: Optional[torch.LongTensor] = None,
	min_masks: int = 0,
) -> np.ndarray:
	"""
	Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
	ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
	CPU as part of the preprocessing during training.

	Args:
		shape: The shape for which to compute masks. This should be of a tuple of size 2 where
			   the first element is the batch size and the second element is the length of the axis to span.
		mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
					independently generated mask spans of length `mask_length` is computed by
					`mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
					actual percentage will be smaller.
		mask_length: size of the mask
		min_masks: minimum number of masked spans
		attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
						each batch dimension.
	"""
	batch_size, sequence_length = shape

	if mask_length < 1:
		raise ValueError("`mask_length` has to be bigger than 0.")

	if mask_length > sequence_length:
		raise ValueError(
			f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
			f" and `sequence_length`: {sequence_length}`"
		)

	# epsilon is used for probabilistic rounding
	epsilon = np.random.rand(1).item()

	def compute_num_masked_span(input_length):
		"""Given input length, compute how many spans should be masked"""
		num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
		num_masked_span = max(num_masked_span, min_masks)

		# make sure num masked span <= sequence_length
		if num_masked_span * mask_length > sequence_length:
			num_masked_span = sequence_length // mask_length

		# make sure num_masked span is also <= input_length - (mask_length - 1)
		if input_length - (mask_length - 1) < num_masked_span:
			num_masked_span = max(input_length - (mask_length - 1), 0)

		return num_masked_span

	# compute number of masked spans in batch
	input_lengths = (
		attention_mask.sum(-1).detach().tolist()
		if attention_mask is not None
		else [sequence_length for _ in range(batch_size)]
	)

	# SpecAugment mask to fill
	spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
	spec_aug_mask_idxs = []

	max_num_masked_span = compute_num_masked_span(sequence_length)

	if max_num_masked_span == 0:
		return spec_aug_mask

	for input_length in input_lengths:
		# compute num of masked spans for this input
		num_masked_span = compute_num_masked_span(input_length)

		# get random indices to mask
		spec_aug_mask_idx = np.random.choice(
			np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
		)

		# pick first sampled index that will serve as a dummy index to pad vector
		# to ensure same dimension for all batches due to probabilistic rounding
		# Picking first sample just pads those vectors twice.
		if len(spec_aug_mask_idx) == 0:
			# this case can only happen if `input_length` is strictly smaller then
			# `sequence_length` in which case the last token has to be a padding
			# token which we can use as a dummy mask id
			dummy_mask_idx = sequence_length - 1
		else:
			dummy_mask_idx = spec_aug_mask_idx[0]

		spec_aug_mask_idx = np.concatenate(
			[spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
		)
		spec_aug_mask_idxs.append(spec_aug_mask_idx)

	spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

	# expand masked indices to masked spans
	spec_aug_mask_idxs = np.broadcast_to(
		spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
	)
	spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

	# add offset to the starting indexes so that indexes now create a span
	offsets = np.arange(mask_length)[None, None, :]
	offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
		batch_size, max_num_masked_span * mask_length
	)
	spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

	# ensure that we cannot have indices larger than sequence_length
	if spec_aug_mask_idxs.max() > sequence_length - 1:
		spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

	# scatter indices to mask
	np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

	return spec_aug_mask

def save_pkl(data, file):
	with open(file, 'wb') as f:
		pickle.dump(data, f)
	return

def load_pkl(file):
	with open(file, 'rb') as f:
		data = pickle.load(f)
	return data

def append_pkl(data, file):
	with open(file, 'ab') as f:
		pickle.dump(data, f)
	return

def read_txt(file):
	file = open(file, 'r')
	data = []
	while True:
		line = file.readline()
		data.append(line.strip().split('\t'))
		if not line:
			break
	file.close()

	return data

def csv_write(file, column_name, data=None):
	with open(file, 'w', newline='\n') as f:
		write = csv.writer(f)
		write.writerow(column_name)
		if data == None:
			return
		else:
			assert len(data) == len(column_name)
			write.writerow(column_name)
			for i in range(len(data[0])):
				line = [data[column_idx][i] for column_idx, _ in enumerate(column_name)]
				write.writerow(line)
	return

def csv_append(file, data):
	with open(file, 'a', newline='\n') as f:
		write = csv.writer(f)
		for i in range(len(data[0])):
			line = [data[j][i] for j in range(len(data))]
			write.writerow(line)
	return

def csv_append_ver2(file, data):
	with open(file, 'a', newline='\n') as f:
		write = csv.writer(f)
		for idx, line in enumerate(data):
			# line = [data[j][i] for j in range(len(data))]
			write.writerow(line)
	return

def csv_read(file):
	data = []
	alphabet = list(string.ascii_letters)
	with open(file, 'r', newline='') as f:
		reader = csv.reader(f, delimiter=' ',  quotechar='|')
		for row in reader:
			# ipdb.set_trace()
			# if len([i for i in row[0] if i not in alphabet]) == 0:
			data.append(row[0])
	return data

def save_jsonl(file, data):
	with open(file, encoding='utf-8', mode='w') as f:
		for i in data:
			f.write(json.dump(i)+'\n')
	return

def append_jsonl(file, data):
	with open(file, encoding='utf-8', mode='a') as f:
		f.write(json.dumps(data)+'\n')
	return

def read_jsonl(file):
	with open(file, 'r', encoding="utf-8") as f:
		data = [json.loads(line) for line in f]
	return data

def print_now(return_flag=0):
	# commented out by esyoon 2023-09-24-18:44:05
	# t_delta = datetime.timedelta(hours=9)
	# JST = datetime.timezone(t_delta, 'JST')
	
	from datetime import date, datetime, timezone, timedelta
	KST  = timezone(timedelta(hours=9))
	date = str(date.today())
	time_record = str(datetime.now(KST).time())[:8]
	now = date+'_'+time_record
	if return_flag == 0:
		print(now)
	elif return_flag == 1:
		return now
	else:
		pass

def normalize_dict(input_dict):
	total = sum(input_dict.values())
	return {key: value / total for key, value in input_dict.items()}

def softmax_dict(input_dict):
	# total = math.exp(sum(input_dict.values()))
	total = sum([math.exp(i) for i in input_dict.values()])
	return {key: math.exp(value) / total for key, value in input_dict.items()}

def reverse_softmax_dict(input_dict):
	# total = math.exp(-1 * sum(input_dict.values()))
	total = sum([math.exp(-1 *i) for i in input_dict.values()])
	return {key: math.exp(-1 * value) / total for key, value in input_dict.items()}

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except Exception as e:
            print("to device error: {}".format(e))
            assert 0
            output[k] = v
    return output