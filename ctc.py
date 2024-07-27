# TODO: try to replace fancy tensor indexing by gather / scatter 

import math
import torch

#@torch.jit.script
def ctc_loss(log_probs : torch.Tensor, targets : torch.Tensor, input_lengths : torch.Tensor, target_lengths : torch.Tensor, blank : int = 0, reduction : str = 'none', finfo_min_fp32: float = torch.finfo(torch.float32).min, finfo_min_fp16: float = torch.finfo(torch.float16).min, alignment : bool = False) -> torch.Tensor:
	input_time_size, batch_size = log_probs.shape[:2]
	B = torch.arange(batch_size, device = input_lengths.device)
	
	_t_a_r_g_e_t_s_ = torch.cat([targets, targets[:, :1]], dim = -1)
	_t_a_r_g_e_t_s_ = torch.stack([torch.full_like(_t_a_r_g_e_t_s_, blank), _t_a_r_g_e_t_s_], dim = -1).flatten(start_dim = -2)
	
	diff_labels = torch.cat([torch.tensor([[False, False]], device = targets.device).expand(batch_size, -1), _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2]], dim = 1)
	
	# if zero = float('-inf') is used as neutral element, custom logsumexp must be used to avoid nan grad in torch.logsumexp
	
	zero_padding, zero = 2, torch.tensor(finfo_min_fp16 if log_probs.dtype == torch.float16 else finfo_min_fp32, device = log_probs.device, dtype = log_probs.dtype)
	log_probs_ = log_probs.gather(-1, _t_a_r_g_e_t_s_.expand(input_time_size, -1, -1))
	log_alpha = torch.full((input_time_size, batch_size, zero_padding + _t_a_r_g_e_t_s_.shape[-1]), zero, device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[0, :, zero_padding + 0] = log_probs[0, :, blank]
	log_alpha[0, :, zero_padding + 1] = log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]]
	# log_alpha[1:, :, zero_padding:] = log_probs.gather(-1, _t_a_r_g_e_t_s_.expand(len(log_probs), -1, -1))[1:]
	for t in range(1, input_time_size):
		log_alpha[t, :, 2:] = log_probs_[t] + logadd(log_alpha[t - 1, :, 2:], log_alpha[t - 1, :, 1:-1], torch.where(diff_labels, log_alpha[t - 1, :, :-2], zero))

	l1l2 = log_alpha[input_lengths - 1, B].gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)) 
	loss = -torch.logsumexp(l1l2, dim = -1)
	return loss

	if not alignment:
		return loss
	
	# below is for debugging, for real alignment use more efficient the distinct ctc_alignment(...) method
	path = torch.zeros(len(log_alpha), len(B), device = log_alpha.device, dtype = torch.int64)
	path[input_lengths - 1, B] = zero_padding + 2 * target_lengths - 1 + l1l2.argmax(dim = -1)
	for t, indices in reversed(list(enumerate(path))[1:]):
		indices_ = torch.stack([(indices - 2) * diff_labels[B, (indices - zero_padding).clamp(min = 0)], (indices - 1).clamp(min = 0), indices], dim = -1)
		path[t - 1] += (indices - 2 + log_alpha[t - 1, B].gather(-1, indices_).argmax(dim = -1)).clamp(min = 0)
	return torch.zeros_like(log_alpha).scatter_(-1, path.unsqueeze(-1), 1.0)[..., (zero_padding + 1)::2]

#@torch.jit.script
def ctc_alignment(log_probs : torch.Tensor, targets : torch.Tensor, input_lengths : torch.Tensor, target_lengths : torch.Tensor, blank: int = 0, finfo_min_fp32: float = torch.finfo(torch.float32).min, finfo_min_fp16: float = torch.finfo(torch.float16).min) -> torch.Tensor:
	input_time_size, batch_size = log_probs.shape[:2]
	B = torch.arange(batch_size, device = input_lengths.device)
	
	_t_a_r_g_e_t_s_ = torch.cat([
		torch.stack([torch.full_like(targets, blank), targets], dim = -1).flatten(start_dim = -2),
		torch.full_like(targets[:, :1], blank)
	], dim = -1)
	diff_labels = torch.cat([
		torch.tensor([[False, False]], device = targets.device).expand(batch_size, -1),
		_t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2]
	], dim = 1)

	zero_padding, zero = 2, torch.tensor(finfo_min_fp16 if log_probs.dtype == torch.float16 else finfo_min_fp32, device = log_probs.device, dtype = log_probs.dtype)
	padded_t = zero_padding + _t_a_r_g_e_t_s_.shape[-1]
	log_alpha = torch.full((batch_size, padded_t), zero, device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[:, zero_padding + 0] = log_probs[0, :, blank]
	log_alpha[:, zero_padding + 1] = log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]]

	packmask = 0b11
	packnibbles = 4 # packnibbles = 1
	backpointers_shape = [len(log_probs), batch_size, int(math.ceil(padded_t / packnibbles))]
	backpointers = torch.zeros(backpointers_shape, device = log_probs.device, dtype = torch.uint8)
	backpointer = torch.zeros((backpointers_shape[-2], backpointers_shape[-1] * packnibbles), device = log_probs.device, dtype = torch.uint8)
	packshift = torch.tensor([[[6, 4, 2, 0]]], device = log_probs.device, dtype = torch.uint8)

	for t in range(1, input_time_size):
		prev = torch.stack([log_alpha[:, 2:], log_alpha[:, 1:-1], torch.where(diff_labels, log_alpha[:, :-2], zero)])
		log_alpha[:, zero_padding:] = log_probs[t].gather(-1, _t_a_r_g_e_t_s_) + prev.logsumexp(dim = 0)
		backpointer[:, zero_padding:(zero_padding + prev.shape[-1] )] = prev.argmax(dim = 0)
		torch.sum(backpointer.unflatten(-1, (-1, packnibbles)) << packshift, dim = -1, out = backpointers[t]) # backpointers[t] = backpointer

	l1l2 = log_alpha.gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1))

	path = torch.zeros(input_time_size, batch_size, device = log_alpha.device, dtype = torch.long)
	path[input_lengths - 1, B] = zero_padding + target_lengths * 2 - 1 + l1l2.argmax(dim = -1)

	for t in range(input_time_size - 1, 0, -1):
		indices = path[t]
		backpointer = (backpointers[t].unsqueeze(-1) >> packshift).view_as(backpointer) #backpointer = backpointers[t]
		path[t - 1] += indices - backpointer.gather(-1, indices.unsqueeze(-1)).squeeze(-1).bitwise_and_(packmask)
	
	return torch.zeros_like(_t_a_r_g_e_t_s_, dtype = torch.int64).scatter_(-1, (path.t() - zero_padding).clamp(min = 0), torch.arange(input_time_size, device = log_alpha.device).expand(batch_size, -1))[:, 1::2]



def ctc_alignment_uncompressed(
	log_probs : torch.Tensor,
	targets : torch.Tensor,
	input_lengths : torch.Tensor,
	target_lengths : torch.Tensor,
	blank: int = 0,
	pack_backpointers: bool = False,
	finfo_min_fp32: float = torch.finfo(torch.float32).min,
	finfo_min_fp16: float = torch.finfo(torch.float16).min
) -> torch.Tensor:
	B = torch.arange(len(targets), device = input_lengths.device)
	_t_a_r_g_e_t_s_ = torch.cat([
		torch.stack([torch.full_like(targets, blank), targets], dim = -1).flatten(start_dim = -2),
		torch.full_like(targets[:, :1], blank)
	], dim = -1)
	diff_labels = torch.cat([
		torch.as_tensor([[False, False]], device = targets.device).expand(len(B), -1),
		_t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2]
	], dim = 1)

	zero, zero_padding = torch.tensor(finfo_min_fp16 if log_probs.dtype is torch.float16 else finfo_min_fp32, device = log_probs.device, dtype = log_probs.dtype), 2
	padded_t = zero_padding + _t_a_r_g_e_t_s_.shape[-1]
	log_alpha = torch.full((len(B), padded_t), zero, device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[:, zero_padding + 0] = log_probs[0, :, blank]
	log_alpha[:, zero_padding + 1] = log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]]

	packmask = 0b11
	packnibbles = 4
	padded_t = int(math.ceil(padded_t / packnibbles)) * packnibbles
	backpointers_shape = [len(log_probs), len(B), padded_t]
	backpointers = torch.zeros(
		backpointers_shape if not pack_backpointers else (backpointers_shape[:-1] + (padded_t // packnibbles, )),
		device = log_probs.device,
		dtype = torch.uint8
	)
	backpointer = torch.zeros(backpointers_shape[1:], device = log_probs.device, dtype = torch.uint8)
	packshift = torch.tensor([[[6, 4, 2, 0]]], device = log_probs.device, dtype = torch.uint8)

	for t in range(1, len(log_probs)):
		prev = torch.stack([log_alpha[:, 2:], log_alpha[:, 1:-1], torch.where(diff_labels, log_alpha[:, :-2], zero)])
		log_alpha[:, 2:] = log_probs[t].gather(-1, _t_a_r_g_e_t_s_) + prev.logsumexp(dim = 0)
		backpointer[:, 2:(2 + prev.shape[-1])] = prev.argmax(dim = 0)
		if pack_backpointers:
			torch.sum(backpointer.view(len(backpointer), -1, 4) << packshift, dim = -1, out = backpointers[t])
		else:
			backpointers[t] = backpointer

	l1l2 = log_alpha.gather(
		-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)
	)

	path = torch.zeros(len(log_probs), len(B), device = log_alpha.device, dtype = torch.long)
	path[input_lengths - 1, B] = zero_padding + target_lengths * 2 - 1 + l1l2.argmax(dim = -1)

	for t in range(len(path) - 1, 0, -1):
		indices = path[t]

		if pack_backpointers:
			backpointer = (backpointers[t].unsqueeze(-1) >> packshift).view_as(backpointer)
		else:
			backpointer = backpointers[t]

		path[t - 1] += indices - backpointer.gather(-1, indices.unsqueeze(-1)).squeeze(-1).bitwise_and_(packmask)
	return torch.zeros_like(_t_a_r_g_e_t_s_, dtype = torch.long).scatter_(
		-1, (path.t() - zero_padding).clamp(min = 0),
		torch.arange(len(path), device = log_alpha.device).expand(len(B), -1)
	)[:, 1::2]

def ctc_alignment_targets(log_probs, targets, input_lengths, target_lengths, blank = 0, ctc_loss = torch.nn.functional.ctc_loss, retain_graph = True):
	loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = blank, reduction = 'sum')
	probs = log_probs.exp()
	# to simplify API we inline log_softmax gradient, i.e. next two lines are equivalent to: grad_logits, = torch.autograd.grad(loss, logits, retain_graph = True). gradient formula explained at https://stackoverflow.com/questions/35304393/trying-to-understand-code-that-computes-the-gradient-wrt-to-the-input-for-logsof
	grad_log_probs, = torch.autograd.grad(loss, log_probs, retain_graph = retain_graph)
	grad_logits = grad_log_probs - probs * grad_log_probs.sum(dim = -1, keepdim = True)
	temporal_mask = (torch.arange(len(log_probs), device = input_lengths.device, dtype = input_lengths.dtype).unsqueeze(1) < input_lengths.unsqueeze(0)).unsqueeze(-1)
	return (probs * temporal_mask - grad_logits).detach()

def logadd(x0, x1, x2):
	# produces nan gradients in backward if -inf log-space zero element is used https://github.com/pytorch/pytorch/issues/31829
	return torch.logsumexp(torch.stack([x0, x1, x2]), dim = 0)
	
	# use if -inf log-space zero element is used
	#return LogsumexpFunction.apply(x0, x1, x2)
	
	# produces inplace modification error https://github.com/pytorch/pytorch/issues/31819
	#m = torch.max(torch.max(x0, x1), x2)
	#m = m.masked_fill(torch.isinf(m), 0)
	#res = (x0 - m).exp() + (x1 - m).exp() + (x2 - m).exp()
	#return res.log().add(m)

class LogsumexpFunction(torch.autograd.function.Function):
	@staticmethod
	def forward(self, x0, x1, x2):
		m = torch.max(torch.max(x0, x1), x2)
		m = m.masked_fill_(torch.isinf(m), 0)
		e0 = (x0 - m).exp_()
		e1 = (x1 - m).exp_()
		e2 = (x2 - m).exp_()
		e = (e0 + e1).add_(e2).clamp_(min = 1e-16)
		self.save_for_backward(e0, e1, e2, e)
		return e.log_().add_(m)

	@staticmethod
	def backward(self, grad_output):
		e0, e1, e2, e = self.saved_tensors
		g = grad_output / e
		return (g * e0, g * e1, g * e2)
