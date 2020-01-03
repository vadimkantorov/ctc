import torch
import torch.nn.functional as F

# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/LossCTC.cpp#L37
# https://github.com/skaae/Lasagne-CTC/blob/master/ctc_cost.py#L162

def ctc_alignment_targets(log_probs, targets, input_lengths, target_lengths, logits, blank = 0):
	ctc_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = blank, reduction = 'sum')
	ctc_grad, = torch.autograd.grad(ctc_loss, (logits,), retain_graph = True)
	temporal_mask = (torch.arange(len(log_probs), device = input_lengths.device, dtype = input_lengths.dtype).unsqueeze(1) < input_lengths.unsqueeze(0)).unsqueeze(-1)
	return (log_probs.exp() * temporal_mask - ctc_grad).detach()

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank : int = 0, reduction : str = 'none', alignment : bool = False):
	B = torch.arange(len(targets), device = input_lengths.device)
	targets_ = torch.cat([targets, targets[:, :1]], dim = -1)
	targets_ = torch.stack([torch.full_like(targets_, blank), targets_], dim = -1).flatten(start_dim = -2)
	diff_labels = torch.cat([torch.as_tensor([[False, False]], device = targets.device).expand(len(B), -1), targets_[:, 2:] != targets_[:, :-2]], dim = 1)
	
	zero, zero_padding = torch.tensor(float('-inf'), device = log_probs.device, dtype = log_probs.dtype), 2
	log_probs_ = log_probs.gather(-1, targets_.expand(len(log_probs), -1, -1))
	log_alpha = torch.full((len(log_probs), len(B), zero_padding + targets_.shape[-1]), zero, device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[0, :, zero_padding + 0] = log_probs[0, :, blank]
	log_alpha[0, :, zero_padding + 1] = log_probs[0, B, targets_[:, 1]]
	for t in range(1, len(log_probs)):
		log_alpha[t, :, 2:] = log_probs_[t] + logadd(log_alpha[t - 1, :, 2:], log_alpha[t - 1, :, 1:-1], torch.where(diff_labels, log_alpha[t - 1, :, :-2], zero))
	#log_alpha = [torch.full((len(B), zero_padding + targets_.shape[-1]), zero, device = log_probs.device, dtype = log_probs.dtype)]
	#log_alpha[0][:, zero_padding + 0] = log_probs[0, :, blank]
	#log_alpha[0][:, zero_padding + 1] = log_probs[0, B, targets_[:, 1]]
	#for t in range(1, len(log_probs)):
	#	log_alpha_ = torch.full_like(log_alpha[-1], zero, device = log_probs.device, dtype = log_probs.dtype)
	#	log_alpha_[:, zero_padding:] = log_probs_[t] + logadd(log_alpha[-1][:, 2:], log_alpha[-1][:, 1:-1], torch.where(diff_labels, log_alpha[- 1][:, :-2], zero))
	#	log_alpha.append(log_alpha_)
	#log_alpha = torch.stack(log_alpha)

	l1l2 = log_alpha[input_lengths - 1, B].gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)) 
	loss = -torch.logsumexp(l1l2, dim = -1)
	if not alignment:
		return loss
	
	path = torch.zeros(len(log_alpha), len(B), device = log_alpha.device, dtype = torch.int64)
	path[input_lengths - 1, B] = zero_padding + 2 * target_lengths - 1 + l1l2.max(dim = -1).indices
	for t in range(len(path) - 1, 1, -1):
		indices = path[t]
		indices_ = torch.stack([(indices - 2) * diff_labels[B, (indices - zero_padding).clamp(min = 0)], (indices - 1).clamp(min = 0), indices], dim = -1)
		path[t - 1] += (indices - 2 + log_alpha[t - 1, B].gather(-1, indices_).max(dim = -1).indices).clamp(min = 0)
	return torch.zeros_like(log_alpha).scatter_(-1, path.unsqueeze(-1), 1.0)[..., 3::2]

def logadd(x0, x1, x2):
	# equivalent to the slower version:
	return torch.logsumexp(torch.stack([x0, x1, x2]), dim = 0)
	m = torch.max(torch.max(x0, x1), x2)
	m.masked_fill_(m == float('-inf'), 0)
	res = (x0 - m).exp_() + (x1 - m).exp_() + (x2 - m).exp_()
	return res.log_().add_(m)

if __name__ == '__main__':
	import time
	import matplotlib.pyplot as plt
	
	T, B, C = 16, 4, 2
	t = 4#T // 2 - 4
	device = 'cuda'
	seed = 1
	rtol = 1e-3
	for set_seed in [torch.manual_seed] + ([torch.cuda.manual_seed_all] if device == 'cuda' else []):
		set_seed(seed)

	logits = torch.randn(T, B, C, device = device).requires_grad_()
	log_probs = logits.log_softmax(dim = -1)
	targets = torch.randint(1, C, (B, t), dtype = torch.long, device = device)
	input_lengths = torch.full((B,), T, dtype = torch.long, device = device)
	target_lengths = torch.full((B,), t, dtype = torch.long, device = device)

	tictoc = lambda: (device == 'cuda' and torch.cuda.synchronize()) or time.time()
	tic = tictoc()
	builtin_ctc = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	print('Built-in CTC loss', tictoc() - tic)
	ce_ctc = (-ctc_alignment_targets(log_probs, targets, input_lengths, target_lengths, blank = 0, logits = logits) * log_probs)

	tic = tictoc()
	custom_ctc = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	print('Custom CTC loss', tictoc() - tic)
	
	builtin_ctc_grad, = torch.autograd.grad(builtin_ctc.sum(), logits, retain_graph = True)
	custom_ctc_grad, = torch.autograd.grad(custom_ctc.sum(), logits, retain_graph = True)
	ce_ctc_grad, = torch.autograd.grad(ce_ctc.sum(), logits, retain_graph = True)

	print('Device:', device)
	print('Log-probs shape:', 'x'.join(map(str, log_probs.shape)))
	print('Custom loss matches:', torch.allclose(builtin_ctc, custom_ctc, rtol = rtol))
	print('Grad matches:', torch.allclose(builtin_ctc_grad, custom_ctc_grad, rtol = rtol))
	print('CE grad matches:', torch.allclose(builtin_ctc_grad, ce_ctc_grad, rtol = rtol))
	
	#P = ctc_alignment(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	#plt.imshow(P[1].t(), origin = 'lower', aspect = 'auto')
	#plt.savefig('data/alignment.jpg')
