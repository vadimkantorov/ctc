import torch
import torch.nn.functional as F

# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/LossCTC.cpp#L37
# https://github.com/skaae/Lasagne-CTC/blob/master/ctc_cost.py#L162
	
	#log_alpha = [torch.full((len(B), zero_padding + targets_.shape[-1]), zero, device = log_probs.device, dtype = log_probs.dtype)]
	#log_alpha[0][:, zero_padding + 0] = log_probs[0, :, blank]
	#log_alpha[0][:, zero_padding + 1] = log_probs[0, B, targets_[:, 1]]
	#for t in range(1, len(log_probs)):
	#	log_alpha_ = torch.full_like(log_alpha[-1], zero, device = log_probs.device, dtype = log_probs.dtype)
	#	log_alpha_[:, zero_padding:] = log_probs_[t] + logadd(log_alpha[-1][:, 2:], log_alpha[-1][:, 1:-1], torch.where(diff_labels, log_alpha[- 1][:, :-2], zero))
	#	log_alpha.append(log_alpha_)
	#log_alpha = torch.stack(log_alpha)

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
	log_alpha[1:,:, zero_padding:] = log_probs.gather(-1, targets_.expand(len(log_probs), -1, -1))[1:]
	for t in range(1, len(log_probs)):
		log_alpha[t, :, 2:] += torch.logsumexp(torch.stack([log_alpha[t - 1, :, 2:], log_alpha[t - 1, :, 1:-1], torch.where(diff_labels, log_alpha[t - 1, :, :-2], zero)]), dim = 0)

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

def ctc_alignment_targets(logits, targets, input_lengths, target_lengths, blank):
	log_probs = F.log_softmax(logits, dim = 1)
	ctc_loss = F.ctc_loss(log_probs.permute(2, 0, 1), targets, input_lengths, target_lengths, blank = blank, reduction = 'sum')
	ctc_grad, = torch.autograd.grad(ctc_loss, (logits,), retain_graph = True)
	temporal_mask = (torch.arange(logits.shape[-1], device = input_lengths.device, dtype = input_lengths.dtype).unsqueeze(0) < input_lengths.unsqueeze(1))[:, None, :]
	return (log_probs.exp() * temporal_mask - ctc_grad).detach()

def logadd(x0, x1, x2):
	# equivalent to the slower version:
	# return torch.logsumexp(torch.stack([x0, x1, x2]), dim = 0)
	m = torch.max(torch.max(x0, x1), x2)
	m.masked_fill_(m == float('-inf'), 0)
	res = (x0 - m).exp_() + (x1 - m).exp_() + (x2 - m).exp_()
	return res.log_().add_(m)

if __name__ == '__main__':
	import time
	import matplotlib.pyplot as plt
	
	T, B, C = 4, 1, 2
	t = 1#T // 2 - 4
	N = 100
	device = 'cuda'
	seed = 1
	
	for set_seed in [torch.manual_seed] + ([torch.cuda.manual_seed_all] if device == 'cuda' else []):
		set_seed(seed)
	#torch.autograd.set_detect_anomaly(True)

	logits = torch.randn(T, B, C, device = device).requires_grad_()
	log_probs = logits.log_softmax(dim = -1).requires_grad_()
	targets = torch.randint(1, C, (B, t), dtype = torch.long, device = device)
	input_lengths = torch.full((B,), T, dtype = torch.long, device = device)
	target_lengths = torch.full((B,), t, dtype = torch.long, device = device)

	builtin_ctc = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	custom_ctc = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	builtin_ctc_grad, = torch.autograd.grad(builtin_ctc.sum(), log_probs)
	custom_ctc_grad, = torch.autograd.grad(custom_ctc.sum(), log_probs)

	print('Device:', device)
	print('Log-probs shape:', 'x'.join(map(str, log_probs.shape)))
	print('Loss matches:', torch.allclose(builtin_ctc, custom_ctc))
	print('Grad matches:', torch.allclose(builtin_ctc_grad, custom_ctc_grad))

	import sys; sys.exit(0)
	for warmup_iteration in range(N):
		F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	
	tictoc = lambda: (device == 'cuda' and torch.cuda.synchronize()) or time.time()
	fwd_builtin, bwd_builtin, fwd_custom, bwd_custom = [], [], [], []
	for i in range(N):
		tic = tictoc()
		loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
		fwd_builtin.append(tictoc() - tic)
		
		tic = tictoc()
		loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
		fwd_custom.append(tictoc() - tic)

	print('Built-in CTC loss', float(torch.FloatTensor(fwd_builtin).mean()))
	print('Custom CTC loss', float(torch.FloatTensor(fwd_custom).mean()))
	
	#P = ctc_alignment(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	#plt.imshow(P[1].t(), origin = 'lower', aspect = 'auto')
	#plt.savefig('data/alignment.jpg')
	
	ctc_loss_via_crossentropy = (-ctc_alignment_targets(logits, targets, input_lengths, target_lengths, blank = 0) * log_probs).sum()
