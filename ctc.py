import time
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/LossCTC.cpp#L37
# https://github.com/skaae/Lasagne-CTC/blob/master/ctc_cost.py#L162

def logsumexp(x, dim):
	#print(x.shape)
	return torch.logsumexp(x, dim = dim)
	#m = x.max(dim = dim, keepdim = True).values
	#return (x - m).exp_().sum(dim = dim, keepdim = True).log_().add_(m).squeeze(dim)

#@torch.jit.script
def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank : int = 0, reduction : str = 'none', alignment : bool = False, zero : float = float('-inf')):#, add = torch.logsumexp):# add = (lambda t, dim = None: torch.max(t, dim = dim).values), ):#,
	targets_ = torch.cat([targets, targets[:, :1]], dim = -1)
	targets_ = torch.stack([torch.full_like(targets_, blank), targets_], dim = -1).flatten(start_dim = -2)
	B = torch.arange(len(targets), device = input_lengths.device)
	zero_padding = 2
	log_alpha = torch.full((len(B), len(log_probs), zero_padding + targets_.shape[-1]), zero, device = log_probs.device, dtype = log_probs.dtype)
	log_alpha[:, 0, zero_padding + 0] = log_probs[0, :, blank]
	log_alpha[:, 0, zero_padding + 1] = log_probs[0, B, targets_[:, 1]]
	log_alpha[:, 1:, zero_padding:] = log_probs.gather(-1, targets_.expand(len(log_probs), -1, -1))[1:].permute(1, 0, 2)
	
	diff_labels = torch.cat([torch.as_tensor([[False, False]], device = targets.device).expand(len(B), -1), targets_[:, 2:] != targets_[:, :-2]], dim = 1)
	zero = torch.tensor(zero, device = log_probs.device, dtype = log_probs.dtype)
	for t in range(1, len(log_probs)):
		log_alpha[:, t, zero_padding:] += logsumexp(torch.stack([log_alpha[:, t - 1, 2:], log_alpha[:, t - 1, 1:-1], torch.where(diff_labels, log_alpha[:, t - 1, :-2], zero)], dim = 0), dim = 0)

	l1l2 = log_alpha[B, input_lengths - 1].gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1)) 
	loss = -torch.logsumexp(l1l2, dim = -1)
	if not alignment:
		return loss

	path = torch.zeros(len(B), len(log_probs), device = log_probs.device, dtype = torch.int64)
	path[B, input_lengths - 1] = zero_padding + 2 * target_lengths - 1 + l1l2.max(dim = -1).indices
	for t in range(len(log_probs) - 1, 1, -1):
		indices = path[:, t]
		indices_ = torch.stack([(indices - 2) * diff_labels[B, (indices - zero_padding).clamp(min = 0)], (indices - 1).clamp(min = 0), indices], dim = -1)
		path[:, t - 1] += (indices - 2 + log_alpha[B, t - 1].gather(-1, indices_).max(dim = -1).indices).clamp(min = 0)
	return torch.zeros_like(log_alpha).scatter_(-1, path.unsqueeze(-1), 1.0)[..., 3::2]

if __name__ == '__main__':
	T, t, B, C = 128, 60, 16, 20
	
	torch.manual_seed(1)
	log_probs = torch.randn(T, B, C).log_softmax(dim = -1)
	targets = torch.randint(1, C, (B, t), dtype = torch.long)
	input_lengths = torch.full((B,), T, dtype = torch.long)
	target_lengths = torch.full((B,), t, dtype = torch.long)

	tictoc = lambda: (torch.cuda.is_available and torch.cuda.synchronize()) or time.time()

	loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	loss_ = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	print(loss)
	print(loss_)

	N = 10

	# warmup
	for i in range(N):
		F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')

	t, t_ = [], []
	for i in range(N):
		tic = tictoc()
		loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
		t.append(tictoc() - tic)
		
		tic = tictoc()
		#with torch.autograd.profiler.profile() as prof:
		loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
		#print(prof.key_averages().table(sort_by="self_cpu_time_total"))
		t_.append(tictoc() - tic)

	print('Built-in CTC loss', float(torch.FloatTensor(t).mean()))
	print('Custom CTC loss', float(torch.FloatTensor(t_).mean()))
	
	#P = ctc_alignment(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
	#plt.imshow(P[1].t(), origin = 'lower', aspect = 'auto')
	#plt.savefig('data/alignment.jpg')
