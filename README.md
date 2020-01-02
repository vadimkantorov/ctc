A primer on CTC implementation in pure Python PyTorch code

### Forward pass times (averaged over 100 runs)

Basically the custom impl runs 10x slower than the C++/CUDA version in PyTorch.

```
Device: cpu
Log-probs shape: 256x128x32
Loss matches: True
Built-in CTC loss 0.03467770665884018
Custom CTC loss 0.4271008372306824

Device: cuda
Log-probs shape: 256x128x32
Loss matches: True
Built-in CTC loss 0.008279071189463139
Custom CTC loss 0.07046514749526978
```
