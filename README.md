A primer on CTC implementation in pure Python PyTorch code. Features:
- CTC impl is in Python and its only loop is over time steps (parallelizes over batch and symbol dimensions)
- Gradients are computed via PyTorch autograd instead of a separate beta computation
- Viterbi path useful for forced alignment
- Get alignment targets out of any CTC impl, so that label smoothing or reweighting can be applied [1, 2]

### Very rough time measurements
```
Device: cuda
Log-probs shape (batch X time X channels): 128x256x32
Built-in CTC loss fwd 0.009581804275512695 bwd 0.012355327606201172
Custom CTC loss fwd 0.09775996208190918 bwd 0.1494584083557129
Custom loss matches: True
Grad matches: True
CE grad matches: True

Device: cpu
Log-probs shape (batch X time X channels): 128x256x32
Built-in CTC loss fwd 0.017041444778442383 bwd 0.23205327987670898
Custom CTC loss fwd 0.3748452663421631 bwd 4.206061363220215
Custom loss matches: True
Grad matches: True
CE grad matches: True
```

### Alignment image example
![](https://user-images.githubusercontent.com/1041752/71736894-8615e800-2e52-11ea-81cb-cb95b92175c6.png)

### References
1. A Novel Re-weighting Method for Connectionist Temporal Classification; Li et al; https://arxiv.org/abs/1904.10619
2. Focal CTC Loss for Chinese Optical Character Recognition on Unbalanced Datasets"; Feng et al; https://www.hindawi.com/journals/complexity/2019/9345861/
3. Improved training for online end-to-end speech recognition systems; Kim et al; https://arxiv.org/abs/1711.02212
4. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks; Graves et all; 
https://www.cs.toronto.edu/~graves/icml_2006.pdf
5. Sequence Modeling With CTC, Hannun et al, https://distill.pub/2017/ctc/
