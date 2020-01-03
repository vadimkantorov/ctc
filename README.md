A primer on CTC implementation in pure Python PyTorch code


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
