# XGrad-CAM implementation in Pytorch 

code for the paper:
### Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs

To be presented at **BMVC 2020**,
<br>
Authors:
<br>
Ruigang Fu,
Qingyong Hu,
Xiaohu Dong,
Yulan Guo,
Yinghui Gao and
Biao Li,
<br>

----------

### XGrad-cam.py
The main difference between XGrad-CAM and Grad-CAM located at line 113 - line119:
#####  Grad-CAM 
`weights = np.mean(grads_val, axis=(2, 3))[0, :]`
#####  XGrad-CAM
`X_weights = np.sum(grads_val[0, :] * target, axis=(1, 2))`

`X_weights = X_weights / (np.sum(target, axis=(1, 2)) + 1e-6)`

Usage: `python XGrad-cam.py --image-path <path_to_image>`

Results:

![Dog](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/dog.jpg?raw=true) ![Cat](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cat.jpg?raw=true)

----------

### Proof_verify.py
This is a script of experimental proof for our claim that given an arbitrary layer in ReLU-CNNs, there
exists a specific equation between the class score and the feature maps of the layer.

Usage: `python proof.py`

The result will show that `class_score-gradients*feature-bias_term=0`

----------

These codes are based on https://github.com/jacobgil/pytorch-grad-cam.
Thanks to Jacob Gildenblat for the beautiful original code.
