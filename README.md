# Learning Efficient Online 3D Bin Packing on Packing Configuration Trees 

**This repository is fully published now, all kinds of feedback and any contribution is welcome!** 

We propose to enhance the practical applicability of online 3D bin packing problem (BPP) via learning on a hierarchical packing configuration tree which makes the deep reinforcement learning (DRL) model easy to deal with practical constraints and well-performing even with continuous solution space.
 Compared to our previous work, the advantages of this repo are:
- [x] Container (bin) size and item sizes can be set arbitrarily.
- [x] Continuous online 3D-BPP is allowed and the continuous environment is provided.
- [x] Algorithms to approximate stability are provided ([see our other work](https://arxiv.org/abs/2108.13680v2)). 
- [x] Better performance and the ability to account for more complex constraints.
- [x] More adequate heuristic baselines for domain development.
- [x] More stable training.

See these links for video demonstration: [YouTube](https://www.youtube.com/watch?v=duWgTskKwws), [bilibili](https://www.bilibili.com/video/BV1rU4y1R74S/?vd_source=b1e4277847248c95062cf16ab3b58e73)

If you are interested, please star this repo! 


![PCT](images/packingtree2D.png)

## Paper
For more details, please see our paper [Learning Efficient Online 3D Bin Packing on Packing Configuration Trees](https://openreview.net/forum?id=bfuGjlCwAq) which has been accepted at [ICLR 2022](https://iclr.cc/Conferences/2022). If this code is useful for your work, please cite our paper:

```
@inproceedings{
zhao2022learning,
title={Learning Efficient Online 3D Bin Packing on Packing Configuration Trees},
author={Hang Zhao and Yang Yu and Kai Xu},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=bfuGjlCwAq}
}
``` 


## Dependencies
* NumPy
* gym
* Python>=3.7
* [PyTorch](http://pytorch.org/) >=1.7
* My suggestion: Python == 3.7, gym==0.13.0, torch == 1.10, OS: Ubuntu 16.04
## Quick start

For training online 3D-BPP on setting 2 (mentioned in our paper) with our PCT method and the default arguments:
```bash
python main.py 
```
The training data is generated on the fly. The training logs (tensorboard) are saved in './logs/runs'. Related file backups are saved in './logs/experiment'.

## Usage

### Data description

Describe your 3D container size and 3D item size in 'givenData.py'
```
container_size: A vector of length 3 describing the size of the container in the x, y, z dimension.
item_size_set:  A list records the size of each item. The size of each item is also described by a vector of length 3.
```
### Dataset
You can download the prepared dataset from [here](https://drive.google.com/drive/folders/1QLaLLnpVySt_nNv0c6YetriHh0Ni-yXY?usp=sharing).
The dataset consists of 3000 randomly generated trajectories, each with 150 items. The item is a vector of length 3 or 4, the first three numbers of the item represent the size of the item, the fourth number (if any) represents the density of the item.

### Model
We provide [pretrained models](https://drive.google.com/drive/folders/14PC3aVGiYZU5AaGdNM9YOVdp8pPiZ3fe?usp=sharing) trained using the EMS scheme in a discrete environment, where the bin size is (10,10,10) and the item size range from 1 to 5.

### Training

For training online 3D BPP instances on setting 1 (80 internal nodes and 50 leaf nodes) nodes:
```bash
python main.py --setting 1 --internal-node-holder 80 --leaf-node-holder 50
```
If you want to train a model that works on the **continuous** domain, add '--continuous', don't forget to change your problem in 'givenData.py':
```bash
python main.py --continuous --setting 1 --internal-node-holder 80 --leaf-node-holder 50
```
#### Warm start
You can initialize a run using a pretrained model:
```bash
python main.py --load-model --model-path path/to/your/model
```

### Evaluation
To evaluate a model, you can add the `--evaluate` flag to `evaluation.py`:
```bash
python evaluation.py --evaluate --load-model --model-path path/to/your/model --load-dataset --dataset-path path/to/your/dataset
```
### Heuristic
Running heuristic.py for test heuristic baselines, the source of the heuristic algorithm has been marked in the code:

Running heuristic on setting 1 （discrete） with LASH method:
```
python heuristic.py --setting 1 --heuristic LSAH --load-dataset  --dataset-path setting123_discrete.pt
```

Running heuristic on setting 2 （continuous） with OnlineBPH method:
```
python heuristic.py --continuous --setting 2 --heuristic OnlineBPH --load-dataset  --dataset-path setting2_continuous.pt
```

### Help
```bash
python main.py -h
python evaluation.py -h
python heuristic.py -h
```

### License
```
This source code is released only for academic use. Please do not use it for commercial purpose without authorization of the author.
```
