# Learning Efficient Online 3D Bin Packing on Packing Configuration Trees 

**This repository is being continuously updated, please stay tuned!** 

Any code contribution is welcome!  **I am also looking for high-quality academic cooperation.** If you are interested or have any problems, please contact me at alexfrom0815@gmail.com.

We propose to enhance the practical applicability of online 3D-BPP via learning on a hierarchical packing configuration tree which makes the DRL model easy to deal with practical constraints and well-performing even with continuous solution space.

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

* Python>=3.7
* NumPy
* [PyTorch](http://pytorch.org/)>=1.7
* gym

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
We provide [pretrained models](https://drive.google.com/drive/folders/14PC3aVGiYZU5AaGdNM9YOVdp8pPiZ3fe?usp=sharing) trained using the EMS scheme in a discrete environment, where the bin size is (10,10,10) and item size range from 1 to 5.

### Training

For training online 3D BPP instances on setting 1 (80 internal nodes and 50 leaf nodes) nodes:
```bash
python main.py --setting 1 --internal-node-holder 80 --leaf-node-holder 50
```
If you want to train a model works on **continuous** domain, add '--continuous', don't forget to change your problem in 'givenData.py':
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

### TODO (This code will be fully published by March 2022)
```
1. Add other leaf node expansion schemes.
2. Feedback of various bugs is welcome.
```
