# Person Re-identification via Attention Pyramid (APNet)
The official PyTorch code implementation for TIP20' Submission: "Person Re-identification via Attention Pyramid"

This repo only contains channel-wise attention(SE-Layer) implementation, to reproduce the result of spatial attention in our paper, please refer to [RGA-S](https://github.com/microsoft/Relation-Aware-Global-Attention-Networks) by Microsoft and simply change the attention agent. We also want to thank FastReid which is the codebase of our implementation.

## Introduction
Recently, attention mechanism has been widely used in the ReID system to facilitate high-performance identification and demonstrates the powerful representation ability by discovering discriminative regions and mitigating the misalignment. However, detecting the salient regions with the attention model is confronted with the dilemma to jointly capture both coarse and fine-grained clues, since the focus varies as the image scale changes. To address the above issue, we propose an effective attention pyramid networks (APNet) to jointly learn the attentions under different scales. Our attention pyramid imitates the process of human vi- sual perception which tends to notice the foreground person over the cluttered background, and further focus on the specific color of the shirt with a close observation. Please see the Figure1 below and our paper for the method detail.

We validate our method in Market1501, DukeMTMC and MSMT17 datasets, and our method shows a superior performance on all the datasets. Please check the Result section for the detaied quantity and quality result.

![image](https://github.com/Gutianpei/APNet/blob/main/images/github_main_graph.png)
Figure 1: The architecture of Attention Pyramid Networks (APNet). Our APNet adopts the “split-attend-merge-stack” principle, which first splits the feature maps into multiple parts, obtains the attention map of each part, and the attention map for current pyramid level is constructed by merging each attention map. Then in deeper pyramid level, we split the features into more fine-grained parts and learn the fine-grained attention guiding by coarse attentions. Finally, attentions with different granularities are stacked as attention pyramid and applies to original input feature by element-wise product.


## Requirements
- Python 3.6+
- PyTorch 1.5+
- CUDA 10.0+

Configuration other than the above setting is untested and we recommend to follow our setting.

To build all the dependency, please follow the instruction below.
```
conda create -n apnet python=3.7 -y
conda activate apnet
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
conda install ignite -c pytorch
git clone https://github.com/Gutianpei/APNet.git
pip install -r requirements.txt
```

To download the pretrained ResNet-50 model, please run the following command in your python console:
```
from torchvision.models import resnet50
resnet50(pretrained=True)
```
The model should be located in RESNET_PATH=```/home/YOURNAME/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth``` or ```/home/YOURNAME/.cache/torch/checkpoints/resnet50-19c8e357.pth```

### Downloading
- Market-1501
- DukeMTMC-reID 
- MSMT17
### Preparation
After downloading the datasets above, move them to the `Datasets/` folder in the project root directory, and rename dataset folders to 'market1501', 'duke' and 'msmt17' respectively. I.e., the `Datasets/` folder should be organized as:
```
|-- market1501
    |-- bounding_box_train
    |-- bounding_box_test
    |-- ...
|-- duke
    |-- bounding_box_train
    |-- bounding_box_test
    |-- ...
|-- msmt17
    |-- bounding_box_train
    |-- bounding_box_test
    |-- ...
```

## Usage
### Training
Change the PRETRAIN_PATH parameter in configs/default.yml to your RESNET_PATH
To train with different pyramid level, please edit LEVEL parareter in configs/default.yml
```
sh train.sh
```
### Evaluation
```
sh test.sh
```

## Result
|    Dataset     | Top-1 |  mAP  |
| :------------: | :---: | :---: |
|  Market-1501   | 96.2 | 90.5 |
| DukeMTMC-Re-ID | 90.4 | 81.5 |
|     MSMT17     | 83.7 | 63.5 |

![image](https://github.com/Gutianpei/APNet/blob/main/images/github_vis.png)
Figure 2: Visualizations of the attention maps with different pyramid level. We adopt the Grad-CAM to visualize the learned attention maps of our attention pyramid. For each sample, from left to right, we show the input image, attention of first level pyramid, attention of second level pyramid. We can observe that attentions in different pyramid levels capture the salient clues of different scales.
