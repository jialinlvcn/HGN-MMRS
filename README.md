<p align="center">
  <h1 align="center">Hierarchical Gated Network for Multimodal Remote Sensing Imagery Classification with Limited Data (IGARSS'2025)</h1>
  <p align="center">
    <a href="https://github.com/jialinlvcn"><strong>Jialin Lyu</strong></a>
    &nbsp;&nbsp;
    <strong>Zhunga Liu</strong></a>
    &nbsp;&nbsp;
    <a href="https://github.com/fuyimin96"><strong>Yimin Fu</strong></a>
  </p>
  <br>

Pytorch implementation for "**Hierarchical Gated Network for Multimodal Remote Sensing Imagery Classification with Limited Data**"

> **Abstract:** *The effective fusion of multimodal data can significantly advance Earth observation-related tasks. However, the collection and annotation of spatial-aligned remote sensing (RS) data across different modalities are resource-intensive and laborious. This hinders the exploitation of complementary information between modalities, leading to unsatisfactory performance in data-limited scenarios. To solve this problem, we propose a hierarchical gated network (HGN) for multimodal RS imagery classification. Specifically, HGN incrementally integrates multimodal information through gating mechanisms at both the feature and decision levels. For the input data of each modality, the corresponding feature representations are extracted in parallel. During this process, feature gates between different convolution blocks are utilized to control the fusion flow across single-modal branches. Finally, logits derived from single-modal and fused representations are selectively combined through the decision gate. This hierarchical fusion framework enables adaptive cross-layer interactions between different modalities, thereby facilitating the exploitation of complementary information. Thorough experiments on the Houston2013 and Augsburg multimodal datasets show that HGN achieves state-of-the-art performance.*

:hammer: **We integrated dataset construction, delineation and mapping for pixel-level classification of multimodal remotely sensed data, and visualization of the results into a toolbox to facilitate subsequent studies. The toolbox is also applicable to unimodal hyperspectral data.**

<p align="center">
    <img src=./figtures/Overview.png width="800">
</p>

## Requirements

To install the requirements, you can run the following in your environment first:

```
pip install uv
UV_TORCH_BACKEND=auto uv pip install -r requirements.txt
```

To run the code with CUDA properly, you can comment out torch and torchvision in requirement.txt, and install the appropriate version of torch and torchvision according to the instructions on [PyTorch](https://pytorch.org/get-started/locally/).

## Datasets

For the dataset used in this paper, please download the following datasets `Houston2013`, `Trento` and `Augsburg` dataset move them to ```./data```.

You can get them at https://drive.google.com/drive/folders/19_8uyoBNACHkhhWwU3Mi7bXMLmSuQx25?usp=sharing

## Run the code
You can also run the code with the following command:

First run the train.py to train the model with multimodal remote sensing dataset.

```sh
python train.py --lr 2e-4 --data_set Houston2013 --model HGN --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints

python train.py --lr 2e-4 --data_set Augsburg --model HGN --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
```

Then you can evaluate and visualize your trained network.

```sh
python test.py --data_set Augsburg --model HGN --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints

python test.py --data_set Houston2013 --model HGN --device cuda:0 --data_dir ./data --model_save_dir ./checkpoints
```

All commands are placed in the `./scripts`. You can easily reproduce all the results of your experiments by:

```sh
chmod 777 scripts/test.sh
scripts/test.sh
chmod 777 scripts/train_Augsburg.sh
scripts/test.sh
chmod 777 scripts/train_Houston2013.sh
scripts/test.sh
```

## Results
We visualized all the experimental results. For more information on how to plot them please refer to `notebook/load_data.ipynb` and `notebook/test_and_draw_as_paper.ipynb`.

<p align="center">
    <img src=./figtures/Augsburg-result.png width="800">
</p>

<p align="center">
    <img src=./figtures/Houston2013-result.png width="800">
</p>

## Citation
If you find our work and this repository useful. Please consider giving a star :star: and citation.
<!-- ```bibtex
@inproceedings{lyu2025hierarchical,
  title={Hierarchical Gated Network for Multimodal Remote Sensing Imagery Classification with Limited Data},
  author={Lyu, Jialin, and Liu, Zhunga and Fu, Yimin},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium},
  year={2025},
  publisher={IEEE}
}
``` -->