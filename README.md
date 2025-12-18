<div align="center">
    
# Silence is Golden: Leveraging Adversarial Examples to Nullify Audio Control in LDM-based Talking-Head Generation [CVPR 2025]

<a href="https://yuangan.github.io/"><strong>Yuan Gan</strong></a>
·
<a href="https://scholar.google.com/citations?user=kQ-FWd8AAAAJ&hl=zh-CN&oi=ao"><strong>Jiaxu Miao</strong></a>
·
<a><strong>Yunze Wang</strong></a>
.
<a href="https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en"><strong>Yi Yang</strong></a>

<a href="https://github.com/yuangan/Silencer"><img src="./figures/intro.png" style="width: 1225px;"></a>

</div>
<div align="justify">

**Abstract**: Advances in talking-head animation based on Latent Diffusion Models (LDM) enable the creation of highly realistic, synchronized videos. These fabricated videos are indistinguishable from real ones, increasing the risk of potential misuse for scams, political manipulation, and misinformation. Hence, addressing these ethical concerns has become a pressing issue in AI security. Recent proactive defense studies focused on countering LDM-based models by adding perturbations to portraits. However, these methods are ineffective at protecting reference portraits from advanced image-to-video animation. The limitations are twofold: 1) they fail to prevent images from being manipulated by audio signals, and 2) diffusion-based purification techniques can effectively eliminate protective perturbations. To address these challenges, we propose **Silencer**, a two-stage method designed to proactively protect the privacy of portraits. First, a nullifying loss is proposed to ignore audio control in talking-head generation. Second, we apply anti-purification loss in LDM to optimize the inverted latent feature to generate robust perturbations. Extensive experiments demonstrate the effectiveness of **Silencer** in proactively protecting portrait privacy. We hope this work will raise awareness among the AI security community regarding critical ethical issues related to talking-head generation techniques.

## Silencer-I

### Environment Setup

```
cd Silencer-I
conda create -n silencer python=3.10
conda activate silencer
pip install -r requirements.txt
```
Download the pretrained models from [hallo](https://github.com/fudan-generative-vision/hallo#-download-pretrained-models).

Download our preprocessed CelebA-HQ and TH1KH datasets from [Yandex](https://disk.yandex.com/d/OLe6c-cjGWiPgw).

These files should be organized as follows:
```
./Silencer-I/
|-- ...
|-- pretrained_models/
|   |-- ...
|-- th1kh/
|   |-- ...
|-- celebahq_512_dataset/
|   |-- ...
```

Change the path of ```pretrained_models``` into your own absolute path. Then install Hallo by executing the following command in ```Silencer-I```.

```
pip install .
```

PS: It may change the version of protobuf, and you should keep the protobuf==3.20.3 with ```pip install protobuf==3.20.3```.

### Run Silencer-I
In ```Silencer-I```, you can run the following command to proactively protect images in CelebA-HQ and TH1KH datasets.
```
CUDA_VISIBLE_DEVICES=0 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' \
    attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' \
    attack.min_timesteps=200

CUDA_VISIBLE_DEVICES=0 python protect/protect_hallo.py attack.img_path='th1kh/th1kh_imgs_100' \
    attack.output_path='protect/out_th1kh_512/' attack.mode='hallo' attack.g_mode='-' \
    attack.min_timesteps=200
```
### Test Protected Portraits of Silencer-I
Refer to [test_hallo_celebahq_hallo-.py](https://github.com/yuangan/Silencer/blob/main/Silencer-I/test_hallo_celebahq_hallo-.py) and [test_hallo_th1kh_hallo-.py](https://github.com/yuangan/Silencer/blob/main/Silencer-I/test_hallo_th1kh_hallo-.py).

Useage examples are in [test_hallo_celebahq.sh](https://github.com/yuangan/Silencer/blob/main/Silencer-I/test_hallo_th1kh.sh) and [test_hallo_th1kh.sh](https://github.com/yuangan/Silencer/blob/main/Silencer-I/test_hallo_th1kh.sh).

For example:
```
python test_hallo_th1kh_hallo-.py 0 0 &
python test_hallo_th1kh_hallo-.py 1 0 &
python test_hallo_th1kh_hallo-.py 2 0 &
python test_hallo_th1kh_hallo-.py 3 0 &
```

## Evaluation

Please refer to the LRW evaluation process of EAT [here](https://github.com/yuangan/evaluation_eat#evaluation-instructions).

1. Generate ground truth videos with original portraits and the hallo model.
2. Generate ground truth for evaluation with GT videos.
3. Evaluate the generated videos with protected portraits.

## Silencer-II

### Environment Setup

Based on the silencer, you need to install additional packages from [DiffAttack](https://github.com/WindVChen/DiffAttack).

You can refer to my environment [here](https://drive.google.com/file/d/1roSLMaeerhI_Wu-wc34s0n9lL6G4BzKN/view?usp=sharing). Untar the file into the ```envs``` folder of conda/mamba, then install hallo and protobuf.

Download the attack code from [here](https://drive.google.com/file/d/1eYNMZXpthsLbkNR4y7AQqLXRb4PFD0-R/view?usp=sharing) and untar it into Silencer-II.

### Run Silencer-II


In ```Silencer-II/DiffAttack```, change the ```root_hallo``` and ```the ouput path of Silencer-I``` into your own path in ``` main_hallo_attnloss.py ```.

Then you can run the following command for protection:

```
CUDA_VISIBLE_DEVICES=0 FORCE_MEM_EFFICIENT_ATTN=1 python  main_hallo_attnloss.py --model_name resnet \
    --save_dir ./adv_out/celebahq_adamw_iter200_1e-2_hallo1_512_19_+mse_t200_s100_fmask_10_100 \
    --images_root ../../Silencer-I/celebahq_512_dataset/celebahq_512/ --attack_mode 'hallo' --g_mode '-'\
    --res 512 --iterations 200 --attack_loss_weight 1 --attack_mse_loss_weight 10 \
    --cross_attn_loss_weight 0 --self_attn_loss_weight 0 --start_step 19 \
    --dataset_name celebahq --use_face_mask 1 \
    --index_use_face_mask 100 --part [0-3]


CUDA_VISIBLE_DEVICES=0 FORCE_MEM_EFFICIENT_ATTN=1 python  main_hallo_attnloss.py --model_name resnet \
    --save_dir ./adv_out/th1kh_adamw_iter200_1e-2_hallo1_512_19_+mse_t200_s100_fmask_10_100_20251217   \
    --images_root ../../Silencer-I/th1kh/th1kh_imgs_100/ --attack_mode 'hallo' --g_mode '-'\
    --res 512 --iterations 200 --attack_loss_weight 1 --attack_mse_loss_weight 10  \
    --cross_attn_loss_weight 0 --self_attn_loss_weight 0 --start_step 19 \
    --dataset_name th1kh --use_face_mask 1 \
    --index_use_face_mask 100 --part [0-3]
```

### Test Protected Portraits of Silencer-II
Refer to [test_hallo_celebahq_diff_hallo-.py](https://github.com/yuangan/Silencer/blob/main/Silencer-I/test_hallo_celebahq_diff_hallo-.py) and [test_hallo_th1kh_diff_hallo-.py](https://github.com/yuangan/Silencer/blob/main/Silencer-I/test_hallo_th1kh_diff_hallo-.py).

Usage examples are the same as Silencer-I:

```
python test_hallo_th1kh_diff_hallo-.py 0 0 &
python test_hallo_th1kh_diff_hallo-.py 1 0 &
python test_hallo_th1kh_diff_hallo-.py 2 0 &
python test_hallo_th1kh_diff_hallo-.py 3 0 &
```

## Citation
If you find our work useful, please cite it as follows:

```
@inproceedings{gan2025silence,
  title={Silence is Golden: Leveraging Adversarial Examples to Nullify Audio Control in LDM-based Talking-Head Generation},
  author={Gan, Yuan and Miao, Jiaxu and Wang, Yunze and Yang, Yi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={13434--13444},
  year={2025}
}
```

## Acknowledgement
We acknowledge these works for their public code: [Diff-Protect](https://github.com/xavihart/Diff-Protect), [DiffAttack](https://github.com/WindVChen/DiffAttack), [Hallo](https://github.com/fudan-generative-vision/hallo), and so on.

