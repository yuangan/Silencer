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

## Setup & Preparation

### Environment Setup
## TODO:
- [ ] Environment setup
- [ ] Release the code of Silencer.
- [ ] Release the metrics of Silencer.
