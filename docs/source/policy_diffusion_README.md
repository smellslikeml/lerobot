## Paper

https://diffusion-policy.cs.columbia.edu

## Ambient loss masking — co-training on suboptimal data

When co-training on a mixture of clean and suboptimal/out-of-distribution
demonstrations, the diffusion loss can be restricted per-sample to only the
diffusion times where each sample's signal is trustworthy. This implements the
core mechanism of *Ambient Diffusion Policy: Imitation Learning from Suboptimal
Data in Robotics* ([arXiv:2606.12365](https://arxiv.org/abs/2606.12365)):
robot action data follows a spectral power law, so high diffusion times carry
global structure and low diffusion times carry local detail, while suboptimal
demonstrations corrupt the mid-frequency band.

Enable it via the policy config:

```python
DiffusionConfig(
    use_ambient_loss_masking=True,
    ambient_quality_key="action_quality",  # per-sample score in [0, 1] read from the batch
    ambient_mask_mode="band",              # "band" suppresses mid-noise; "high" suppresses low-noise
)
```

Provide a per-sample quality score (`1.0` = clean) under `ambient_quality_key`
in each batch. Clean samples — or a missing quality column — reproduce the
vanilla Diffusion Policy loss exactly, so the feature is opt-in and backward
compatible. See `lerobot/policies/diffusion/ambient_loss_mask.py`.

## Citation

```bibtex
@article{chi2024diffusionpolicy,
	author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
	title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	journal = {The International Journal of Robotics Research},
	year = {2024},
}
```
