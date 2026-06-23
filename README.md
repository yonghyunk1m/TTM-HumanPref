<div align="center">

# Improving Text-to-Music Generation<br>with Human Preference Rewards

### ICME 2026 Academic Text-to-Music (ATTM) Grand Challenge · Efficiency Track

<span style="white-space:nowrap;">[Yonghyun Kim](https://yonghyunk1m.notion.site/)<sup>♭</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Junwon Lee](https://jnwnlee.github.io/)<sup>♮</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Haiwen Xia](https://haiwen-xia.github.io/)<sup>♯</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Yinghao Ma](https://nicolaus625.github.io/)<sup>♭♭</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Chris Donahue](https://chrisdonahue.com/)<sup>♮♮</sup></span>

<sub><sup>♭</sup>Georgia Tech&nbsp;·&nbsp;<sup>♮</sup>KAIST&nbsp;·&nbsp;<sup>♯</sup>Peking University&nbsp;·&nbsp;<sup>♭♭</sup>Queen Mary University of London&nbsp;·&nbsp;<sup>♮♮</sup>Carnegie Mellon University</sub>

<br/>

[![Paper](https://img.shields.io/badge/arXiv-2606.21670_·_Paper-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.21670)
[![Listening demo](https://img.shields.io/badge/🎧_Demo-Listening_samples-0071e3.svg)](https://yonghyunk1m.github.io/TTM-HumanPref/)
[![Hugging Face](https://img.shields.io/badge/🤗_Space-Landing_page-FFD21E.svg)](https://huggingface.co/spaces/yonghyunk1m/TTM-HumanPref)
[![Reward model](https://img.shields.io/badge/arXiv-2606.17006_·_TuneJury-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.17006)
[![Grand Challenge](https://img.shields.io/badge/🏆_ATTM-Grand_Challenge-f39c12.svg)](https://ntu-musicailab.github.io/ICME26-ATTM-Grand-Challenge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

</div>

---

<div align="center">

*Built on the [TuneJury](https://arxiv.org/abs/2606.17006) preference reward ([code](https://github.com/yonghyunk1m/TuneJury)), itself a follow-up to [Music Arena](https://huggingface.co/music-arena) that distills its live human preference votes into a reusable signal.*

</div>

## TL;DR

A 120M parameter FluxAudio-S backbone (the official challenge baseline) is conditioned on a learned human-preference reward (**TuneJury**) and refined through expert iteration and a short CRPO pass. The pipeline fits in ~40 GPU-hours on one NVIDIA RTX A5000 and produces 10 s clips in under a second at inference.

## Five engineering decisions

Four at training time, one at inference:

1. **Training-time reward conditioning** that doubles as an inference-time CFG axis. *Effect:* score-conditioned variants improve FAD-CLAP by 0.025–0.040 absolute over the FluxAudio-S baseline (text-conditioned, no score conditioning) at the SFT stage. The inference-time score knob, however, ends up saturated after the full chain (Sub. 1's reward is essentially flat across `s ∈ [0, 6]`).
2. **Five-head sweep** over score-conditioning architectures; deployed via a v1→v2 hybrid (train in `GlobalAdaLN (v1)`, cross-load into `InputAdd (v2)` at Stage 3). *Effect:* direction matters: v1→v2 cross stays within 0.02 FAD-CLAP of native, while v2→v1 cross collapses (FAD-CLAP ~0.69, reward ~−0.50).
3. **Expert iteration** on the top decile by combined reward + CLAP-text score. *Effect:* **dominant chain contributor**: FAD-CLAP −0.0362 on the v1 chain, paired-t significant on both CLAP and Reward.
4. **Short CRPO preference-tuning** with a DPO-style objective for audio-caption alignment. *Effect:* within paired-t noise at this scale (FAD-CLAP −0.003, CLAP +0.002, Reward −0.002 over Chain-end). We keep the 5 k-step pass because it is inexpensive, not because it moves the headline numbers.
5. **Inference post-processing**: joint CFG on text and reward, 3×Demucs `mdx_extra` source separation, LUFS normalization to −16.5. *Effect:* consistently improves internal validation metrics; the score-scalar knob itself is already saturated by this point in the chain, so we hold `s = 5.0` because validation picked it, not because it remains a useful lever.

## Submissions

Two seed-varied submissions:

| Submission | Seed | FAD-CLAP ↓ | CLAP ↑ | Reward ↑ |
|---|---|---|---|---|
| Sub. 1 (`seed=42`) | 42 | **0.4238** | 0.285 | +0.533 |
| Sub. 2 (`seed=55`) | 55 | 0.4370 | **0.300** | **+0.550** |

All numbers above are on a 100-prompt held-out slice of the Song Describer Dataset, evaluated against SDD-706 with LAION-CLAP-Music. Under the challenge's hidden Jamendo reference set, our submission scored FAD 0.498, CLAP 0.270, CCS 0.763 (Submission e02 in the [ATTM Grand Challenge summary paper](https://ntu-musicailab.github.io/ICME26-ATTM-Grand-Challenge/)).

## Repository layout

This repository contains **only our additions** to the official ATTM-GC-FluxAudio baseline: the score-conditioning training recipes, expert-iteration / CRPO fine-tuning scripts, inference post-processing, and our unified CLI. The MeanAudio / FluxAudio codebase itself is not redistributed here. Please clone the upstream baseline separately (see "Setup" below).

```
TTM-HumanPref/
├── run.py              # Unified CLI (params / infer / postproc / eval / reproduce / train)
├── config/             # Our training configs (score-conditioning + CRPO recipes)
├── scripts/            # Our training, inference, and post-processing scripts
│   ├── flowmatching/   # Stage 1–3 training shell scripts
│   ├── postproc/       # 3×Demucs and LUFS normalization
│   └── ...
└── docs/               # Demo page hosted via GitHub Pages
```

## Setup

This repo provides the recipes; the **upstream baseline** provides the model code and environment.

```bash
# 1. Clone the official ATTM-GC-FluxAudio baseline and follow its environment setup.
git clone https://github.com/ntu-musicailab/ICME26-ATTM-GC-FluxAudio.git
cd ICME26-ATTM-GC-FluxAudio
# (follow that repo's setup.sh / requirements; install the `meanaudio` conda env)

# 2. Overlay this repo's run.py, config/, scripts/ on top of the baseline clone.
git clone https://github.com/yonghyunk1m/TTM-HumanPref.git /tmp/icme-overlay
cp -r /tmp/icme-overlay/run.py /tmp/icme-overlay/config /tmp/icme-overlay/scripts .

# 3. Print the trainable / total / frozen parameter counts.
python run.py params

# 4. Download submission checkpoints (see "Pre-trained checkpoints" below).
```

## Pre-trained checkpoints

The submitted FluxAudio-S checkpoints (Sub. 1, Sub. 2) and the TuneJury reward model are published as **GitHub Release artifacts**: see the [Releases page](https://github.com/yonghyunk1m/TTM-HumanPref/releases).

| File | Size | Role |
|---|---|---|
| `sub1_seed42_fluxaudio_s.pt` | ~ 460 MB | Submitted Hybrid checkpoint (seed 42) |
| `sub2_seed55_fluxaudio_s.pt` | ~ 460 MB | Submitted Hybrid checkpoint (seed 55) |
| `tunejury_reward.pt` | ~ 10 MB | TuneJury MLP head over LAION-CLAP + MERT (challenge-submission version) |

The LAION-CLAP-Music encoder and BigVGAN 44.1 kHz vocoder used for inference are third-party assets. Please download them via the upstream baseline's `download_weights` helper.

## Training pipeline (3 stages)

After completing **Setup**, run each stage in order. Each stage initialises from the previous stage's weights.

```bash
# Stage 1: Score-conditioned SFT on the full instrumental Jamendo set (v1 GlobalAdaLN forward, ~32 h on 1× A5000)
bash scripts/flowmatching/train_fluxaudio_s_44k_score_v1.sh
# (train_fluxaudio_s_44k_score_v2.sh trains the v2 InputAdd SFT used for the SFT-only v2 cells in the cross-mechanism ablation, not used by the deployed chain.)

# Stage 2: Expert iteration: rank ~600 self-generated samples by (TuneJury + CLAP-text),
# keep top 64, 5× oversample into the 535 k-clip mixture, fine-tune for 30 k + 5 k steps
bash scripts/flowmatching/finetune_v1_44k_expert_iter.sh

# Stage 3: CRPO with DPO-style objective on 2 k preference pairs, β = 2000, 5 k updates
bash scripts/flowmatching/train_crpo_A_from_FT_expert.sh
```

End-to-end pipeline (SFT + expert-iter + CRPO + ranker training) fits in ~40 GPU-hours on one NVIDIA RTX A5000.

## Inference

Single-value joint CFG on text and reward, with `s = 5.0`, `w = 4.0`, 25 Euler steps, prompt prefix `"high quality instrumental music, "`. Post-processing pipes the output through 3× Demucs `mdx_extra` and LUFS-normalizes to −16.5.

```bash
# Inference + post-processing (Sub. 1 = seed 42)
python run.py reproduce --seed 42 --prompts data/sdd100_prompts.csv --output_dir output/sub1
```

See `python run.py --help` for sub-commands (`params`, `infer`, `postproc`, `eval`, `reproduce`, `train`).

## TuneJury preference reward

TuneJury is a small (~2.8 M trainable) MLP head over frozen LAION-CLAP + MERT-v1-330M encoders, trained on ~22 K human preference pairs pooled from Music Arena, MusicPrefs, AIME, and SongEval. Full architecture, training data, and ablations are described in the companion preprint ([TuneJury, arXiv:2606.17006](https://arxiv.org/abs/2606.17006), [code](https://github.com/yonghyunk1m/TuneJury)). The reward checkpoint used at challenge-submission time is the one published with this repository's release. The full released model (the bench-clean checkpoint behind all paper numbers) lives in the [TuneJury repository](https://github.com/yonghyunk1m/TuneJury).

## Citation

If you find this work useful, please cite the arXiv preprint (to appear in the ICME 2026 ATTM Grand Challenge proceedings):

```bibtex
@misc{kim2026ttmhumanpref,
  title         = {Improving Text-to-Music Generation with Human Preference Rewards},
  author        = {Kim, Yonghyun and Lee, Junwon and Xia, Haiwen and Ma, Yinghao and Donahue, Chris},
  year          = {2026},
  eprint        = {2606.21670},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2606.21670}
}
```

## Acknowledgements

This work builds on the official ATTM Grand Challenge baseline ([Hsieh et al., 2026](https://github.com/ntu-musicailab/ICME26-ATTM-GC-FluxAudio)), the FluxMusic architecture ([Fei et al., 2024](https://arxiv.org/abs/2409.00587)), and a number of open music-preference datasets (Music Arena, MusicPrefs, AIME, SongEval). The engineering pipeline was assembled in a human–agent loop with Claude Code (Anthropic Claude Opus 4.6 / 4.7).

## License

MIT. See [LICENSE](LICENSE).
