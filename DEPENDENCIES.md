# Dependencies

Install `requirements.txt` and a compatible Torch/Torchvision pair. The archived
numerical and visual environments used Torch 2.13.0+cu130 and Torchvision
0.28.0+cu130. Select wheels compatible with your platform. A fresh installation
and full training have not been rerun for this code-only packaging update.

The licensed LAOM subset is included under `third_party/laom/` with its license.
The external author models require separate checkouts and dependency environments:

| Model | Upstream | Pinned commit | Local directory |
|---|---|---|---|
| OTF | https://github.com/Hazel-Heejeong-Nam/lam_agent_ambiguity | c71d276abfbbfa848aa757d5e00ff1d28c8da7a4 | `third_party/otf_lam_official/` |
| FLAM | https://github.com/wangzizhao/flam | 1c707becc33e0d45411a5cb6137cd984efa7eb8b | `third_party/flam_source_complete_1c707be/` |

OTF also requires OmegaConf and its upstream dependencies. FLAM requires its
separate Linux environment. No top-level license was found in the pinned FLAM
snapshot, so its source is not redistributed. Perceptual assets, teacher weights,
DAVIS and other datasets must be obtained separately under their own terms.
