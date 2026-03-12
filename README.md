## MOMENTS dataset ⚽️
A corpus of video fragments extracted from football games in [SoccerReplay-1988](https://huggingface.co/datasets/Homie0609/SoccerReplay-1988). For each video fragment (`*.mp4`) the dataset includes an 'importance' annotation (```important``` or ```non-important```), associated audio commentary (`*_v2.wav`), and its corresponding textual transcription (`*_v2.json`).

For obtaining access to the MOMENTS dataset, follow these steps:
1. Request access to the [SoccerReplay-1988 dataset](https://huggingface.co/datasets/Homie0609/SoccerReplay-1988) by signing this [NDA form](https://bifbrprted3.feishu.cn/share/base/form/shrcnkMPY0WTotp0K3mtjJtMgSf).
2. Upon receiving access to SoccerReplay-1988, please forward the information to [this email](mailto:a.k.surikuchi@uva.nl).

**Structure:** We use game-ids provided in the SoccerReplay-1988 dataset to uniquely identify football games (e.g., `0jJj5Mme`). Each game contains important and non-important moments, and for both these classes, moments belonging to both halves of the game are placed under corresponding directories&mdash;`1/` & `2/`. Our code for experiments and analyses relies on the [data.json](data.json) file, that comprises IDs for all the 3954 moments in the dataset (e.g., `0jJj5Mme-1-IM_1`).
```
0jJj5Mme
├── important-moments
│   ├── 1
│   │   ├── IM_1.mp4
│   │   ├── IM_1_v2.json
│   │   ├── IM_1_v2.wav
        .
        .
        .
│   │   ├── IM_17.mp4
│   │   ├── IM_17_v2.json
│   │   └── IM_17_v2.wav
│   └── 2
│       ├── IM_1.mp4
│       ├── IM_1_v2.json
│       ├── IM_1_v2.wav
        .
        .
        .
│       ├── IM_23.mp4
│       ├── IM_23_v2.json
│       └── IM_23_v2.wav
└── non-important-moments
    ├── 1
    │   ├── NIM_1.mp4
    │   ├── NIM_1_v2.json
    │   ├── NIM_1_v2.wav
        .
        .
        .
    │   ├── NIM_21.mp4
    │   ├── NIM_21_v2.json
    │   └── NIM_21_v2.wav
    └── 2
        ├── NIM_1.mp4
        ├── NIM_1_v2.json
        ├── NIM_1_v2.wav
        .
        .
        .
        ├── NIM_19.mp4
        ├── NIM_19_v2.json
        └── NIM_19_v2.wav

6 directories, 400 files
```
**Note:** The `*_v2.json` files include both `local` and `global` transcriptions. These refer to text obtained from individual audio segments (moment-level) and the full match audio, respectively. We primarily used `global` transcriptions for our work (see [experiments/classify.py](experiments/classify.py) for more details).

## Experiments ⚖️

Our code for conducting classification and evaluation is provided under [experiments](experiments/).  
**Prerequisite:** Libraries in the [requirements.txt](requirements.txt) file need to be installed.

```python
python -u experiments/classify.py --help
python -u experiments/evaluate.py --help
```

The train:test splits we used for baseline models are provided in [Baseline/data_splits.json](results/Baseline/data_splits.json). Furthermore, the module in [experiments/evaluate.py#L17-L29](experiments/evaluate.py#L17-L29) can be used for analyzing learned weights of the `baseline (text)` model.

## Analyses 🧐

Our code for examining behavior of models (in terms of their confidence) is provided under [analyses](analyses/).

```python
python -u analyses/influence_of_modalities.py --help
python -u analyses/role_of_multimodality.py --help
```

🔗 More details about the construction and usage of MOMENTS are available through our preprint:
```
@misc{surikuchi2026multimodalgoalpostability,
      title={Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Contextually Important Moments}, 
      author={Aditya K Surikuchi and Raquel Fernández and Sandro Pezzelle},
      year={2026},
      eprint={2601.16333},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.16333}, 
}
```