# LiTTA
[INTERSPEECH'24] Official code for "LI-TTA: Language Informed Test-Time Adaptation for Automatic Speech Recognition"


## Environmental Setup
```
conda create -y -n LiTTA python=3.10
conda activate LiTTA
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
conda env update --file environment.yaml --prune
```


## Pre-trained Models
- [CTC-based Model](https://huggingface.co/facebook/wav2vec2-base-960h)
  - CTC-based model will be automatically downloaded if you set ``asr`` as ``facebook/wav2vec2-base-960h``.
- [4-gram Language Model for CTC-based Model](https://huggingface.co/patrickvonplaten/wav2vec2-base-100h-with-lm)
  - You need to download language by your own using following command:
  ```
  git lfs install
  git clone https://huggingface.co/patrickvonplaten/wav2vec2-base-100h-with-lm pretrained_models/wav2vec2-base-100h-with-lm
  ```


## Run
You can run main.py (baseline) or main_lm.py (litta) using the command below:
```
python main_lm.py \
    --config-name [CONFIG.YAML] \
    dataset_name=[DATASET_NAME] \
    dataset_dir=[DATASET_DIR] \
```
Currently available parameters are as follows:

  Parameter | Value
  --- | ---
  CONFIG.YAML | config.yaml, config_{sgem\|litta}_ctc.yaml
  DATASET_NAME | librispeech, chime, ted, commonvoice, valentini, l2arctic


## Acknowledgement

This work was partially supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and SAMSUNG Research, Samsung Electronics Co.,Ltd.

Also, we thank the authors of the [SGEM](https://github.com/drumpt/SGEM) for their open-source contributions and their assistance with the data preparation.

## Citation
```
@inproceedings{yoon2024li,
  title={LI-TTA: Language Informed Test-Time Adaptation for Automatic Speech Recognition},
  author={Yoon, Eunseop and Yoon, Hee Suk and Harvill, John and Hasegawa-Johnson, Mark and Yoo, Chang D},
  booktitle={Proc. Interspeech 2024},
  pages={3490--3494},
  year={2024}
}
```
