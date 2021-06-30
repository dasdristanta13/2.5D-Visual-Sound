# 2.5D-Visual-Sound
**Visualising Sound**

This repository is the result of the Machine Learning Project in my 1st year of Master's Degree.

## FAIR-Play Dataset
The [FAIR-Play](https://github.com/facebookresearch/FAIR-Play) repository contains the dataset we collected and used in our paper. It contains 1,871 video clips and their corresponding binaural audio clips recorded in a music room. The code provided can be used to train mon2binaural models on this dataset.


## Working
(The code has beed tested under the following system environment: ***Ubuntu 20.04 LTS, CUDA 10.2, Python 3.7, PyTorch 1.9.0***)

1. Download the FAIR-Play dataset and prepare the json split using the ***splitter.py*** with given root prefixes.

```
python3 splitter.py
```


### Preprocess

2. To generate frames from each video contained in a folder using the ***generate_frames.py***:

```
python3 generate_frames.py --folder folderpath/
```


### Training

3. Use the following command to train the mono2binaural model:
```
python3 train.py --jsonFolderPath /YOUR_CODE_PATH/2.5d_visual_sound/json/ --name mono2binaural --model audioVisual --checkpoints_dir /YOUR_CHECKPOINT_PATH/ --save_epoch_freq 50 --display_freq 10 --save_latest_freq 100 --batchSize 256 --learning_rate_decrease_itr 10 --niter 1000 --lr_visual 0.0001 --lr_audio 0.001 --nThreads 32 --gpu_ids 0,1,2,3,4,5,6,7 --validation_on --validation_freq 100 --validation_batches 50 --tensorboard True |& tee -a mono2binaural.log
```

### Testing
4. Use the following command to test your trained mono2binaural model:
```
python3 demo.py --input_audio_path /BINAURAL_AUDIO_PATH --video_frame_path /VIDEO_FRAME_PATH --weights_visual /VISUAL_MODEL_PATH --weights_audio /AUDIO_MODEL_PATH --output_dir_root /YOUT_OUTPUT_DIR/ --input_audio_length 10 --hop_size 0.05
```

### Evaluation
5. Use the following command for evaluation:
```
python3 evaluate.py --results_root /YOUR_RESULTS --normalization True
```
## Project Powerpoint

***ML_Project_report*** Contains the powerpoint I presented.

## Acknowlegements

The paper on which I worked:

```
        @inproceedings{gao2019visualsound,
          title={2.5D Visual Sound},
          author={Gao, Ruohan and Grauman, Kristen},
          booktitle={CVPR},
          year={2019}
        }
```       
Direct Link to the paper: [[arXiv]](https://arxiv.org/abs/1812.04204)

The Dataset link: [[Dataset]](https://github.com/facebookresearch/FAIR-Play)


Most portion of the code are adapted from (https://github.com/facebookresearch/2.5D-Visual-Sound)

