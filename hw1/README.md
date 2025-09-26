# HW1

## File Structure

```
data/artist20/
├── train.json
├── val.json
├── train_val/
│   ├── singer_1/
│   │   ├── album_1/
│   │   │   ├── song_1.mp3
│   │   │   ├── song_2.mp3
│   │   │   └── ...
│   │   ├── ...
│   │   └── album_5/
│   ├── ...
│   └── singer_20/
│
└── test/
    ├── 001.mp3
    ├── ...
    └── 233.mp3
```

## Dataset: Artist20

- 20 artists, 120 albums, 1413 tracks
- Train / Validation / Test
- Album-level split [1]: 4 / 1 / 1
- Tracks number: 949 / 231 / 233
- dataset size: 1.28G
- Sample rate: 16000Hz
- Ext: mp3
- Channel: Mono
- Length: full song

## Task 1: Train a Traditional Machine Learning Model

- Train a traditional ML model (e.g. k-NN, SVM, random forest) with any features extracted from the audio
- Need to report the features you use and the model implementation clearly
- Need to report the validation result with confusion matrix, top1 accuracy, and top3 accuracy
- Remember to utilize standardization (e.g. mean, std), pooling and normalization to ensure consistent feature scales, reducing overfitting, and improving model stability and performance during training
- Recommendation: Librosa, Torchaudio, Sklearn

## Task 2: Train a Deep Learning Model

- Train a deep learning model from scratch
- Need to report how to implement the model clearly
- If you referred to some previous work, cite the paper or code appropriately
- Need to report the validation result with confusion matrix, top1 accuracy, and top3 accuracy
- Need to submit the testing set result to NTUCool
- Some reference methods
    - music classification: https://github.com/minzwon/sota-music-tagging-models
    - singer recognition: 
        - 19[ijcnn] [Musical artist classification with convolutional recurrent neural networks](https://github.com/ZainNasrullah/music-artist-classification-crnn)
        - 20[icassp] [Addressing the confounds of accompaniments in singer identification](https://github.com/bill317996/Singer-identification-in-artist20)
        - 21[aaai] [Positions, channels, and layers fully generalized non-local network for singer](https://github.com/ian-k-1217/Fully-Generalized-Non-Local-Network)
        - 23[ismir] [Singer identity representation learning using self-supervised techniques](https://telecom-paris.hal.science/hal-04186048/file/ISMIR_singer_id%20%2832%29.pdf)
    - speaker recognition and verification
        - 22[icassp] [Domain Adaptation for Speaker Recognition in Singing and Spoken Voice](https://github.com/rssr25/voice-recognition-speak-sing)
- Some optional baselines (not required)
    - Use pre-trained models
        - 23[apsipa] [Toward leveraging pre-trained self-supervised frontends for automatic singing voice understanding tasks- three case studies](https://arxiv.org/abs/2306.12714)
        - 23[ismir] [On the effectiveness of speech self-supervised learning for music](https://arxiv.org/abs/2307.05161)
    - Audio Language Model
        - 25[arxiv] [Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models](https://arxiv.org/pdf/2507.08128)
        - [Qwen](https://github.com/QwenLM/Qwen-Audio)


## Evaluation

1. Confusion matrix
    - A table that summarizes the performance of a classification algorithm by comparing the predicted labels with the actual labels


2. Top k accuracy (Top 1 and Top 3)
    - For a given k, if the correct label is within the top k predicted classes, it's counted as correct; otherwise, it's counted as incorrect



## Some things you need to know

- Use the file in `train.json` for training, and so as `val.json` for validation.

- DON'T USE audio in `test` folder for training.

You need to follow format as `test_pred.json` for your output. 

The format of answer is like `test_ans.json`
```
[
    label_1,
    label_2,
    ...
    label_id
]
```

The format of your output should be like `test_pred_example.json`
```
{
    '1': [top1_pred, top2_pred, top3_pred],
    '2': [top1_pred, top2_pred, top3_pred],
    ...
    'id': [top1_pred, top2_pred, top3_pred]
}
```

I'll use `count_score.py` to calculate your final results.

```
python count_score.py [answer_path] [your_output_path]

# Example
python count_score.py ./test_ans.json ./test_pred.json
```