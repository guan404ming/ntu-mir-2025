# CLAUDE.md

## Overview

1. Learn to manipulate MIDI file and represent symbolic music as tokens
2. Learn to train an autoregressive model for symbolic music generation

## Task 1: Unconditional generation

1. Train a transformer-based model from scratch to generate 32 bars symbolic music
    - Model: You can use any autoregressive model
    - Either 1-stage generation or 2-stage generation is fine

2. Report
    - What’s your Token representation, model architecture
    - Implementation detail (data augmentation, hyper-parameters...)
    - Implement at least 3 combinations of your inference configurations. (it’s quite influential!)
        - e.g. model loss, sampling method, top-k, temperature, etc.
        - Table:
            | Model         | representation | event     | loss | top-k | temperature | H1 | H4 | GS | SI_short | SI_mid | SI_long |
            |----------------|----------------|------------|------|--------|--------------|----|----|----|-----------|---------|----------|
            | example model 1 | example representation 1 | example event 1   | example loss 1 | example top-k 1 | example temperature 1 | example H1 1 | example H4 1 | example GS 1 | example SI_short 1 | example SI_mid 1 | example SI_long 1 |
            | example model 2 | example representation 2 | example event 2   | example loss 2 | example top-k 2 | example temperature 2 | example H1 2 | example H4 2 | example GS 2 | example SI_short 2 | example SI_mid 2 | example SI_long 2 |
            | example model 3 | example representation 3 | example event 3   | example loss 3 | example top-k 3 | example temperature 3 | example H1 3 | example H4 3 | example GS 3 | example SI_short 3 | example SI_mid 3 | example SI_long 3 |

    - For each combination, generate 20 mid/midi files (32 bars) to calaulate their average objective metrics results (H4, GS)

3. Submission files
    - Choose one combination, generate 20 mid/midi files (32 bars) and convert into wav files as listening samples

## Task 2: Conditional generation

1. TA will provide 3 midi files (8 bars) as prompt, you need to generate their continuation for 24 bars. (Total: 8+24 bars)
    - You can use the checkpoint in Task 1; no need to train another model
    - You can consider this task as a 「指定曲」 competition. Since everyone is given the same prompts, you can compare your generation results with others.
2. Report
    - No need to report objective metrics results; it's not needed in Task2.
    - For each prompt song, need to generate 1 mid/midi files (8+24 bars) for 3 different inference configurations and convert into wav files as listening samples.
    - Need to specify the model, representation, inference configurations you used for each continuation.
3. Submission files
    - For each prompt song, need to specify the one you think that generates the best among different inference configurations

## Implementation

### Rules

1. use uv to manage the environment abd use ruff to format the code
2. follow the best practices and keep the code clean and simple but precise
3. put source code in `./src`
4. there are some sample code in `./tutorial/`, you can use them as reference
5. NO need to create any readme or any doc for the changes you made
6. keep every epochs checkpoint in `./checkpoints`

### Steps

- **Step (1):** Task & Data  
- **Step (2):** Token representation  
- **Step (3):** Choose a model  
- **Step (4):** Train the model till the loss is sufficiently low  
- **Step (5):** Do inference  
- **Step (6):** Listen to the generated music!  
- **Step (7):** Do evaluation  

### Dataset: Pop1k7

- location: `./data/Pop1k7`
- 1747 pop music piano performances (mid or midi file) transcribed from youtube audio
- Single track, 4 minutes in duration, totaling 108 hours.
- 4/4 time signature (four beats per bar)
- You can use whole dataset for training, there is no need to split train/validation set for generation task.

### Evaluation

- (H4 Required) Pitch-Class Histogram Entropy (H4) : measures erraticity of pitch usage in shorter timescales (e.g., 1 or 4 bars).
- (GS Required) Grooving Pattern Similarity (GS) : measures consistency of rhythm across the entire piece.
- (SI Optional) Structureness Indicator (SI) : detects presence of repeated structures within a specified range of timescale.
- Use MusDr. (TA will provide sample code: `./tutorial/eval_metrics.py`)
- The closer the score of generation results compare to the original dataset, the better

