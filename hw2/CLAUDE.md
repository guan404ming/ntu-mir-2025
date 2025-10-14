# CLAUDE.md

## Task 1 - Retrieval

We provide `data/referecne_music_list_60s`, retrieve the most similar ones for each music in `data/target_music_list_60s/`.

Suggestions for Audio Encoder Options include but not limited to:
1. Stable-Audio-Open VAE encode
2. Music2latent
3. CLAP
4. MuQ

For each song in `data/target_music_list_60s/`, you will have to report:
1. CLAP,
   a. calculate cosine similarity between:
      i. Generated music and target music
2. Meta Audiobox Aesthetics
   a. CE: Content Enjoyment
   b. CU: Content Usefulness
   c. PC: Production Complexity
   d. PQ: Production Quality
3. Melody similarity (accuracy)

## Task 2 - Generation

Method:
1. Audio Captioning
   a. To obtain the text description of music in target_music_list
2. Text-to-Music
   a. Simple: Text condition only
   b. Medium: Any condition extracted from music (e.g. Melody, Rhythm)
   c. Strong: Adjust the classifier free guidance.

Suggestions for ALMs and Text-to-Music models:
1. ALMs
   a. [Audio Flamingo 3](https://github.com/NVIDIA/audio-flamingo)
   b. [Qwen-audio](https://github.com/QwenLM/Qwen2-Audio)
   c. [LP-MusicCaps](https://github.com/seungheondoh/lp-music-caps)
2. Controllable Text-to-Music models
   a. [MuseControlLite](https://github.com/fundwotsai2001/MuseControlLite) (text, rhythm, dynamics, melody)
   b. [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) (text, melody)
   c. [MusicGen-style](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN_STYLE.md) (text, style)
   d. [Jasco](https://github.com/facebookresearch/audiocraft/blob/main/docs/JASCO.md) (text, chords, melody, separated drum tracks, full-mix audio)
   e. [Coco-mulla](https://github.com/Kikyo-16/coco-mulla-repo) (pitch, chords, drum track)

The music in `data/target_music_list_60s/` are 60-second, you can generate music up to the models limit (e.g. MuseControlLite generates 47-second music)

For each song in `data/target_music_list_60s/`, you will have to report:
1. CLAP
   a. calculate cosine similarity between:
      i. Generated music and target music
2. Meta Audiobox Aesthetics
   a. CE: Content Enjoyment
   b. CU: Content Usefulness
   c. PC: Production Complexity
   d. PQ: Production Quality
3. Melody similarity (accuracy)

## Rules

1. Do the music captioning with ALMs.
2. You can not directly use the music from target_music_list as condition to generate music, only condition extracted using MIR tools are allowed.
   a. You can not use an auto-encoder to encode and decode the target audio, and use it as submission
   b. You can not use the “audio condition” in MuseControlLite.
3. You can use different methods for each music in target_music_list.
