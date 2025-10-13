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
   * You should slice the target music to the length same as the music your model generates. (i.e. If you use MuseControlLite, trim the target music to the first 47 seconds)

## Rules

1. Do the music captioning with ALMs.
2. You can not directly use the music from target_music_list as condition to generate music, only condition extracted using MIR tools are allowed.
   a. You can not use an auto-encoder to encode and decode the target audio, and use it as submission
   b. You can not use the “audio condition” in MuseControlLite.
3. You can use different methods for each music in target_music_list.
