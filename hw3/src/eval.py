import itertools
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI
from tqdm import tqdm

from src.musdr import (
    get_bars_crop, 
    get_pitch_histogram, 
    compute_histogram_entropy, 
    get_onset_xor_distance,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--eval_folder",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    return parser.parse_args()


def compute_piece_pitch_entropy(
    piece_ev_seq,
    window_size,
    bar_ev_id,
    pitch_evs,
    verbose=False,
):
    '''
    Computes the average pitch-class histogram entropy of a piece.
    (Metric ``H``)

    Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    window_size (int): length of segment (in bars) involved in the calc. of entropy at once.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    verbose (bool): whether to print msg. when a crop contains no notes.

    Returns:
    float: the average n-bar pitch-class histogram entropy of the input piece.
    '''
    # remove redundant ``Bar`` marker
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]

    n_bars = piece_ev_seq.count(bar_ev_id)
    if window_size > n_bars:
        print ('[Warning] window_size: {} too large for the piece, falling back to #(bars) of the piece.'.format(window_size))
        window_size = n_bars

    # compute entropy of all possible segments
    pitch_ents = []
    for st_bar in range(n_bars - window_size + 1):
        seg_ev_seq = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1, bar_ev_id)
        pitch_hist = get_pitch_histogram(seg_ev_seq, pitch_evs=pitch_evs)
        if pitch_hist is None:
            if verbose:
                print ('[Info] No notes in this crop: {}~{} bars.'.format(st_bar, st_bar + window_size - 1))
            continue

        pitch_ents.append(compute_histogram_entropy(pitch_hist))
    return np.mean(pitch_ents)


def compute_piece_groove_similarity(
    piece_ev_seq,
    bar_ev_id,
    pos_evs,
    pitch_evs,
    max_pairs=1000,
):
    '''
    Computes the average grooving pattern similarity between all pairs of bars of a piece.
    (Metric ``GS``)

    Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    max_pairs (int): maximum #(pairs) considered, to save computation overhead.

    Returns:
    float: 0~1, the average grooving pattern similarity of the input piece.
    '''
    # remove redundant ``Bar`` marker
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]

    # get every single bar & compute indices of bar pairs
    n_bars = piece_ev_seq.count(bar_ev_id)
    bar_seqs = []
    for b in range(n_bars):
        bar_seqs.append(get_bars_crop(piece_ev_seq, b, b, bar_ev_id))
    pairs = list(itertools.combinations(range(n_bars), 2))

    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    # compute pairwise grooving similarities
    grv_sims = []
    for p in pairs:
        grv_sims.append(
            1. - get_onset_xor_distance(
                bar_seqs[p[0]],
                bar_seqs[p[1]],
                bar_ev_id,
                pos_evs,
                pitch_evs=pitch_evs,
            )
        )

    return np.mean(grv_sims)


if __name__ == "__main__":
    args = parse_arguments()
    config = TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_programs=True,
        use_tempos=True,
    )
    tokenizer = REMI(config)

    BAR_EV = [v for k, v in tokenizer.vocab.items() if "Bar" in k][0]
    POS_EVS = [v for k, v in tokenizer.vocab.items() if "Position" in k]
    PITCH_EVS = [v for k,v  in tokenizer.vocab.items() if "Pitch" in k]

    dataset = DatasetMIDI(
        files_paths=list(Path(args.eval_folder).glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    
    scores = []
    for data in tqdm(dataset):
        seq = data['input_ids'].tolist()
        h1 = compute_piece_pitch_entropy(
            seq,
            1,
            bar_ev_id=BAR_EV,
            pitch_evs=PITCH_EVS
        )

        h4 = compute_piece_pitch_entropy(
            seq,
            4,
            bar_ev_id=BAR_EV,
            pitch_evs=PITCH_EVS
        )

        gs = compute_piece_groove_similarity(
            seq,
            bar_ev_id=BAR_EV,
            pos_evs=POS_EVS,
            pitch_evs=PITCH_EVS,
        )
        scores.append(
            {
                "h1": h1,
                "h4": h4,
                "gs": gs,
            }
        )
    output_path = (
        args.output_path
        if args.output_path is not None
        else Path(args.eval_folder, "scores.csv")
    )
    score_df = pd.DataFrame(scores)
    print(score_df.mean().round(4))
    score_df.to_csv(output_path, index=False)
