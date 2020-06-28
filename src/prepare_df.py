from music21 import *
import pandas as pd
import numpy as np
from itertools import groupby
from operator import add, itemgetter
from functools import reduce
import matplotlib.pyplot as plt

def generate_row(mus_object, part, pitch_class=np.nan):
    """
    Prepare rows for musical score `pandas.DataFrame`.

    Parameters
    ----------
    mus_object: music21.Music21Object
    part: music21.part
    pitch_class: float
        Default of `numpy.nan` for unpitched objects and a pitch class for rows
        with `Type` value `music21.pitch.Pitch`

    Returns
    -------
    dict
    """
    d = {}
    d.update({'id': mus_object.id,
              'Part Name': part.partName,
              'Offset': mus_object.offset,
              'Duration': mus_object.duration.quarterLength,
              'Type': type(mus_object),
              'Pitch Class': pitch_class})
    return d


def generate_df(score):
    """
    Prepare `pandas.DataFrame` from musical score.

    Parameters
    ----------
    score: music21.stream.Score

    Returns
    -------
    pandas.DataFrame

    """
    parts = score.parts
    rows_list = []
    for part in parts:
        for index, elt in enumerate(part.flat.stripTies().getElementsByClass(
                [note.Note, note.Rest, chord.Chord, bar.Barline])):
            if hasattr(elt, 'pitches'):
                pitches = elt.pitches
                for pitch in pitches:
                    rows_list.append(generate_row(elt, part, pitch.pitchClass))
            else:
                rows_list.append(generate_row(elt, part))
    return pd.DataFrame(rows_list)


def combine_rests(df):
    """
    Condense consecutive rests of a musical score (as stored in a
    `pandas.DataFrame`) into single long rests.

    Parameters
    ----------
    df: pandas.DataFrame
        `df` contains columns as returned by `generate_df`.

    Returns
    -------
    pandas.DataFrame
        `df` with consecutive rows with `Type` value `music21.note.Rest`
        combined, `Duration` values summed.

    """
    rests = df[df['Type'] == note.Rest]

    # Group the indices of consecutive runs of rests that occur in the same
    # part
    parts_names = rests['Part Name'].unique()
    rest_runs_in_parts = []
    for part_name in parts_names:
        rest_idx = rests[rests['Part Name'] == part_name].index
        rest_runs_in_part = [list(map(itemgetter(1), g)) for _, g in
                             groupby(enumerate(rest_idx),
                                     lambda ix: ix[1] - ix[0])]
        rest_runs_in_parts.append(rest_runs_in_part)
    rest_runs = reduce(add, rest_runs_in_parts)

    # Map each index group to a single index (maintaining order)
    initial_rest_lookup = {}
    for nums in rest_runs:
        initial_rest_lookup.update(dict.fromkeys(nums, nums[0]))

    def get_initial_rest(k):
        return initial_rest_lookup.get(k, k)

    # Condense each rest group to the first with durations summed
    agg_func = dict.fromkeys(df, 'first')
    agg_func.update({'Duration': 'sum'})
    return df.groupby(get_initial_rest, axis=0).agg(agg_func)


if __name__ == '__main__':
    clara_search = corpus.search('clara')
    clara_score = clara_search[0].parse()
    clara_df = generate_df(clara_score)
    clara_condensed_df = combine_rests(clara_df)
    # clara_df.to_csv('test_dataframe_before_condensing.csv')
    # clara_condensed_df.to_csv('test_dataframe_after_condensing.csv')
