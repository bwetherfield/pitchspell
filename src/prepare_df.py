from music21 import *
import pandas as pd
import numpy as np
from itertools import groupby
from operator import add, itemgetter
from functools import reduce

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
        for index, elt in enumerate(part.flat):
            if hasattr(elt, 'pitches'):
                pitches = elt.pitches
                for pitch in pitches:
                    rows_list.append(generate_row(elt, part, pitch.pitchClass))
            else:
                rows_list.append(generate_row(elt, part))
    return pd.DataFrame(rows_list)

def generate_row(musObject, part, pitchClass=np.nan):
    """
    Prepare rows for musical score `pandas.DataFrame`.

    Parameters
    ----------
    musObject: music21.Music21Object
    part: music21.part
    pitchClass: float
        Default of `numpy.nan` for unpitched objects and a pitch class for rows with `Type` value `music21.pitch.Pitch`

    Returns
    -------
    dict
    """
    d = {}
    d.update({'id': musObject.id,
              'Part Name': part.partName,
              'Offset': musObject.offset,
              'Duration': musObject.duration.quarterLength,
              'Type': type(musObject),
              'Pitch Class': pitchClass})
    return d

def combine_rests(df):
    """
    Condense consecutive rests of a musical score (as stored in a `pandas.DataFrame`) into single long rests.

    Parameters
    ----------
    df: pandas.DataFrame
        `df` contains columns as returned by `generate_df`.

    Returns
    -------
    pandas.DataFrame
        `df` with consecutive rows with `Type` value `music21.note.Rest` combined, `Duration` values summed.

    """
    rests = df[df['Type'] == note.Rest]

    # Group the indices of consecutive runs of rests that occur in the same part.
    parts_names = rests['Part Name'].unique()
    rest_runs_in_parts = []
    for part_name in parts_names:
        rest_idx = rests[rests['Part Name'] == part_name].index
        rest_runs_in_part = [list(map(itemgetter(1), g)) for _, g in
                             groupby(enumerate(rest_idx), lambda ix: ix[1] - ix[0])]
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
