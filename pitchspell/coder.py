def encode(pitch):
    if pitch.getLowerEnharmonic().accidental.alter > 2.0:
        return (1, 1)
    elif pitch.getHigherEnharmonic().accidental.alter < -2.0:
        return (0, 0)
    else:
        return (0, 1)

def decode(pair, pitch):
    test = sum(pair) - sum(encode(pitch))
    if test == 0:
        return pitch
    elif test == 1:
        return pitch.getLowerEnharmonic()
    elif test == -1:
        return pitch.getHigherEnharmonic()
    elif test == 2:
        return pitch.getLowerEnharmonic().getLowerEnharmonic()
    elif test == -2:
        return pitch.getHigherEnharmonic().getHigherEnharmonic()
