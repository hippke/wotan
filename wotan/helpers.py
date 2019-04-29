from numpy import isnan, array, inf


def cleaned_array(t, y, dy=None):
    """Takes numpy arrays with masks and non-float values.
    Returns unmasked cleaned arrays."""

    def isvalid(value):
        valid = False
        if value is not None:
            if not isnan(value):
                if value < inf:
                    valid = True
        return valid

    # Start with empty Python lists and convert to numpy arrays later (reason: speed)
    clean_t = []
    clean_y = []
    if dy is not None:
        clean_dy = []

    # Cleaning numpy arrays with both NaN and None values is not trivial, as the usual
    # mask/delete filters do not accept their simultanous ocurrence without warnings.
    # Instead, we iterate over the array once; this is not Pythonic but works reliably.
    for i in range(len(y)):

        # Case: t, y, dy
        if dy is not None:
            if isvalid(y[i]) and isvalid(t[i]) and isvalid(dy[i]):
                clean_y.append(y[i])
                clean_t.append(t[i])
                clean_dy.append(dy[i])

        # Case: only t, y
        else:
            if isvalid(y[i]) and isvalid(t[i]):
                clean_y.append(y[i])
                clean_t.append(t[i])

    clean_t = array(clean_t, dtype=float)
    clean_y = array(clean_y, dtype=float)

    if dy is None:
        return clean_t, clean_y
    else:
        clean_dy = array(clean_dy, dtype=float)
        return clean_t, clean_y, clean_dy
