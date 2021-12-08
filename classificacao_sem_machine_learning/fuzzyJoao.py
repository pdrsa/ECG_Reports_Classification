from difflib import SequenceMatcher

def make_type_consistent(s1, s2):
    """If both objects aren't either both string or unicode instances force them to unicode"""
    if isinstance(s1, str) and isinstance(s2, str):
        return s1, s2

    elif isinstance(s1, unicode) and isinstance(s2, unicode):
        return s1, s2

    else:
        return unicode(s1), unicode(s2)

def partial_ratio(s1, s2):
    """"Return the ratio of the most similar substring
    as a number between 0 and 100."""
    s1, s2 = make_type_consistent(s1, s2)

    if len(s1) <= len(s2):
        shorter = s1
        longer = s2
    else:
        shorter = s2
        longer = s1

    m = SequenceMatcher(None, shorter, longer)
    blocks = m.get_matching_blocks()

    # each block represents a sequence of matching characters in a string
    # of the form (idx_1, idx_2, len)
    # the best partial match will block align with at least one of those blocks
    #   e.g. shorter = "abcd", longer = XXXbcdeEEE
    #   block = (1,3,3)
    #   best score === ratio("abcd", "Xbcd")
    
    best_long_start = -1
    best_long_end = -1
    biggest_r = 0

    for block in blocks:
        long_start = block[1] - block[0] if (block[1] - block[0]) > 0 else 0
        long_end = long_start + len(shorter)
        long_substr = longer[long_start:long_end]

        m2 = SequenceMatcher(None, shorter, long_substr)
        r = m2.quick_ratio()
        if r > .995:
            best_long_start = long_start
            best_long_end = long_end
            return [best_long_start, best_long_end, 100]
        elif r > biggest_r:
            best_long_start = long_start
            best_long_end = long_end
            biggest_r = r

    return [best_long_start, best_long_end, int(round((100 * biggest_r)))]