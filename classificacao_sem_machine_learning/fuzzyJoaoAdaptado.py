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
        longer  = s2
    else:
        shorter = s2
        longer  = s1

    blocks = [[j, (j+len(shorter))] for j in range(len(longer) - len(shorter) - 1)]
    
    #if (longer[j] == shorter[0]) or (longer[len(shorter)+j-1] == shorter[-1])

    # each block represents a sequence of matching characters in a string
    # of the form (idx_1, idx_2, len)
    # the best partial match will block align with at least one of those blocks
    #   e.g. shorter = "abcd", longer = XXXbcdeEEE
    #   block = (1,3,3)
    #   best score === ratio("abcd", "Xbcd")
    
    ratios = [[(SequenceMatcher(None, shorter, longer[block[0]:block[1]]).ratio()), block[0]] for block in blocks]
    if len(ratios) > 0:
        biggest_r = max([ratio[0] for ratio in ratios])
    else:
        return [0, 0, 0]
    
    best_long_start = ratios[[ratio[0] for ratio in ratios].index(biggest_r)][1]
    best_long_end = best_long_start + len(shorter)
#     biggest_r = 0

#     for block in blocks:
#         m2 = SequenceMatcher(None, shorter, block[0])
#         r = m2.ratio()
#         if r > .95:
#             best_long_start = block[1]
#             best_long_end = len(shorter) + block[1]
#             return [best_long_start, best_long_end, 100];
#         elif r > biggest_r:
#             best_long_start = block[1]
#             best_long_end = len(shorter) + block[1]
#             biggest_r = r

    return [best_long_start, best_long_end, int(round((100 * biggest_r)))]
