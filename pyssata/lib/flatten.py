def flatten(x):
    return list(_flatten(x))

def _flatten(x):
    '''
    Generator that will flatten a list that may contain
    other lists (nested arbitrarily) and simple items
    into a flat list.

    >>> flat = flatten([[1,[2,3]],4,[5,6]])
    >>> list(flat)
    [1,2,3,4,5,6]

    '''
    for item in x:
        try:
            yield from flatten(item)
        except TypeError:
            yield item
