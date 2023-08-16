""" Operations on trees implemented by dictionaries """

def treemap(fn, tree):
    new_tree = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            new_tree[key] = treemap(fn, value)
        else:
            new_tree[key] = fn(value)
    return new_tree
