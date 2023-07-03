""" Utils for dealing with type-l vector dictionaries """
import torch

def type_l_add(d1, d2):
    """ Add together two dictionaries of type-l vectors """
    assert sorted(d1.keys()) == sorted(d2.keys()), "d1 and d2 must have the same keys"
    ret = {}
    for key in d1.keys():
        vecs_d1 = d1[key]
        vecs_d2 = d2[key]
        assert vecs_d1.shape == vecs_d2.shape, (f"vectors for key {key} should have the same shape "
                                                f"but instead have shapes {vecs_d1.shape} and {vecs_d2.shape}")
        ret[key] = vecs_d1 + vecs_d2
    return ret


def type_l_sub(d1, d2):
    """ Subtract one dictionary of type-l vectors from another """
    assert sorted(d1.keys()) == sorted(d2.keys()), "d1 and d2 must have the same keys"
    ret = {}
    for key in d1.keys():
        vecs_d1 = d1[key]
        vecs_d2 = d2[key]
        assert vecs_d1.shape == vecs_d2.shape, (f"vectors for key {key} should have the same shape "
                                                f"but instead have shapes {vecs_d1.shape} and {vecs_d2.shape}")
        ret[key] = vecs_d1 - vecs_d2
    return ret


def type_l_cat(d1, d2):
    """ Concatenate together two dictionaries of type-l vectors """
    assert sorted(d1.keys()) == sorted(d2.keys()), "d1 and d2 must have the same keys"
    ret = {}
    for key in d1.keys():
        vecs_d1 = d1[key]
        vecs_d2 = d2[key]
        ret[key] = torch.cat([vecs_d1, vecs_d2], dim=-2)
    return ret


def type_l_partial_cat(d_large, d_small):
    """ Concatenate together two dictionaries of type-l vectors, where d_large has strictly more
     degrees than d_small """
    ret = {}
    for key in d_large.keys():
        vecs_d_large = d_large[key]
        if key in d_small.keys():
            vecs_d_small = d_small[key]
            ret[key] = torch.cat([vecs_d_large, vecs_d_small], dim=-2)
        else:
            ret[key] = vecs_d_large
    return ret


def type_l_mult(d1, d2):
    """ Multiplication involving type-l vectors """
    ret = {}
    if isinstance(d1, dict) and isinstance(d2, dict):
        assert sorted(d1.keys()) == sorted(d2.keys()), "d1 and d2 must have the same keys"
        for key in d1.keys():
            vecs_d1 = d1[key]
            vecs_d2 = d2[key]
            ret[key] = vecs_d1 * vecs_d2
    elif isinstance(d1, dict) and isinstance(d2, (int, float, torch.Tensor)):
        # print({k:v.shape for k,v in d1.items()})
        for key, vecs_d1 in d1.items():
            # if isinstance(d2, torch.Tensor):
            #     print(key, vecs_d1.shape)
            #     print(vecs_d1.device, d2.device)
            # vecs_d1 = vecs_d1.to(d2.device)
            ret[key] = vecs_d1 * d2
    elif isinstance(d1, (int, float, torch.Tensor)) and isinstance(d2, dict):
        ret = type_l_mult(d2, d1)
    else:
        raise ValueError("d1 and d2 are not an accepted type pair")

    return ret


def type_l_apply(func, d):
    """ Apply a unary function to type-l vector dicts """
    return {k: func(v) for k, v in d.items()}


def type_l_randn_like(d):
    """ Generate a random type-l vector dict with the same fiber shape as input `d` """
    return {k: torch.randn_like(v) for k, v in d.items()}


def int_to_str_key(d):
    """ Copy a dict but with str keys instead of int keys """
    return {str(k): v for k,v in d.items()}


def str_to_int_key(d):
    """ Copy a dict but with int keys instead of str keys """
    return {int(k): v for k,v in d.items()}
