import matplotlib.pyplot as plt


def format_list(l, fmt_str):
    return [float(fmt_str.format(i)) for i in l]


def gen_pbar_str(loss_dict, include_keys=None):
    if include_keys is None:
        include_keys = loss_dict.keys()
    pbar_str = []
    for key, value in loss_dict.items():
        if key in include_keys:
            if len(value.shape) == 0:
                value = value[None]
            pbar_str.append(f"{key}: {format_list(value.tolist(), '{:.4f}')}")
    pbar_str = ", ".join(pbar_str)
    return pbar_str


def update_epoch_loss_dict(epoch_dict, loss_dict):
    for key, value in loss_dict.items():
        if len(value.shape) == 0:
            value = value.unsqueeze(0)
        if key not in epoch_dict.keys():
            epoch_dict[key] = value.tolist()
        else:
            epoch_dict[key] += value.tolist()

    return epoch_dict


def plot_training_curves(train_curves, curve, keys=None, ylim=None, yscale=None):
    if keys is None:
        keys = train_curves[0][curve].keys()

    plot_dict = {}
    for epoch in train_curves[1:]:
        curve_dict = epoch[curve]
        for key in keys:
            if key in plot_dict:
                plot_dict[key].append(curve_dict[key])
            else:
                plot_dict[key] = [curve_dict[key]]

    for key, key_curve in plot_dict.items():
        plt.plot(key_curve, label=key)
    if ylim is not None:
        plt.ylim(ylim)
    if yscale is not None:
        plt.yscale(yscale)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.savefig("train_curves.png")
    # plt.show()
