import argparse
import os
import pandas as pd
import tqdm

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir")
    parser.add_argument("data_csv")
    parser.add_argument("--cluster", default=False, action='store_true')
    parser.add_argument("--num_subset", default=1000*100, type=int)
    parser.add_argument("--max_len", default=1000, type=int)
    args = parser.parse_args()

    cache_df = pd.read_csv(os.path.join(args.cache_dir, "metadata.csv"))
    data_df = pd.read_csv(args.data_csv)
    df = data_df.merge(cache_df, on="pdb_name")
    df = df[df['modeled_seq_len'] < args.max_len]
    if args.cluster:
        df = df.groupby("cluster").sample(1)

    if args.num_subset > len(df):
        subsample = df
    else:
        subsample = df.sample(args.num_subset)
    latent_means = []
    latent_samples = []
    for path in tqdm.tqdm(subsample['latent_cache_file']):
        data = torch.load(path)
        latent_sigma = torch.exp(data['latent_logvar'] * 0.5)
        eps = torch.randn_like(latent_sigma)
        latent_means.append(data['latent_mu'])
        latent_samples.append(
            data['latent_mu'] + eps * latent_sigma
        )

    latent_size = latent_means[0].shape[-1]
    print("compute mean")
    latent_means = torch.cat(latent_means, dim=0)
    latent_center = torch.mean(latent_means, dim=0)
    print(latent_center.shape, latent_means.shape)
    print("compute std")
    latent_samples = torch.cat([l.view(-1, latent_size) for l in latent_samples], dim=0)
    latent_samples = latent_samples - latent_center[None]
    latent_std = latent_samples.std(dim=0)

    stats_dict = {
        "paths": subsample['latent_cache_file'].tolist(),
        "mu": latent_center.cpu(),
        "std": latent_std.cpu(),
    }
    print(stats_dict)
    torch.save(stats_dict, os.path.join(args.cache_dir, "latent_stats.pt"))


