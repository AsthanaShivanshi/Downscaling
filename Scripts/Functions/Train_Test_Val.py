#Splitting schenme: 70:20:10 with moving blocks for each decade starting 1971
#1971-2020: 50 full years with this scheme
#rest of the years ---> 2021-2023 ----> went into training 

#Total split : Train : 38 yeras
#Val : 10 years
#Test : 5 years-----> total 53 years from 1971-2023

import torch
import pandas as pd
from collections import defaultdict
import random

def split_by_moving_blocks(times, config=None):
    if config is None:
        config = {}

    split_cfg = config.get("split", {})
    seed = split_cfg.get("seed", 42)

    torch.manual_seed(seed)
    random.seed(seed)

    times = pd.to_datetime(times)
    year_to_indices = defaultdict(list)

    for idx, t in enumerate(times):
        year_to_indices[t.year].append(idx)

    years = sorted(year_to_indices.keys())

    # 10-year custom blocks from 1971 onward
    start_year = 1971
    decade_blocks = []

    while start_year + 9 <= 2020:
        block_years = list(range(start_year, start_year + 10))
        if all(y in year_to_indices for y in block_years):
            decade_blocks.append(block_years)
        start_year += 10

    # remaining 2021–2023 to train
    last_block_year = decade_blocks[-1][-1] if decade_blocks else 1980
    remaining_years = [y for y in years if y > last_block_year]

    train_indices, val_indices, test_indices = [], [], []
    train_years_all, val_years_all, test_years_all = [], [], []

    for block in decade_blocks:
        valid_splits = []

        for train_start in range(0, 4): 
            train_years = block[train_start:train_start + 7]

            remaining_years_in_block = [y for y in block if y not in train_years]

            for val_start in range(len(remaining_years_in_block) - 2 + 1): 
                val_years = remaining_years_in_block[val_start:val_start + 2]
                test_years = [y for y in remaining_years_in_block if y not in val_years]

                if len(test_years) == 1:
                    valid_splits.append((train_years, val_years, test_years))

        # Picking one valid combination randomly, for moving blocks within the decade being randomly placed
        train_years, val_years, test_years = random.choice(valid_splits)

        def gather(year_list):
            indices = sum([year_to_indices[y] for y in year_list], [])
            return torch.tensor(indices)[torch.randperm(len(indices))] if indices else torch.tensor([])

        train_indices.append(gather(train_years))
        val_indices.append(gather(val_years))
        test_indices.append(gather(test_years))

        train_years_all.extend(train_years)
        val_years_all.extend(val_years)
        test_years_all.extend(test_years)

    # leftover to training
    for y in remaining_years:
        if y in year_to_indices:
            train_indices.append(torch.tensor(year_to_indices[y])[torch.randperm(len(year_to_indices[y]))])
            train_years_all.append(y)
    return torch.cat(train_indices), torch.cat(val_indices), torch.cat(test_indices)
