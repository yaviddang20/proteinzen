import pandas as pd

sc_df = pd.read_csv("esmfold/sc_rmsd.csv")
folding_df = pd.read_csv("esmfold/folding_rmsd.csv")

all_tasks = sorted(folding_df['task'].unique().tolist())

pmpnn_pass = sc_df[
    (sc_df['motif_all_atom_rmsd'] < 1.5)
    & (sc_df['motif_bb_rmsd'] < 1)
    & (sc_df['global_bb_rmsd'] < 2)
]
pmpnn_pass = pmpnn_pass.groupby("name").sample(1)
pmpnn_task_pass = pmpnn_pass.groupby("task").count()

designable = sc_df[(sc_df['motif_all_atom_rmsd'] < 1.5) & (sc_df['global_bb_rmsd'] < 2)]

# print(pmpnn_task_pass)
# print(pmpnn_task_pass.index)
# pmpnn_passed_tasks = pmpnn_task_pass.sample(1)['task'].unique().tolist()

pz_pass = folding_df[
    (folding_df['motif_all_atom_rmsd'] < 1.5)
    & (folding_df['motif_bb_rmsd'] < 1)
    & (folding_df['global_all_atom_rmsd'] < 2)
]
pz_task_pass = pz_pass.groupby("task").count()
# print(pz_task_pass)
# pz_passed_tasks = pz_task_pass.sample(1)['task'].unique().tolist()

with open("tasks_passed.txt", 'w') as fp:
    fp.write("PMPNN pass\n")
    for task in all_tasks:
        if task in pmpnn_task_pass.index:
            pass_rate = int(pmpnn_task_pass.loc[task]['name']) / 100
            fp.write(f"{task}: {pass_rate}\n")
        else:
            fp.write(f"{task}: 0.0\n")
    fp.write("\n")
    fp.write("PZ pass\n")
    for task in all_tasks:
        if task in pz_task_pass.index:
            pass_rate = int(pz_task_pass.loc[task]['name']) / 100
            fp.write(f"{task}: {pass_rate}\n")
        else:
            fp.write(f"{task}: 0\n")