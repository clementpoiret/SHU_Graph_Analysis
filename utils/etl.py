import pandas as pd

from pathlib import Path


def nacsports_correction(file):
    with open(file, "r+") as f:
        content = f.read()
        f.seek(0)
        f.truncate()
        f.write(content.replace("'", ""))


def get_dataset(path, save=True):
    p = Path(path)
    files = p.glob("*.csv")

    dataset = pd.DataFrame()
    for file in files:
        nacsports_correction(file)

        d = pd.read_csv(
            file, names=["Start", "Stop", "Player A", "Player B", "Score"])
        d = d.drop(["Start", "Stop"], axis=1)
        d["Player A"] = d["Player A"].str[-1]
        d["Player B"] = d["Player B"].str[-1]
        d.Score = d.Score.fillna(0)
        d.loc[d.Score == "Score 1", "Score"] = 1
        d.loc[d.Score == "Score 2", "Score"] = 1

        teams, strategy = file.stem.split("_")
        team_a, team_b = teams.split("v")
        d["Team A"] = team_a
        d["Team B"] = team_b
        d["Strategy"] = strategy

        dataset = dataset.append(d)

        if save:
            dataset.to_csv("./dataset.csv")

    return dataset.reset_index(drop=True)
