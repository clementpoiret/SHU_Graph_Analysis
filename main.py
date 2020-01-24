import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pingouin as pg
import dabest

import utils.etl as etl

# Global Variables
AREAS = {"base": 1, "A": .5, "B": .25}
N = 4
p = 2


def create_graph(a, b):
    """Creating directed graph, with passes going from a to b
    
    Arguments:
        a {array} -- Sending vertices
        b {array} -- Receiving vertices
    
    Returns:
        G {Networkx Graph} -- Initialized graph
    """
    G = nx.DiGraph()
    G.add_nodes_from(np.unique(a.append(b).dropna().values))

    edges = list(zip(a, b))
    G.add_edges_from(edges)

    return G


def get_features(G, p_names, strategy, team, f, s):
    """Printing Degrees and (in/out) Centrality
    
    Arguments:
        G {Networkx Graph} -- Initialized graph
    """
    features = pd.DataFrame({"ID": p_names})
    features["Degrees"] = dict(G.degree()).values()
    features["Centrality"] = dict(nx.degree_centrality(G)).values()
    features["In-Centrality"] = dict(nx.in_degree_centrality(G)).values()
    features["Out-Centrality"] = dict(nx.out_degree_centrality(G)).values()
    features["Eigencentrality"] = dict(nx.eigenvector_centrality(G)).values()
    features["Harmonic_Centrality"] = dict(nx.harmonic_centrality(G)).values()
    features["Assortativity"] = nx.degree_assortativity_coefficient(G)
    features["Freedom"] = f
    features["Team"] = team
    features["Score"] = s
    features["Strategy"] = strategy

    return features


def draw_graph(G, save, path):
    """Printing the Network
    
    Arguments:
        G {Networkx Graph} -- Initialized graph
    """
    plt.figure()
    nx.draw_spectral(G, with_labels=True, font_weight='bold')

    if save:
        plt.savefig(path)


def freedom(w, p=2, α=1):
    x = (1 / len(w)) * (1 / 2) * np.dot(w, np.repeat(p, len(w)))
    return x


def stats(X, var, teams, save=True):
    """Computes Fridman Tests and Correlations for var in X
    
    Arguments:
        X {[DataFrame]} -- Dataset
        var {[list]} -- List of features to computes stats on
        teams {[list]} -- Teams to split the dataset in two groups
    
    Keyword Arguments:
        save {bool} -- [Save to CSV] (default: {True})
    """

    friedman = pd.DataFrame()
    correlation = pd.DataFrame()
    for team in teams:
        for v in var:
            frid = pg.friedman(data=X[X.Team == team],
                               dv=v,
                               within="Strategy",
                               subject="ID")
            frid["Team"] = team
            frid["Var"] = v
            friedman = friedman.append(frid)

            corr_score = pg.corr(X.loc[X.Team == team, "Score"],
                                 X.loc[X.Team == team, v])
            corr_score["Against"] = "Score"
            corr_freedom = pg.corr(X.loc[X.Team == team, "Freedom"],
                                   X.loc[X.Team == team, v])
            corr_freedom["Against"] = "Freedom"
            corr = corr_score.append(corr_freedom)
            corr["Team"] = team
            corr["Var"] = v
            correlation = correlation.append(corr)

    if save:
        friedman.to_csv("./friedman.csv", index=False)
        correlation.to_csv("./correlations.csv", index=False)

    return friedman, correlation


def main():
    """ Main Function
    Getting the dataset, Computing a Graph, and printing Degrees and Centrality.
    """

    # Getting the dataset
    X = etl.get_dataset("./Data")

    # Computing freedom
    X["Freedom"] = 0
    for area in AREAS.keys():
        f = freedom(np.repeat(AREAS[area], N), p, α=1)
        X.loc[X.Strategy == area, "Freedom"] = f

    # Creating the Graph
    features = pd.DataFrame()
    for area in AREAS.keys():
        f = freedom(np.repeat(AREAS[area], N), p, α=1)

        X_strat = X[X["Strategy"] == area]
        team_a = X_strat[X_strat["Player A"].str.isnumeric()].dropna()
        team_b = X_strat[X_strat["Player A"].str.isalpha()].dropna()
        g_a = create_graph(team_a["Player A"], team_a["Player B"])
        g_b = create_graph(team_b["Player A"], team_b["Player B"])

        # Printing interesting features
        s_a = X.loc[(X["Strategy"] == area) &
                    (X["Player A"].str.isnumeric()), "Score"].sum()
        s_b = X.loc[(X["Strategy"] == area) &
                    (X["Player B"].str.isnumeric()), "Score"].sum()
        a = get_features(g_a, range(1, 6), area, "A", f, s_a)
        b = get_features(g_b, ["A", "B", "C", "D", "E"], area, "B", f, s_b)

        features = features.append(pd.concat((a, b)))

        # Drawing the graph
        draw_graph(g_a, True, "./Networks/{}_{}".format(5, area))
        draw_graph(g_b, True, "./Networks/{}_{}".format(4, area))

    # Saving the features
    features.reset_index(drop=True).to_csv("./features.csv", index=False)

    # NHST
    var = [
        "Degrees", "Centrality", "In-Centrality", "Out-Centrality",
        "Eigencentrality", "Harmonic_Centrality", "Assortativity"
    ]
    _, _ = stats(features, var, ["A", "B"], save=True)

    # Some simple visualizations using DABEST
    wide_dataset = pd.DataFrame({
        "ID":
            features.loc[(features.Strategy == "base"), "ID"],
        "Base":
            features.loc[(features.Strategy == "base"), "Centrality"],
        "Base ":
            features.loc[(features.Strategy == "base"), "Centrality"],
        "Intervention 1":
            features.loc[(features.Strategy == "A"), "Centrality"],
        "Intervention 2":
            features.loc[(features.Strategy == "B"), "Centrality"]
    })

    paired_groups = dabest.load(wide_dataset,
                                idx=(("Base", "Intervention 1"),
                                     ("Base ", "Intervention 2")),
                                paired=True,
                                id_col="ID")
    paired_groups.hedges_g.plot()
    plt.savefig("./Centrality.pdf")
    plt.show()

    wide_dataset = pd.DataFrame({
        "Base T5":
            features.loc[(features.Strategy == "base") &
                         (features.Team == "A"), "Centrality"],
        "Int1 T5":
            features.loc[(features.Strategy == "A") &
                         (features.Team == "A"), "Centrality"],
        "Int2 T5":
            features.loc[(features.Strategy == "B") &
                         (features.Team == "A"), "Centrality"],
        "Base T4":
            features.loc[(features.Strategy == "base") &
                         (features.Team == "B"), "Centrality"],
        "Int1 T4":
            features.loc[(features.Strategy == "A") &
                         (features.Team == "B"), "Centrality"],
        "Int2 T4":
            features.loc[(features.Strategy == "B") &
                         (features.Team == "B"), "Centrality"],
    })

    unpaired_groups = dabest.load(wide_dataset,
                                  idx=(("Base T5", "Int1 T5", "Int2 T5"),
                                       ("Base T4", "Int1 T4", "Int2 T4")),
                                  paired=False)
    unpaired_groups.mean_diff.plot()
    plt.savefig("./Unpaired_Centrality.pdf")
    plt.show()


if __name__ == "__main__":
    main()
