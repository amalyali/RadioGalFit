"""
Use flux cut and mean-shift clustering to find modes which correspond to true galaxies within the modal set returned
by MultiNest.
A. Malyali 2018
"""
import numpy as np
from sklearn.cluster import MeanShift
import sys
import argparse
import pandas as pd
import pickle

ARCS2RAD = np.pi/648000.
FOV = 3600. * ARCS2RAD

l, m, s, a, e1, e2 = 0, 1, 2, 3, 4, 5  # define parameter indexes


def flux_filter(df_modes, flux_cut):
    """
    Filter all modes with flux below flux limit.
    :param df_modes: input df of MultiNest modal output in dictionary.
    :return: input df with boolean column describing above or below flux cut
    """
    df_modes['above_cut'] = np.where(df_modes['mean'].str[s] > flux_cut, 1, 0)
    return df_modes


def identify_clusters(df_modes):
    """
    Use meanshift algorithm to identify spatial clustering of modes returned by MultiNest.
    :param df_modes:
    :return: input df with each mode assigned a cluster id.
    """
    positions = []
    for x, y in zip(df_modes['mean'].str[l].values, df_modes['mean'].str[m].values):
        positions.append((x, y))

    positions = np.asarray(positions)

    # Break each measurement run into clusters of points by position
    ms = MeanShift(bandwidth=0.0001, bin_seeding=True)
    ms.fit(positions)

    # cluster_centers = ms.cluster_centers_
    labels = ms.labels_
    df_modes['cluster_id'] = labels
    return df_modes


def final_modes(df_modes, flux_cut):
    """
    Within each cluster, select mode with highest local log evidence as
    best estimate for that group (and therefore the source).
    :param modes: as above
    :param clusters: output of identify_clusters
    :return:
    """
    df_modes['max_in_cluster'] = df_modes.groupby(['cluster_id'])['local log-evidence'].transform(max) == df_modes['local log-evidence']
    df_modes = flux_filter(df_modes, flux_cut)
    df_modes['final'] = np.where((df_modes['max_in_cluster'] == True) & (df_modes['above_cut'] == True), 1, 0)
    return df_modes


if __name__ == "__main__":
    """
    Load MultiNest output.
    1. Find cluster centres
    2. Select mode in each cluster with highest local log evidence
    3. Apply a flux cut
    4. Save final selected modes to a text file. 
    """
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('filename', help='filename of the output file produced by GalNest')
    parser.add_argument('s_cut', help='Flux cut')
    args = parser.parse_args(sys.argv[1:])

    FLUX_CUT = float(args.s_cut)
    pickled_multinest_output = args.filename

    results_pickle = './data/seed1_100_0.8_0.1_.pkl'
    data = pd.read_pickle(results_pickle)
    df = pd.DataFrame.from_dict(data['modes'])
    df_final = final_modes(identify_clusters(df), FLUX_CUT)

    # Write results to file
    output_pickle = './data/seed1_100_0.8_0.1_final.pkl'
    with open(output_pickle, 'wb') as f:
        pickle.dump(df_final, f)
