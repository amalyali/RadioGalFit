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
    df_modes['above_cut'] = np.where(df_modes['mean'].str[s] > flux_cut, 'True', 'False')
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


def final_modes(df_modes):
    """
    Within each cluster, select mode with highest local log evidence as
    best estimate for that group (and therefore the source).
    :param df_modes: as above
    :return: df with boolean column identifying highest Z_loc mode in each cluster.
    """
    df_modes['max_in_cluster'] = df_modes.groupby(['cluster_id'])['local log-evidence'].transform(max) == df_modes[
        'local log-evidence']
    return df_modes


if __name__ == "__main__":
    """
    Load MultiNest output.
    1. Apply a flux cut
    2. Find cluster centres
    3. Select mode in each cluster with highest local log evidence
    4. Sort final selected modes via l positional value
    5. Save final selected modes to a text file. 
    """
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('filename', help='filename of the output file produced by GalNest')
    parser.add_argument('s_cut', help='Flux cut')
    args = parser.parse_args(sys.argv[1:])

    FLUX_CUT = float(args.s_cut)
    pickled_multinest_output = args.filename
    #results_pickle = '%s/seed%s_%s_%s_%s_.pkl' % (DATA_DIR, SEED, N_LIVE, S_EFF, EV_TOL)
    results_pickle = './seed1_100_0.8_0.1_.pkl'
    data = pd.read_pickle(results_pickle)
    df = pd.DataFrame.from_dict(data['modes'])

    flux_filter(df, FLUX_CUT)



    data = np.loadtxt(multinest_output_file)
    data = flux_filter(data, FLUX_CUT)
    cluster_centres = identify_clusters(data)
    best_modes = final_modes(data, cluster_centres)
    best_modes = best_modes[best_modes[:, 3].argsort()[::-1]]
    np.savetxt('best_modes.txt', best_modes)