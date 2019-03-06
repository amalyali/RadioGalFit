"""
Use mean-shift clustering and SNR cut to find modes which correspond to true galaxies within the modal set returned
by MultiNest.
A. Malyali 2019
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


def snr_filter(df_modes, snr_cut):
    """
    Filter out all modes with SNR below SNR threshold.
    """
    df_modes['above_cut'] = np.where(df_modes['SNR'] > snr_cut, 1, 0)
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


def final_modes(df_modes, cut):
    """
    Within each cluster, select mode with highest local log evidence as
    best estimate for that group (and therefore the source).
    :param modes: as above
    :param clusters: output of identify_clusters
    :return:
    """
    df_modes['max_in_cluster'] = df_modes.groupby(['cluster_id'])['local log-evidence'].transform(max) == df_modes['local log-evidence']
    df_modes = snr_filter(df_modes, cut)
    df_modes['final'] = np.where((df_modes['max_in_cluster'] == True) & (df_modes['above_cut'] == True), 1, 0)
    return df_modes


if __name__ == "__main__":
    """
    Load MultiNest output with mode SNRs.
    1. Find cluster centres
    2. Select mode in each cluster with highest local log evidence
    3. Apply a SNR cut
    4. Save final selected modes. 
    """
    # TODO: rename get_results_SNR_pickle.py -> compute_mode_SNR.py
    # TODO: rename + update contents of run_snr_cut.sh - > run_compute_mode_SNR.sh
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('filename', help='pickle output file produced by compute_mode_SNR.py')  # TODO
    parser.add_argument('snr_cut', help='SNR cut')
    args = parser.parse_args(sys.argv[1:])

    snr_cut = float(args.snr_cut)
    pickled_multinest_output = args.filename
    output_results_pickle = './data/seed1_100_0.8_0.1_final.pkl'

    # Load in modes, then cluster + perform SNR cut.
    data = pd.read_pickle(pickled_multinest_output)
    df = pd.DataFrame.from_dict(data['modes'])
    df_final = final_modes(identify_clusters(df), snr_cut)

    with open(output_results_pickle, 'wb') as f:
        pickle.dump(df_final, f)
