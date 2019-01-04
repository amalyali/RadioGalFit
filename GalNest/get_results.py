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
    :param df_modes:
    :return: filtered mode list
    """
    df_modes['above_cut'] = np.where(df_modes['mean'].str[s] > flux_cut, 1, 0)
    return df_modes


def identify_clusters(modes):
    """
    Use meanshift algorithm to identify spatial clustering of modes returned by MultiNest.
    :param modes:
    :return: all identified clusters.
    """
    positions = []

    for x, y in zip(modes[:,3], modes[:,7]):
        positions.append((x,y))

    positions = np.asarray(positions)

    # Break each measurement run into clusters of points by position
    ms = MeanShift(bandwidth=0.0001, bin_seeding=True)
    ms.fit(positions)

    # cluster_centers = ms.cluster_centers_
    labels = ms.labels_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    clusters = []
    for i in range(n_clusters_):
        # select only data observations with cluster label == i
        clusters.append(positions[np.where(labels==i)])

    return clusters


def match_positions_2_array(a,B):
    for b in B:
        if (np.array_equal(a,np.array([b[3],b[7]]))):
            return b


def final_modes(modes, clusters):
    """
    Within each cluster, select mode with highest local log evidence as
    best estimate for that group (and therefore the source).
        Match each item in cluster, with position from original modes
            ie. we link with fluxes, shapes...
        Then, add the highest local log evidence of each cluster
            to the final modes array.
    :param modes: as above
    :param clusters: output of identify_clusters
    :return:
    """
    final_modes = []

    for cluster in clusters:
        mode_vals = []
        for position in cluster:
            position = match_positions_2_array(position, modes)
            mode_vals.append(position)

        mode_vals = np.asarray(mode_vals)
        final_modes.append(mode_vals[np.argmax(mode_vals[:,1], axis=0)])

    final_modes = np.asarray(final_modes)
    return final_modes


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