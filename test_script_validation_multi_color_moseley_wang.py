from scipy.cluster.hierarchy import average as scipy_avlk
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import pdist
from helper_functions import find_maximal_clusters, calculate_distance, calculate_balance_clusters_multi_color, subsample_multi_color
from helper_functions_moseley_wang import convert_dist, average_linkage, calculate_hc_obj, get_mw_upper_bound, avlk_with_fairlets_multi_color
from eps_local_opt_fairlet_multi_color import load_data_with_multi_color, find_eps_local_opt_random_multi_color, calculate_obj_multi_color

import random
import numpy as np
import time
import os
import sys

from datetime import datetime
def test(filename, num_list, color_types, alpha, output_direc, num_instances, rho):
    colored_points = load_data_with_multi_color(filename, color_types)
    time_file = "{}/{}_{}_{}.out".format(output_direc, filename.split(".")[0], "time", "alpha="+str(alpha))
    balance_file = "{}/{}_{}_{}.out".format(output_direc, filename.split(".")[0], "balance", "alpha="+str(alpha))
    obj_file = "{}/{}_{}_{}.out".format(output_direc, filename.split(".")[0], "obj", "alpha="+str(alpha))
    ratio_file = "{}/{}_{}_{}.out".format(output_direc, filename.split(".")[0], "ratio","alpha="+str(alpha))
    swap_file = "{}/{}_{}_{}.out".format(output_direc, filename.split(".")[0], "swap","alpha="+str(alpha))
    random_file = "{}/{}_{}_{}.out".format(output_direc, filename.split(".")[0], "random","alpha="+str(alpha))

    time_f = open(time_file, "w")
    balance_f = open(balance_file, "w")
    obj_f = open(obj_file, "w")
    ratio_f = open(ratio_file, "w")
    swap_f = open(swap_file, "w")
    random_f = open(random_file, "w")

    obj_f.write("avlk_obj  fair_avlk_obj  fair/unfair avlk ratio  random_avlk_obj  random/unfair avlk ratio  upper bound\n")

    for num in num_list:
        print("sample number: %d" %num)
        time_f.write("{} ".format(num))
        obj_f.write("{}\n".format(num))
        balance_f.write("{} ".format(num))
        ratio_f.write("{} ".format(num))
        swap_f.write("{} ".format(num))
        random_f.write("{} ".format(num))

        for i in range(num_instances):
            colored_pts_sample = subsample_multi_color(colored_points, num)
            color_nums = [len(colored_points[color]) for color in range(color_types)]
            data = []
            for sample in colored_pts_sample:
                data.extend(sample)
            data = np.array(data)

            # first calculate the pairwise distance for all points

            dist, d_max = calculate_distance(data, dist_type="euclidean")
            total_dist = np.sum(np.sum(dist)) / 2
            start = time.time()
            fairlets, swap_counter, random_counter, random_fairlets = find_eps_local_opt_random_multi_color(colored_pts_sample, dist, d_max, alpha, 0.1, rho)
            end = time.time()

            dist_ratio = float(calculate_obj_multi_color(random_fairlets, dist) / total_dist)

            t = end - start
            time_f.write("{} ".format(t))
            ratio_f.write("{} ".format(dist_ratio))
            swap_f.write("{} ".format(swap_counter))
            random_f.write("{} ".format(random_counter))
            simi = convert_dist(dist)
            avlk_root, _ = average_linkage(simi)

            # find the balance of maximal clustering
            avlk_maximal_cluster = find_maximal_clusters(avlk_root, 2 * alpha)
            avlk_root_balance = calculate_balance_clusters_multi_color(avlk_maximal_cluster, color_nums)
            #print("the balance of (b+r)-maximal clusters for original average-linkage is:")
            #print(scipy_root_balance)
            balance_f.write("{} ".format(avlk_root_balance))

            # find how much we lose by being fair

            avlk_obj = calculate_hc_obj(simi, avlk_root)
            fair_root = avlk_with_fairlets_multi_color(simi, fairlets)
            fair_obj = calculate_hc_obj(simi, fair_root)
            ratio1 = float(fair_obj / avlk_obj)
            random_root = avlk_with_fairlets_multi_color(simi, random_fairlets)
            random_obj = calculate_hc_obj(simi, random_root)
            ratio2 = float(random_obj / avlk_obj)

            upper_bound = get_mw_upper_bound(simi)

            #obj_f.write("{} {} {} \n".format(avlk_obj, avlk_time, upper_bound))
            obj_f.write("{} {} {} {} {} {}\n".format(avlk_obj, fair_obj, ratio1, random_obj, ratio2, upper_bound))
            obj_f.flush()
            time_f.flush()
            balance_f.flush()
            ratio_f.flush()
            swap_f.flush()
            random_f.flush()

        time_f.write("\n")
        balance_f.write("\n")
        ratio_f.write("\n")
        swap_f.write("\n")
        random_f.write("\n")

    time_f.close()
    balance_f.close()
    obj_f.close()
    ratio_f.close()
    swap_f.close()
    random_f.close()


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    filename = "adult_4_color.csv"
    alpha = 3
    color_types = 4
    eps = 0.1
    rho = 0
    num_instances = 5
    num_list = [100, 200, 400, 800, 1600, 3200, 6400]
    np.random.seed(0)
    random.seed(0)
    output_direc = "./experiments_random_swap_multi_color_mw"
    test(filename, num_list, color_types, alpha, output_direc, num_instances, rho)