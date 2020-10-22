import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

from eps_local_opt_fairlet import load_data_with_color, calculate_obj, random_fairlet_decompose, calculate_point_fairlet_wise, local_swap
from helper_functions import subsample, calculate_distance, calculate_hc_obj, avlk_with_fairlets

from scipy.cluster.hierarchy import average as scipy_avlk
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import pdist

if __name__ == "__main__":
    filename = "./adult.csv"
    b = 1
    r = 3
    eps = 1
    rho = 0
    num = 1600
    random.seed(0)
    np.random.seed(0)

    #plot the plots that show the trend for two objectives
    print("number of samples: %d" %num)
    blue_points, red_points = load_data_with_color(filename)
    blue_points, red_points = subsample(blue_points, red_points, num)

    data = []
    data.extend(blue_points)
    data.extend(red_points)
    data = np.array(data)

    # first calculate the pairwise distance for all points

    dist, d_max = calculate_distance(data)

    total_dist = np.sum(np.sum(dist)) / 2

    upper_bound = total_dist * dist.shape[0]
    print("upper bound: %f" % upper_bound)

    Z = pdist(dist)
    cluster_matrix = scipy_avlk(Z)
    scipy_root = to_tree(cluster_matrix)

    avlk_obj, _ = calculate_hc_obj(dist, scipy_root)
    print("the cohen-addad obj for original avlk:")
    print(avlk_obj)

    B = len(blue_points)
    R = len(red_points)

    #obj_list records the fairness objective value, every 100 swaps
    obj_list = []
    #fair_obj_list records the fair hierarchical clustering tree's objective value, every 100 swaps
    fair_obj_list = []



    #start with a random fairlet decomposition, validate after every 100 swaps
    fairlets = random_fairlet_decompose(B, R, b, r)
    random_fairlets = copy.deepcopy(fairlets)
    num_f = len(fairlets)


    # take records after every 100 swaps
    counter = 100
    swap_counter = 0
    random_counter = 0

    # make a dictionary of mapping: from points to the fairlet
    #dict[u] = (number of fairlet u belongs to, index of u in the corresponding color list in the fairlet)
    dict = {}
    for i in range(num_f):
        for j in range(len(fairlets[i][0])):
            u = fairlets[i][0][j]
            dict[u] = (i, j)
        for j in range(len(fairlets[i][1])):
            v = fairlets[i][1][j]
            dict[v] = (i, j)

    point_fairlet_dist = calculate_point_fairlet_wise(dist, dict, fairlets)
    obj = calculate_obj(fairlets, dist)
    token = 0
    Delta = d_max / (num * num)

    while token == 0:
        if obj <= Delta:
            break
        token = 1
        old_obj = obj

        # randomly swap red and blue points, if not found in O(n^(1+\rho)) times, just move on
        #k is the constant in O(n^(1+\rho))
        k = 1
        for t in range(math.ceil(k * num ** (1 + rho))):
            random_counter += 1
            i = random.randint(0, num - 1)
            if i <= B - 1:
                color = 0
                j = random.randint(0, B - 1)
            else:
                color = 1
                j = random.randint(B, num - 1)
            obj, swap_or_not = local_swap(fairlets, dict, dist, point_fairlet_dist, i, j, color, obj, eps)
            if swap_or_not:
                swap_counter += 1
                token = 0

                #if swap % counter == 0 take records
                if swap_counter % counter == 0:
                    fairness_obj = calculate_obj(fairlets, dist)
                    obj_list.append(fairness_obj)

                    fair_root = avlk_with_fairlets(dist, fairlets)
                    fair_obj, _ = calculate_hc_obj(dist, fair_root)
                    fair_obj_list.append(fair_obj)

    fair_root = avlk_with_fairlets(dist, fairlets)
    fair_obj, _ = calculate_hc_obj(dist, fair_root)
    print("the cohen-addad obj for fair avlk:")
    print(fair_obj)
    print("fair avlk accounts for: %f percent of the unfair avlk." % float(fair_obj / avlk_obj))

    swaps_axis = [counter * x for x in range(len(fair_obj_list))]
    cohen_addad_ratio_list = [float(x / avlk_obj) for x in fair_obj_list]
    intra_dist_ratio_list = [float(x / total_dist) for x in obj_list]
    plt.plot(swaps_axis, cohen_addad_ratio_list)
    plt.xlabel("# of swaps")
    plt.ylabel("fair cohen-addad obj /unfair obj")
    filename1 = "cohen_addad_every_{}_swaps_".format(str(counter)) + str(num) + "pts.pdf"
    plt.savefig(filename1)
    plt.show()
    plt.clf()

    plt.plot(swaps_axis, intra_dist_ratio_list)
    plt.xlabel("# of swaps")
    plt.ylabel("fairness obj / total dist")
    filename2 = "fairness_obj_every_{}_swaps_".format(str(counter)) + str(num) + "pts.pdf"
    plt.savefig(filename2)
    plt.show()
    plt.clf()
