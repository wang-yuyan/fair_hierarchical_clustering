#this file contains the eps/n locally opt algorithm
import numpy as np
import random
import math
import time
import copy
import cProfile
import pstats
import io

from scipy.cluster.hierarchy import average as scipy_avlk
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import pdist
from helper_functions import calculate_distance, subsample_multi_color, find_maximal_clusters, calculate_balance_clusters, calculate_hc_obj,condense_dist, avlk_with_fairlets_multi_color, print_tree
import matplotlib.pyplot as plt

def is_float(x):
    try:
        x = float(x)
    except ValueError:
        return False
    return True

#calculates total objective value
def calculate_obj_multi_color(fairlets, dist):
    obj = 0.0
    color_types = len(fairlets[0])
    for y in fairlets:
        pts = []
        for color in range(color_types):
            pts.extend(y[color])

        for i in range(1, len(pts)):
            for j in range(i):
                p1 = pts[i]
                p2 = pts[j]
                obj += dist[p1][p2]
    return obj

def load_data_with_multi_color(filename, color_types):
    delim = ','
    colored_points = [[] for color in range(color_types)]
    for line in open(filename):
        y = line.split(delim)
        n = len(y)
        for i in range(n):
            if is_float(y[i]):
                y[i] = float(y[i])
        # assume the first column denotes the color
        color = int(y[0])
        colored_points[color].append(y[1:])
    return colored_points

def calculate_point_fairlet_wise_multi_color(dist, dict, fairlets):
    n = len(dict)
    m = len(fairlets)
    color_types = len(fairlets[0])

    point_fairlet_dist = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for color in range(color_types):
                for u in fairlets[j][color]:
                    point_fairlet_dist[i][j] += dist[i][u]

    return point_fairlet_dist

def random_fairlet_decompose_multi_color(color_nums, alpha):
    color_types = len(color_nums)
    total_num = sum(color_nums)
    t = math.floor(total_num / alpha)
    fairlets = [[[], [], [], []] for i in range(t)]
    color = 0
    colored_pt_list = list(np.random.permutation(range(color_nums[0])))
    fairlet_id = 0
    while True:
        pt = colored_pt_list.pop()
        fairlets[fairlet_id][color].append(pt)
        fairlet_id += 1
        if fairlet_id >= t:
            fairlet_id = 0
        if colored_pt_list == []:
            if color == color_types - 1:
                break
            else:
                color += 1
                begin = sum(color_nums[ : color])
                end = sum(color_nums[ : color + 1])
                colored_pt_list = list(np.random.permutation(range(begin, end)))
    return fairlets

#given a fairlet distance matrix and a pair of points (of same color), swap them
def local_swap_multi_color(fairlets, dict, dist, point_fairlet_dist, p1, p2, color, obj, eps):
    #update the objective function first
    n = len(dict)
    m = len(fairlets)

    f1 = dict[p1][0]
    f2 = dict[p2][0]

    if f1 == f2:
        return obj, False

    ratio = eps / n

    x1 = dict[p1][2]
    x2 = dict[p2][2]
    new_obj = obj - point_fairlet_dist[p1][f1] - point_fairlet_dist[p2][f2] + point_fairlet_dist[p1][f2] + point_fairlet_dist[p2][f1] - 2 * dist[p1][p2]

    #check if the new objective is sufficiently small

    if obj <= (1 + ratio) * new_obj:
        return obj, False
    else:
        #if new obj is small, swap the points
        fairlets[f1][color][x1] = p2
        fairlets[f2][color][x2] = p1
        dict[p1] = (f2, color, x2)
        dict[p2] = (f1, color, x1)
        #update the distances in point_fairlet_dist
        for i in range(n):
                point_fairlet_dist[i][f1] = point_fairlet_dist[i][f1] - dist[i][p1] + dist[i][p2]
                point_fairlet_dist[i][f2] = point_fairlet_dist[i][f2] - dist[i][p2] + dist[i][p1]

        #check if the two objectives are the same
        #objj = calculate_obj_multi_color(fairlets, dist)
        #if abs(objj - new_obj)>=0.1:
            #print(objj)
            #print(new_obj)
            #print("WTF!! Wrong total distance!!!!")

    return new_obj, True

def find_eps_local_opt_random_multi_color(colored_points, dist, d_max, alpha=3, eps=0.5, rho=0.5):
    color_types = len(colored_points)
    color_nums = [len(colored_points[color]) for color in range(color_types)]
    n = sum(color_nums)
    balance_set = float(color_nums[0] / n)
    balance_aim = float(1 / alpha)
    if balance_set > balance_aim:
        print(balance_set, balance_aim)
        raise ValueError("the balance of original set is too low")

    fairlets = random_fairlet_decompose_multi_color(color_nums, alpha)

    random_fairlets = copy.deepcopy(fairlets)
    num_f = len(fairlets)

    # take records after every 100 swaps
    swap_counter = 0
    random_counter = 0

    # make a dictionary of mapping: from points to the fairlet
    dict = {}
    for i in range(num_f):
        for j in range(color_types):
            for k in range(len(fairlets[i][j])):
                pt = fairlets[i][j][k]
                dict[pt] = (i, j, k)

    point_fairlet_dist = calculate_point_fairlet_wise_multi_color(dist, dict, fairlets)

    obj = calculate_obj_multi_color(fairlets, dist)
    token = 0
    Delta = d_max / (n * n)

    while token == 0:
        if obj <= Delta:
            break
        token = 1
        old_obj = obj

        # randomly swap points of the same color, if not found in O(n^(1+\rho)) times, just move on
        k = 1
        for t in range(math.ceil(k * n ** (1 + rho))):
            random_counter += 1
            i = random.randint(0, n - 1)
            color = dict[i][1]
            #assert color == dict[i][1]
            begin = sum(color_nums[ : color])
            end = begin + color_nums[color]
            j = random.randint(begin, end - 1)
            obj, swap_or_not = local_swap_multi_color(fairlets, dict, dist, point_fairlet_dist, i, j, color, obj, eps)

            if swap_or_not:
                swap_counter += 1
                token = 0

    return fairlets, swap_counter, random_counter, random_fairlets

if __name__ == "__main__":
    filename = "./adult_4_color.csv"
    alpha = 3
    eps = 1
    rho = 0
    num_list = [100]
    color_types = 4
    #num = 100
    random.seed(0)
    np.random.seed(0)
    for num in num_list:
        print("number of samples: %d" % num)
        colored_points = load_data_with_multi_color(filename, color_types)
        colored_points = subsample_multi_color(colored_points, num)
        data = []
        for sample in colored_points:
            data.extend(sample)
        data = np.array(data)

        # first calculate the pairwise distance for all points
        start = time.time()
        dist, d_max = calculate_distance(data)

        total_dist = np.sum(np.sum(dist)) / 2


        #upper_bound = total_dist * dist.shape[0]
        #print("upper bound: %f" % upper_bound)

        Z = pdist(dist)
        cluster_matrix = scipy_avlk(Z)
        scipy_root = to_tree(cluster_matrix)

        avlk_obj, _ = calculate_hc_obj(dist, scipy_root)
        print("the cohen-addad obj for original avlk:")
        print(avlk_obj)


        fairlets, swap_counter, random_counter, _ = find_eps_local_opt_random_multi_color(colored_points, dist, d_max, alpha, eps, rho)
        end = time.time()
        print("total swaps: %d" %swap_counter)
        print("total randomization: %d" %random_counter)
        print(fairlets)

        print("total time spent: %f s" % (end - start))
        fairness_obj = calculate_obj_multi_color(fairlets, dist)
        print("the fairness objective function ratio is:%f" %float(fairness_obj / total_dist))


        fair_root = avlk_with_fairlets_multi_color(dist, fairlets)
        print_tree(fair_root)
        fair_obj, _ = calculate_hc_obj(dist, fair_root)
        print("the cohen-addad obj for fair avlk:")
        print(fair_obj)
        print("fair avlk accounts for: %f percent of the unfair avlk." %float(fair_obj/avlk_obj))
        '''
        swaps_axis = [100 * x for x in range(len(fair_obj_list))]
        cohen_addad_ratio_list = [float(x / avlk_obj) for x in fair_obj_list]
        intra_dist_ratio_list = [float(x / total_dist) for x in obj_list]
        plt.plot(swaps_axis, cohen_addad_ratio_list)
        plt.xlabel("# of swaps")
        plt.ylabel("fair cohen-addad obj /unfair obj")
        filename1 = "cohen_addad_every_100_swaps_" + str(num) + "pts_fake.pdf"
        plt.savefig(filename1)
        plt.show()
        plt.clf()
        plt.plot(swaps_axis, intra_dist_ratio_list)
        plt.xlabel("# of swaps")
        plt.ylabel("fairness obj / total dist")
        filename2 = "fairness_obj_every_100_swaps_" + str(num) + "pts_fake.pdf"
        plt.savefig(filename2)
        plt.show()
        plt.clf()
        '''