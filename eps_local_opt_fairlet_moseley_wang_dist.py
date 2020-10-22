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
from helper_functions_moseley_wang import calculate_distance, subsample, calculate_hc_obj, avlk_with_fairlets, convert_dist, average_linkage, get_mw_upper_bound
import matplotlib.pyplot as plt

def is_float(x):
    try:
        x = float(x)
    except ValueError:
        return False
    return True

#calculates total objective value
def calculate_obj(fairlets, dist):
    obj = 0.0
    for y in fairlets:
        pts = []
        for u in y[0]:
            pts.append(u)
        for v in y[1]:
            pts.append(v)

        for i in range(1, len(pts)):
            for j in range(i):
                p1 = pts[i]
                p2 = pts[j]
                obj += dist[p1][p2]
    return obj


def load_data_with_color(filename):
    delim = ','
    red_points = []
    blue_points = []
    for line in open(filename):
        y = line.split(delim)
        n = len(y)
        for i in range(n):
            if is_float(y[i]):
                y[i] = float(y[i])
        # assume the first column denotes the color
        #blue_points <= red points
        if y[0] == 0:
            blue_points.append(y[1:])
        else:
            red_points.append(y[1:])

    return blue_points, red_points

def calculate_point_fairlet_wise(dist, dict, fairlets):
    n = len(dict)
    m = len(fairlets)

    point_fairlet_dist = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            for u in fairlets[j][0]:
                point_fairlet_dist[i][j] += dist[i][u]

            for v in fairlets[j][1]:
                point_fairlet_dist[i][j] += dist[i][v]

    return point_fairlet_dist


def get_random_fairlet(index_blue, index_red, B, R, size1, size2):

    blue_pts = list(index_blue[B - size1 : B])
    red_pts = list(index_red[R - size2 : R])
    return (blue_pts, red_pts)

#get a random fairlet decomposition
def random_fairlet_decompose(B, R, b, r):
    if float(B / R) < float(b / r):
        raise Exception("the balance of the original set is not big enough!")
    fairlets = []
    index_blue = np.array(range(B))
    index_red = np.array(range(B, B + R))
    index_blue = np.random.permutation(index_blue)
    index_red = np.random.permutation(index_red)


    while R - B >= r - b:
        new_fairlet = get_random_fairlet(index_blue,index_red, B, R, b, r)
        B -= b
        R -= r
        fairlets.append(new_fairlet)

    if R - B > 0:
        new_fairlet = get_random_fairlet(index_blue, index_red, B, R, b, b + R - B)
        B -= b
        R = B
        fairlets.append(new_fairlet)

    if R != B:
        raise Exception("R and B don't match!")

    for i in range(B):
        new_fairlet = get_random_fairlet(index_blue, index_red, B, R, 1, 1)
        B -= 1
        R -= 1
        fairlets.append(new_fairlet)

    return fairlets

#given a fairlet distance matrix and a pair of points (of same color), swap them
def local_swap(fairlets, dict, dist, point_fairlet_dist, p1, p2, color, obj, eps, fake=False):
    #update the objective function first
    n = len(dict)
    m = len(fairlets)

    f1 = dict[p1][0]
    f2 = dict[p2][0]
    if f1 == f2:
        return obj, False

    if fake == False:
        ratio = eps / n
    else:
        ratio = eps

    x1 = dict[p1][1]
    x2 = dict[p2][1]
    new_obj = obj - point_fairlet_dist[p1][f1] - point_fairlet_dist[p2][f2] + point_fairlet_dist[p1][f2] + point_fairlet_dist[p2][f1] - 2 * dist[p1][p2]

    #check if the new objective is sufficiently small

    if obj <= (1 + ratio) * new_obj:
        return obj, False
    else:
        #if new obj is small, swap the points
        fairlets[f1][color][x1] = p2
        fairlets[f2][color][x2] = p1
        dict[p1] = (f2,x2)
        dict[p2] = (f1,x1)
        #update the distances in point_fairlet_dist
        for i in range(n):
                point_fairlet_dist[i][f1] = point_fairlet_dist[i][f1] - dist[i][p1] + dist[i][p2]
                point_fairlet_dist[i][f2] = point_fairlet_dist[i][f2] - dist[i][p2] + dist[i][p1]
        #check if the two objectives are the same
        #objj = calculate_obj(fairlets, dist)
        #if abs(objj - new_obj)>=0.1:
            #print(objj)
            #print(new_obj)
            #print("WTF!! Wrong total distance!!!!")
        return new_obj, True


#validation with cohen-addad-obj
def validation_moseley_wang(simi, fairlets):
    fair_root = avlk_with_fairlets(simi, fairlets)
    fair_obj = calculate_hc_obj(simi, fair_root)
    return fair_obj


def find_eps_local_opt_random(blue_points, red_points, dist, d_max, b=1, r=1, eps=0.5, rho=0.5):
    B = len(blue_points)
    R = len(red_points)
    n = B + R
    balance_set = float(B / R)
    balance_aim = float(b / r)
    if balance_set < balance_aim:
        print(balance_set, balance_aim)
        raise ValueError("the balance of original set is too low")

    fairlets = random_fairlet_decompose(B, R, b, r)
    random_fairlets = copy.deepcopy(fairlets)
    num_f = len(fairlets)

    # take records after every 100 swaps
    counter = 100
    swap_counter = 0
    random_counter = 0

    # make a dictionary of mapping: from points to the fairlet
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
    Delta = d_max / (n * n)

    while token == 0:
        if obj <= Delta:
            break
        token = 1
        old_obj = obj

        # randomly swap red and blue points, if not found in O(n^(1+\rho)) times, just move on
        k = 1
        for t in range(math.ceil(k * n ** (1 + rho))):
            random_counter += 1
            i = random.randint(0, n - 1)
            if i <= B - 1:
                color = 0
                j = random.randint(0, B - 1)
            else:
                color = 1
                j = random.randint(B, n - 1)
            obj, swap_or_not = local_swap(fairlets, dict, dist, point_fairlet_dist, i, j, color, obj, eps)
            if swap_or_not:
                swap_counter += 1
                token = 0

    return fairlets, swap_counter, random_counter, random_fairlets



if __name__ == "__main__":
    filename = "./adult.csv"
    b = 1
    r = 3
    eps = 1
    rho = -0.3
    num_list = [100, 200, 400, 800]
    #num = 100
    random.seed(0)
    np.random.seed(0)
    for num in num_list:
        print("number of samples: %d" %num)
        blue_points, red_points = load_data_with_color(filename)
        blue_points, red_points = subsample(blue_points, red_points, num)

        data = []
        data.extend(blue_points)
        data.extend(red_points)
        data = np.array(data)

        # first calculate the pairwise distance for all points
        start = time.time()
        dist, d_max = calculate_distance(data)

        simi = convert_dist(dist)
        moseley_wang_ub = get_mw_upper_bound(simi)
        total_dist = np.sum(np.sum(dist)) / 2
        total_simi = moseley_wang_ub / (num - 2)

        avlk_root, _ = average_linkage(simi)
        avlk_obj = calculate_hc_obj(simi, avlk_root)
        print("the moseley-wang obj for original avlk:")
        print(avlk_obj)

        fairlets, swap_counter, random_counter, _ = find_eps_local_opt_random(blue_points, red_points, dist, d_max, b, r, eps, rho)
        end = time.time()
        print("total swaps: %d" %swap_counter)
        print("total randomization: %d" %random_counter)

        print("total time spent: %f s" % (end - start))
        fairness_obj = calculate_obj(fairlets, dist)
        print("the fairness objective function ratio is:%f" %float(fairness_obj / total_dist))

        fair_root = avlk_with_fairlets(simi, fairlets)
        fair_obj = calculate_hc_obj(simi, fair_root)
        print("the moseley-wang obj for fair avlk:")
        print(fair_obj)
        print("fair avlk accounts for: %f percent of the unfair avlk." %float(fair_obj/avlk_obj))

