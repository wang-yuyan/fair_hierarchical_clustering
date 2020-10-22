#this file is the helper file for the average linkage tracking
import numpy as np
import math
import time
import random

def inter_fairlet_simi(simi, fairlets):
    m = len(fairlets)
    fairlets_simi = np.zeros((m, m))
    fairlets_flatten = []
    for i in range(m):
        x = []
        x.extend(fairlets[i][0])
        x.extend(fairlets[i][1])
        fairlets_flatten.append(x)

    for i in range(1,m):
        for j in range(i):
            fairlets_simi[i][j] = np.sum(np.sum(simi[fairlets_flatten[i]][:,fairlets_flatten[j]]))
            fairlets_simi[j][i] = fairlets_simi[i][j]
    return fairlets_simi

def inter_fairlet_simi_multi_color(simi, fairlets):
    m = len(fairlets)
    color_types = len(fairlets[0])
    fairlets_simi = np.zeros((m, m))
    fairlets_flatten = []
    for i in range(m):
        x = []
        for color in range(color_types):
            x.extend(fairlets[i][color])
        fairlets_flatten.append(x)

    for i in range(1, m):
        for j in range(i):
            fairlets_simi[i][j] = np.sum(np.sum(simi[fairlets_flatten[i]][:,fairlets_flatten[j]]))
            fairlets_simi[j][i] = fairlets_simi[i][j]
    return fairlets_simi

#calculate the upper bound of mw objective using similarity
def get_mw_upper_bound(simi):
    edges = 0.0
    n = simi.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            edges += simi[i][j]
    return edges * (n-2)

def calculate_distance(points, dist_type = "euclidean"):
    n = len(points)
    dist = np.zeros((n,n))
    d_max = 0.0

    for i in range(1,n):
        for j in range(i):
            dif = np.array(points[i]) - np.array(points[j])
            d = np.sqrt(np.einsum("i,i->", dif, dif))
            dist[i][j] = d
            dist[j][i] = d
            if d > d_max:
                d_max = d
    return dist, d_max

#simi = 1/(1+d)
def convert_dist(dist):
    n = dist.shape[0]
    simi = np.copy(dist)
    for i in range(n):
        for j in range(n):
            if i != j:
                simi[i][j] = 1 / (simi[i][j] + 1)
    return simi

class Node:
    def __init__(self, id = None, left = None, right = None, count = 0):
        self.id = id
        self.left = left
        self.right = right
        self.count = count

    def get_count(self):
        return self.count

    def get_id(self):
        return self.id

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def is_leaf(self):
        if self.left == None and self.right == None:
            return True
        return False


def subsample(blue_points, red_points, num):
    B = len(blue_points)
    R = len(red_points)
    blue_num = math.ceil(num * B / (B + R))
    red_num = num - blue_num
    blue_index = random.sample(range(B), blue_num)
    red_index = random.sample(range(R), red_num)
    blue_points_sample = []
    red_points_sample = []
    for i in blue_index:
        blue_points_sample.append(blue_points[i])
    for j in red_index:
        red_points_sample.append(red_points[j])

    return blue_points_sample, red_points_sample


def get_children(root):
    if root.is_leaf():
        return [root.get_id()]
    else:
        return get_children(root.get_left()) + get_children(root.get_right())


def print_tree(node, s=""):
    print(s, node.get_id(), node.get_count())
    if not node.is_leaf():
        print_tree(node.get_left(), "\t" + s[:-3] + "|--")
        print_tree(node.get_right(), "\t" + s[:-3] + "\\--")


def find_maximal_clusters(root,size):
    if root.count > size:
        return find_maximal_clusters(root.get_left(), size) + find_maximal_clusters(root.get_right(), size)
    else:
        return [get_children(root)]

def calculate_balance_clusters(clusters, B):
    balance = 1.0
    for cluster in clusters:
        blue = 0
        red = 0
        for u in cluster:
            if u < B:
                blue += 1
            else:
                red += 1
        if blue == 0 or red == 0:
            return 0
        this_balance = np.minimum(float(blue / red),float(red / blue))
        if this_balance < balance:
            balance = this_balance
    return balance

#calculate moseley wang objective function using recursion, returns obj, children
def calculate_hc_obj(simi, root):
    n = simi.shape[0]
    current_nodes = [root]
    obj = 0.0
    while len(current_nodes) > 0:
        parent = current_nodes.pop()
        left_child = parent.left
        right_child = parent.right
        if left_child is None or right_child is None:
            continue
        current_nodes.append(left_child)
        current_nodes.append(right_child)

        left_children = get_children(left_child)
        right_children = get_children(right_child)
        term = n - len(left_children) - len(right_children)

        for i in left_children:
            for j in right_children:
                obj += simi[i][j] * term

    return obj

#calculate average similarity between nodes
def average_simi(u,v,j,list_nodes, simi):
    count1 = list_nodes[u].get_count()
    count2 = list_nodes[v].get_count()
    return float((simi[u][j] * count1 + simi[v][j] * count2) / (count1 + count2))


#return s_max with u<v (indices)
def find_max(simi, x):
    u = 0
    v = 0
    d_max = -math.inf
    for i in x:
        for j in x:
            if i == j:
                continue
            if simi[i][j] > d_max:
                u = np.minimum(i,j)
                v = np.maximum(i,j)
                d_max = simi[i][j]
    return u,v


#returns an array like pdist
def condense_dist(dist):
    n = dist.shape[0]
    if n == 0 or n == 1:
        return None
    condensed = dist[0][1:]
    for i in range(1,n - 1):
        condensed = np.hstack((condensed, dist[i][i+1:]))
    condensed = np.array(condensed)
    return condensed

def update_simi(simi, left_index, right_index, left_weight, right_weight):
    new_row = (simi[left_index,:] * left_weight + simi[right_index,:] * right_weight) / (left_weight + right_weight)
    simi = np.vstack((simi, new_row))
    new_row = np.append(new_row, 0)
    new_column = new_row.reshape((-1,1))
    simi = np.hstack((simi, new_column))
    simi = np.delete(simi, [left_index, right_index], axis=0)
    simi = np.delete(simi, [left_index, right_index], axis=1)
    return simi


def average_linkage(simi, current_id=None, indices=None, leaves=None):
    n = simi.shape[0]
    if current_id is None:
        current_id = n
    if leaves is None:
        if indices is not None:
            leaves = [Node(id=id, left=None, right=None, count=1) for id in indices]
        else:
            leaves = [Node(id=id, left=None, right=None, count=1) for id in range(n)]

    while len(leaves) > 1:
        left_index, right_index = find_max(simi, range(len(leaves)))

        left_node = leaves[left_index]
        right_node = leaves[right_index]
        left_weight = left_node.get_count()
        right_weight = right_node.get_count()
        new_node = Node(id=current_id, left=left_node, right=right_node,count=left_weight + right_weight)
        current_id += 1
        leaves.append(new_node)
        simi = update_simi(simi, left_index, right_index, left_weight, right_weight)
        del leaves[right_index]
        del leaves[left_index]

    return leaves[0], current_id

#write an average linkage algorithm with fairlets
def avlk_with_fairlets(simi, fairlets):
    fairlet_roots = []
    n = simi.shape[0]
    m = len(fairlets)
    current_id = n
    for y in fairlets:
        x = []
        x.extend(y[0])
        x.extend(y[1])
        this_root, current_id = average_linkage(simi=simi[x][:,x], current_id=current_id, indices=x)
        fairlet_roots.append(this_root)

    fairlet_simi = inter_fairlet_simi(simi, fairlets)
    for i in range(m):
        for j in range(m):
            fairlet_simi[i][j] = fairlet_simi[i][j] / (fairlet_roots[i].get_count() * fairlet_roots[j].get_count())
    root, _ = average_linkage(simi=fairlet_simi, current_id = current_id, leaves = fairlet_roots)
    return root

def avlk_with_fairlets_multi_color(simi, fairlets):
    color_types = len(fairlets[0])
    fairlet_roots = []
    n = simi.shape[0]
    m = len(fairlets)
    current_id = n
    for y in fairlets:
        x = []
        for color in range(color_types):
            x.extend(y[color])
        this_root, current_id = average_linkage(simi=simi[x][:,x], current_id=current_id, indices=x)
        fairlet_roots.append(this_root)

    fairlet_simi = inter_fairlet_simi_multi_color(simi, fairlets)
    for i in range(m):
        for j in range(m):
            fairlet_simi[i][j] = fairlet_simi[i][j] / (fairlet_roots[i].get_count() * fairlet_roots[j].get_count())
    root, _ = average_linkage(simi=fairlet_simi, current_id = current_id, leaves = fairlet_roots)
    return root

if __name__ == "__main__":
    data = [[1],[2],[3],[7],[8],[9],[4],[5]]
    fairlets = [([0],[4,3]),([1],[5,6]),([2],[7])]
    dist, _ = calculate_distance(data)
    simi = convert_dist(dist)
    root, _ = average_linkage(simi)
    print_tree(root)
    fair_root = avlk_with_fairlets(simi, fairlets)
    print_tree(fair_root)
    print(get_mw_upper_bound(simi))
    print(calculate_hc_obj(simi, root))
    print(calculate_hc_obj(simi, fair_root))
