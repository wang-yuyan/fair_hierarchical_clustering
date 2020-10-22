#this file is the helper file for the average linkage construction,
import numpy as np
import math
import random
#return the color of a point
def which_color(color_nums, pt):
    #input: color_nums is a list where color_nums[i] is the number of points with the ith colors
    #by default we assume that all data points are [1, ..., n] and they are aligned
    #according to their colors: color_nums[0] pts with color 0,  then  color_nums[1] pts with color 1, ...
    color = 0
    end = color_nums[0]

    while pt >= end:
        color += 1
        if color >= len(color_nums):
            print("error in finding color for a point")
        end += color_nums[color]
    return color

#calculate the average distance matrix of fairlets, according to the distance matrix dist
#every fairlet is two-color
def inter_fairlet_dist(dist, fairlets):
    m = len(fairlets)
    fairlets_dist = np.zeros((m, m))
    fairlets_flatten = []
    for i in range(m):
        x = []
        x.extend(fairlets[i][0])
        x.extend(fairlets[i][1])
        fairlets_flatten.append(x)

    for i in range(1,m):
        for j in range(i):
            fairlets_dist[i][j] = np.sum(np.sum(dist[fairlets_flatten[i]][:,fairlets_flatten[j]]))
            fairlets_dist[j][i] = fairlets_dist[i][j]
    return fairlets_dist

#calculate the average distance matrix of fairlets, according to the distance matrix dist
#every fairlet is multi-color
def inter_fairlet_dist_multi_color(dist, fairlets):
    m = len(fairlets)
    color_types = len(fairlets[0])
    fairlets_dist = np.zeros((m, m))
    fairlets_flatten = []
    for i in range(m):
        x = []
        for color in range(color_types):
            x.extend(fairlets[i][color])
        fairlets_flatten.append(x)

    for i in range(1, m):
        for j in range(i):
            fairlets_dist[i][j] = np.sum(np.sum(dist[fairlets_flatten[i]][:,fairlets_flatten[j]]))
            fairlets_dist[j][i] = fairlets_dist[i][j]
    return fairlets_dist

#calculate pairwise distance, return a matrix of pairwise distances and a maximum distance
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

#class node, every node has two children
#here we assume the tree is binary. alternatively we can assume it's not and a fairlet
#can be directly split into singleton leaves. since fairlets have small sizes this nuance
#doesn't change the tree performance much.
class Node:
    def __init__(self, id=None, left=None, right=None, count=0):
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
        if self.left is None and self.right is None:
            return True
        return False

#in two-color case, sample proportionally from red and blue points
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

#in multi-color case, sample proportionally from all colored points
def subsample_multi_color(colored_points, num):
    #colored_points: a list, colored_points[i] is the list of points of color i
    color_types = len(colored_points)
    color_nums = [len(colored_points[color]) for color in range(color_types)]
    total_pts = sum(color_nums)
    color_sample_nums = []
    #get the numbers of each color proportionally
    for i in range(color_types - 1):
        color_sample_nums.append(math.floor(num * color_nums[i] / total_pts))
    color_sample_nums.append(num - sum(color_sample_nums))
    #sample
    colored_samples = []
    for i in range(color_types):
        indices = random.sample(range(color_nums[i]), color_sample_nums[i])
        colored_samples.append([colored_points[i][index] for index in indices])
    return colored_samples

#get the children of subtree rooted at root node, return a list of children
def get_children(root):
    if root.is_leaf():
        return [root.get_id()]
    else:
        return get_children(root.get_left()) + get_children(root.get_right())

#print the whole tree
def print_tree(node, s=""):
    print(s, node.get_id(), node.get_count())
    if not node.is_leaf():
        print_tree(node.get_left(), "\t" + s[:-3] + "|--")
        print_tree(node.get_right(), "\t" + s[:-3] + "\\--")

#get the maximal clusters at some size level, return a list of clusters, each cluster is a list of points
def find_maximal_clusters(root,size):
    if root.count > size:
        return find_maximal_clusters_by_size(root.get_left(), size) + find_maximal_clusters_by_size(root.get_right(), size)
    else:
        return [get_children(root)]

#get all the clusters, at least one of its children is a leaf
def find_non_leaf_clusters(root):
    active_nodes = [root]
    #check_clusters is a list that stores the clusters that need to be checked
    check_clusters = []
    while len(active_nodes) > 0:
        some_node = active_nodes.pop()
        #add this cluster to the list of clusters that we need to check, if one of its children is a leaf
        if some_node.get_left.is_children() or some_node.get_right.is_children():
            check_clusters.append(get_children(some_node))
        if not some_node.left.is_children():
            active_nodes.append(some_node.left)
        if not some_node.right.is_children():
            active_nodes.append(some_node.right)
    return check_clusters


#calculate the balance of a given cluster b/r
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

#calculate the fairness in the multi-color case
def calculate_balance_clusters_multi_color(clusters, color_nums):
    balance = 0.0
    for cluster in clusters:
        pts_of_this_color = [0 for color in range(len(color_nums))]
        for pt in cluster:
            pts_of_this_color[which_color(color_nums, pt)] += 1
        this_balance = float(np.max(pts_of_this_color)) / len(cluster)
        if this_balance > balance:
            balance = this_balance
    return balance

#calculate cohen addad objective function using recursion, returns obj, number of children
def calculate_hc_obj(dist, root):
    if root.is_leaf():
        return 0, [root.get_id()]
    obj_left, tree_left = calculate_hc_obj(dist, root.get_left())
    obj_right, tree_right = calculate_hc_obj(dist, root.get_right())
    obj = 0.0
    count = root.get_count()
    for i in tree_left:
        for j in tree_right:
            obj += dist[i][j] * count

    #check if the number is correct
    if len(tree_left) + len(tree_right) != root.get_count():
        print("Something went wrong...")
    return obj + obj_left + obj_right, tree_left + tree_right

#calculate the average distance avg(u + v, j) from avg(u, j) and avg(v, j)
def average_dist(u, v, j, list_nodes, dist):
    count1 = list_nodes[u].get_count()
    count2 = list_nodes[v].get_count()
    return float((dist[u][j] * count1 + dist[v][j] * count2) / (count1 + count2))

#return d_min with u<v (indices)
def find_min(dist, x):
    u = 0
    v = 0
    d_min = math.inf
    for i in x:
        for j in x:
            #i and j must be different
            if i == j:
                continue
            if dist[i][j] < d_min:
                u = np.minimum(i, j)
                v = np.maximum(i, j)
                d_min = dist[i][j]
    return u, v

#transform dist into a structure like pdist so that it works with the scipy package
def condense_dist(dist):
    n = dist.shape[0]
    if n == 0 or n == 1:
        return None
    condensed = dist[0][1:]
    for i in range(1,n - 1):
        condensed = np.hstack((condensed, dist[i][i+1:]))
    condensed = np.array(condensed)
    return condensed

#when we merge left_index and right_index with left_weight and right_weight,
#update the corresponding distance matrix
def update_dist(dist, left_index, right_index, left_weight, right_weight):
    new_row = (dist[left_index,:] * left_weight + dist[right_index,:] * right_weight) / (left_weight + right_weight)
    dist = np.vstack((dist, new_row))
    new_row = np.append(new_row, 0)
    new_column = new_row.reshape((-1,1))
    dist = np.hstack((dist, new_column))
    dist = np.delete(dist, [left_index, right_index], axis=0)
    dist = np.delete(dist, [left_index, right_index], axis=1)
    return dist

#this function faciliates doing average linkage based on a group of fairlets
#dist is average distances, current_id is the id we start with
def average_linkage(dist, current_id=None, indices=None, leaves=None):
    n = dist.shape[0]
    if current_id is None:
        current_id = n
    if leaves is None:
        if indices is not None:
            leaves = [Node(id=id, left=None, right=None, count=1) for id in indices]
        else:
            leaves = [Node(id=id, left=None, right=None, count=1) for id in range(n)]

    while len(leaves) > 1:
        left_index, right_index = find_min(dist, range(len(leaves)))

        left_node = leaves[left_index]
        right_node = leaves[right_index]
        left_weight = left_node.get_count()
        right_weight = right_node.get_count()
        new_node = Node(id=current_id, left=left_node, right=right_node,count=left_weight + right_weight)
        current_id += 1
        leaves.append(new_node)
        dist = update_dist(dist, left_index, right_index, left_weight, right_weight)
        del leaves[right_index]
        del leaves[left_index]

    return leaves[0], current_id

#write an average linkage algorithm with fairlets
def avlk_with_fairlets(dist, fairlets):
    fairlet_roots = []
    n = dist.shape[0]
    m = len(fairlets)
    current_id = n
    for y in fairlets:
        x = []
        x.extend(y[0])
        x.extend(y[1])
        this_root, current_id = average_linkage(dist=dist[x][:,x], current_id=current_id, indices=x)
        fairlet_roots.append(this_root)

    fairlet_dist = inter_fairlet_dist(dist, fairlets)
    for i in range(m):
        for j in range(m):
            fairlet_dist[i][j] = fairlet_dist[i][j] / (fairlet_roots[i].get_count() * fairlet_roots[j].get_count())
    root, _ = average_linkage(dist=fairlet_dist, current_id = current_id, leaves = fairlet_roots)
    return root

#write an average linkage algorithm with multi-color fairlets
def avlk_with_fairlets_multi_color(dist, fairlets):
    #count the number of colors
    color_types = len(fairlets[0])
    fairlet_roots = []
    n = dist.shape[0]
    m = len(fairlets)
    current_id = n
    for y in fairlets:
        x = []
        for color in range(color_types):
            x.extend(y[color])
        this_root, current_id = average_linkage(dist=dist[x][:,x], current_id=current_id, indices=x)
        fairlet_roots.append(this_root)

    fairlet_dist = inter_fairlet_dist_multi_color(dist, fairlets)
    for i in range(m):
        for j in range(m):
            fairlet_dist[i][j] = fairlet_dist[i][j] / (fairlet_roots[i].get_count() * fairlet_roots[j].get_count())
    root, _ = average_linkage(dist=fairlet_dist, current_id = current_id, leaves = fairlet_roots)
    return root

if __name__ == "__main__":
    data = [[1],[2],[3],[7],[8],[9],[4],[5]]
    fairlets = [[[0], [4], [7]], [[1], [5], [6]], [[2], [3], []]]
    dist, _ = calculate_distance(data)
    fair_root = avlk_with_fairlets_multi_color(dist, fairlets)
    print_tree(fair_root)
