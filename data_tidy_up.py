import numpy as np
import matplotlib.pyplot as plt
import math


if __name__ == "__main__":
    output_direc = ". /experiments_random_swap_multi_color"
    filename = "adult_4_color.csv"
    num_list = [2**k * 100 for k in range(5)]
    b = 1
    r = 7

    num_instances = 5

    time_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0], "time", "b=" + str(b),
                                               "r=" + str(r))
    balance_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0], "balance", "b=" + str(b),
                                                  "r=" + str(r))
    obj_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0], "obj", "b=" + str(b), "r=" + str(r))
    ratio_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0], "ratio", "b=" + str(b),
                                                "r=" + str(r))
    time_f = open(time_file, "r")
    balance_f = open(balance_file, "r")
    obj_f = open(obj_file, "r")
    ratio_f = open(ratio_file, "r")

    times = []
    for size in num_list:
        line = time_f.readline().strip()
        y = line.split(' ')
        times.append([float(y[0]),np.average(np.array([y[i + 1] for i in range(num_instances)],dtype=float)), np.std(np.array([y[i + 1] for i in range(num_instances)],dtype=float))])

    np.savetxt("{}/{}_{}.txt".format(output_direc, filename.split('.')[0], "time") , np.array(times), delimiter=' ', fmt='%s')

    '''
    times = [y[1] for y in times]
    print(times)
    plt.plot(num_list, times)
    plt.xlabel("# of samples")
    plt.ylabel("running time/s")
    filename1 = output_direc + "/" + filename.split('.')[0] + "_running_time.pdf"
    plt.savefig(filename1)
    plt.show()
    plt.clf()
    
    times_log = [math.log(y) for y in times]
    num_log = [math.log(x) for x in num_list]
    plt.plot(num_log, times_log )
    plt.xlabel("log(# of samples)")
    plt.ylabel("log(running time)")
    filename2 = output_direc + "/" + filename.split('.')[0] + "_log_running_time.pdf"
    plt.savefig(filename2)
    plt.show()
    plt.clf()
    '''
    ratios = []
    for size in num_list:
        line = ratio_f.readline().strip()
        y = line.split(' ')
        ratios.append([float(y[0]), np.average([float(y[i + 1]) for i in range(num_instances)]), np.std([float(y[i + 1]) for i in range(num_instances)])])

    np.savetxt("{}/{}_{}.txt".format(output_direc, filename.split('.')[0], "ratio") , np.array(ratios), delimiter=' ',fmt='%s')


    objs = []
    next(obj_f)
    for num in num_list:
        next(obj_f)
        good_data = []
        good_ratio = []
        for i in range(num_instances):
            line = obj_f.readline().strip()
            y = line.split(' ')
            good_data.append(float(y[2]))
            good_ratio.append(float(y[1]) / float(y[3]))
        objs.append([num, np.average(good_data), np.std(good_data), np.average(good_ratio), np.std(good_ratio)])
    np.savetxt("{}/{}_{}.txt".format(output_direc, filename.split('.')[0], "obj") , np.array(objs), delimiter=' ', fmt='%s')
