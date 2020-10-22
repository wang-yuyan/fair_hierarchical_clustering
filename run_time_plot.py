import numpy as np
import numpy.random as random
import scipy.stats.mstats as stats
import matplotlib.pyplot as plt
import sys
from datetime import datetime


def log_linear_reg(x, y):
    # take the log of data
    log_x = np.log10(x)
    log_y = np.log10(y)

    slope, intercept, _, _, _ = stats.linregress(log_x, log_y)

    return slope, intercept

def generate_run_plot(filename, runtime_direc, b, r, sample_sizes):
    filename = filename.split(".")[0]
    avlk_times = []
    fair_avlk_times = []
    with open("{}/{}_{}_{}{}_{}{}.{}".format(runtime_direc, filename, "time", "b=", str(b), "r=", str(r), "out")) as fair_f:
        for i in range(len(sample_sizes)):
            fair_avlk_times.append(np.mean([float(y) for y in fair_f.readline().strip().split(" ")[1:]]))
        fair_f.close()

    with open("{}/{}_{}_{}{}_{}{}.{}".format(runtime_direc, filename, "obj", "b=", str(b), "r=", str(r), "out")) as avlk_f:
        next(avlk_f)
        for i in range(len(sample_sizes)):
            next(avlk_f)
            times = []
            for j in range(5):
                line = avlk_f.readline().strip().split(" ")
                times.append(float(line[1]))
            avlk_times.append(np.mean(times))
        avlk_f.close()
    print(sample_sizes)
    print("the fair avlk times:")
    print(fair_avlk_times)
    print("the avlk times:")
    print(avlk_times)
    avlk_slope, _ = log_linear_reg(sample_sizes, avlk_times)
    fair_slope, _ = log_linear_reg(sample_sizes, fair_avlk_times)
    print("slopes are:")
    print(avlk_slope, fair_slope)
    plot_name = "./{}_{}_{}{}_{}{}.{}".format(filename, "runtime", "b=", str(b), "r=", str(r), "pdf")
    plt.plot(sample_sizes, avlk_times, "D-", label="average-linkage", color = "blue")
    plt.plot(sample_sizes, fair_avlk_times, "D-", label="fair average-linkage", color="crimson")
    plt.text(3200, 3200, "slope="+str(round(fair_slope, 2)))
    plt.text(3200, 1, "slope="+str(round(avlk_slope, 2)))
    plt.xlabel("log(sample size)")
    plt.ylabel("log(run time)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Log run time versus log sample size")
    plt.legend()
    plt.savefig(plot_name)
    return avlk_times, fair_avlk_times

if __name__ == "__main__":
    filename = "adult.csv"
    runtime_direc = "./experiments_random_swap_no_validation_rho_0"
    sample_sizes = [2**k * 100 for k in range(7)]
    b = 1
    r = 3
    avlk_times, fair_avlk_times = generate_run_plot(filename, runtime_direc, b, r, sample_sizes)
    output_direc = "./{}_{}.{}".format(filename.split(".")[0], "runtime", "txt")
    with open(output_direc, "w") as time_f:
        time_f.write("Size")
        for x in sample_sizes:
            time_f.write(" " + str(x))
        time_f.write("\n")
        time_f.write("Fair time")
        for x in fair_avlk_times:
            time_f.write(" " + str(x))
        time_f.write("\n")
        time_f.write("Avlk time")
        for x in avlk_times:
            time_f.write(" "+ str(x))
        time_f.close()