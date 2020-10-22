import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
def empty_attr(line):
    if line[0] == '\n':
        return True
    for x in line:
        if x == '?' or x == '':
            return True
    return False

def is_float(x):
    try:
        y = float(x)
        return True
    except:
        return False


def load_data(filename):
    out = []
    delim = ''
    print(filename)
    if filename == "adult.data":
        delim = ', '
    if filename == "bank-additional.csv" or filename == "bank-full.csv":
        delim = ';'

    if filename == "adult.data":
        f = open(filename, 'r')
        for line in f.readlines():
            y = line.split(delim)
            x = []
            if empty_attr(y) == False:
            # white = 1, non-white = 0
                if y[8] == 'White':
            # male = 1, female = 0
                #if y[9] == 'Male':
                    x.append(1)
                else:
                    x.append(0)
                for i in range(len(y)):
                    if is_float(y[i]):
                        x.append(float(y[i]))
                out.append(x)

    if filename == "bank-additional.csv" or filename == "bank-full.csv":
        delim = ";"
        with open(filename) as f:
            next(f)
            for line in f.read().splitlines():
                y = line.split(delim)
                x = []
                if empty_attr(y) == False:
                # married = 1, non-married = 0
                    #if y[2] == '"married"':
                # age < 40 is 1, >= 40 is 0
                    if float(y[0]) < 40:
                        x.append(1)
                    else:
                        x.append(0)
                    #for i in range(len(y))
                    for i in range(len(y)):
                        if is_float(y[i]):
                            x.append(float(y[i]))
                    out.append(x)

    if filename == "mushroom.data":
        f = open(filename, 'r')
        delim = ','
        for line in f.read().splitlines():
            y = line.split(delim)
            x = []
            if empty_attr(y) == False:

                if y[4] == 't':
                    x.append('1')
                else:
                    x.append('0')
                for i in range(len(y)):
                    if i!= 4:
                        x.append((y[i]))
                out.append(x)
    out = np.array(out)
    print(out.shape)
    return out

#producing dataset with multiple colors, chop into 4 categories according to age
def load_data_multi_color(filename):
    out = []
    delim = ''
    print(filename)
    if filename == "adult.data":
        delim = ', '
    if filename == "bank-additional.csv" or filename == "bank-full.csv":
        delim = ';'

    if filename == "adult.data":
        f = open(filename, 'r')
        color_nums = [0, 0, 0, 0]
        for line in f.readlines():
            y = line.split(delim)
            x = []
            # <= 26: 3, 6413; (26, 38]: 0, 9796; (38, 48]: 1, 7131; >48: 2, 6822:
            if empty_attr(y) == False:
                if int(y[0]) <= 26:
                    x.append(3)
                    color_nums[3] += 1
                elif int(y[0]) <= 38:
                    x.append(0)
                    color_nums[0] += 1
                elif int(y[0]) <= 48:
                    x.append(1)
                    color_nums[1] += 1
                else:
                    x.append(2)
                    color_nums[2] += 1
                for i in range(len(y)):
                    if is_float(y[i]):
                        x.append(float(y[i]))
                out.append(x)

    if filename == "bank-additional.csv" or filename == "bank-full.csv":
        delim = ";"
        with open(filename) as f:
            next(f)
            color_nums = [0, 0, 0, 0]
            for line in f.read().splitlines():
                y = line.split(delim)
                x = []
                #<=30: 3, 7030; (30, 38]: 0, 14845; (38, 48]: 1, 12148; > 48: 2, 11188
                if empty_attr(y) == False:
                    if int(y[0]) <= 30:
                        x.append(3)
                        color_nums[3] += 1
                    elif int(y[0]) <= 38:
                        x.append(0)
                        color_nums[0] += 1
                    elif int(y[0]) <= 48:
                        x.append(1)
                        color_nums[1] += 1
                    else:
                        x.append(2)
                        color_nums[2] += 1
                    #for i in range(len(y))
                    for i in range(len(y)):
                        if is_float(y[i]):
                            x.append(float(y[i]))
                    out.append(x)
    out = np.array(out)
    print(out.shape)

    return out, color_nums

if __name__ == "__main__":
    #change the filename into "adult.data" if want to use census, and "bank-full.csv" if want to use bank dataset.
    filename = "adult.data"
    #use function "load_data" if want to produce two color data sets
    data, color_nums = load_data_multi_color(filename)
    print(color_nums)
    #see the README file for default settings of output filenames
    np.savetxt("./adult.csv", data, delimiter=',', fmt='%s')