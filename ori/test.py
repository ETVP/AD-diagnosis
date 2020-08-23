import numpy as np


if __name__ == '__main__':
    values = [i for i in range(20)]
    values = np.array(values)
    print(np.median(values))
