import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    dat = pd.read_csv("hw4res.csv").to_numpy()

    plt.figure()
    plt.plot(dat[:,0], dat[:,1])
    plt.xlabel("beta")
    plt.ylabel("||x||")
    plt.title("Long-run behavior of ||x|| versus beta")
    plt.show()
    
