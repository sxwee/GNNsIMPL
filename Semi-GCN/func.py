import matplotlib.pyplot as plt

def drawPlot(heights,fname,ylabel,legends=None):
    """
    fname：save file name
    marker：shape of dot
    """
    plt.figure()
    x = [i for i in range(1,len(heights[0]) + 1)]
    # 绘制训练集和测试集上的loss变化曲线子图
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    for i in range(len(heights)):
        plt.plot(x,heights[i])
    if legends:
        plt.legend(legends)
    plt.savefig("images/{}".format(fname))
    plt.show()

if __name__ == "__main__":
    pass