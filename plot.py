import argparse
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def recursive_plot(folder_path):
    for f in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, f)):
            recursive_plot(os.path.join(folder_path,f))
        else:
            if "_cfmtx.txt" in f:
                # Visualize confusion matrix
                print("Checking", f)
                cfmtx = np.loadtxt(os.path.join(folder_path, f), delimiter=",")
                sns.heatmap(data=cfmtx, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
                figname = f.split(".")[0]
                plt.title(f"{figname}, Acc = {np.mean(np.diag(cfmtx)):>.3f}")
                plt.savefig(folder_path + "/" + figname + ".png")
                plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="models")
    args = parser.parse_args()

    recursive_plot(args.folder)
