from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class Parallel_Coordinates:
    def __init__(self, dataframe, ax=None, fs=10) -> None:
        self.columns = dataframe.columns[1:]
        self.data = dataframe.to_numpy()[:,1:]
        self.index = dataframe.iloc[:,0].to_list()
        self.ax = ax
        self.lim = self.data.shape[1] - 1
        self.fs = fs

    def wrap_xlabel_strings(self, max_char=9):
        strings = self.columns
        wrapped_strings = []
        for string in strings:
            if len(string) > max_char:
                wrapped_string = '\n'.join([string[i:i+max_char] for i in range(0, len(string), max_char)])
                wrapped_strings.append(wrapped_string)
            else:
                wrapped_strings.append(string)
        self.columns = wrapped_strings
    
    def transforms(self):
        self.data[self.data == 0] = 1e-5
        ys = self.data
        ymins = ys.min(axis=0)
        ymaxs = ys.max(axis=0)
        dys = ymaxs - ymins
        ymins -= dys * 0.05  # add 5% padding below and above
        ymaxs += dys * 0.05
        ymaxs[1], ymins[1] = ymins[1], ymaxs[1]  # reverse axis 1 to have less crossings

        self.ymin = ymins
        self.ymax = ymaxs

        dys = ymaxs - ymins
        # transform all data to be compatible with the main axis
        zs = np.zeros_like(ys)
        zs[:, 0] = ys[:, 0]
        zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

        self.zs = zs
    
    def create_axes(self):
        axes = [self.ax] + [self.ax.twinx() for i in range(self.lim)]

        for i, nx in enumerate(axes):
            nx.set_ylim(self.ymin[i], self.ymax[i])
            nx.spines['top'].set_visible(False)
            nx.spines['bottom'].set_visible(False)
            nx.tick_params(axis='y', labelsize=self.fs-2)
            if nx != self.ax:
                nx.spines['left'].set_visible(False)
                nx.yaxis.set_ticks_position('right')
                nx.spines["right"].set_position(("axes", i / (self.data.shape[1] - 1)))

        self.ax.set_xlim(0, self.lim)
        self.ax.set_xticks(range(self.data.shape[1]))
        self.ax.set_xticklabels(self.columns, fontsize=self.fs, rotation=0)
        self.ax.tick_params(axis='x', which='major', pad=7)
        self.ax.spines['right'].set_visible(False)
        self.ax.xaxis.tick_top()
    
    def plot_curves(self):
        colors = plt.cm.Set2.colors
        mult = np.ceil(len(self.data)/len(colors)).astype(int)
        colors = list(colors) * mult
        legend_handles = [None for _ in self.index]
        for j in range(self.data.shape[0]):
            col = colors[j]
            lw = 0.3
            ls = "-"
            if j == 45:
                lw = 2
                col = "k"
                ls = "--"
            # create bezier curves
            verts = list(zip([x for x in np.linspace(0, 
                                                     len(self.data) - 1, 
                                                     len(self.data) * 3 - 2, endpoint=True)],
                            np.repeat(self.zs[j, :], 3)[1:-1]))
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=lw, alpha=1, edgecolor=col, ls=ls)
            legend_handles[j] = patch
            self.ax.add_patch(patch)
    
    def plot(self):
        self.wrap_xlabel_strings()
        self.transforms()
        self.create_axes()
        self.plot_curves()