import matplotlib.pyplot as plt
import numpy# as np
import numpy.ma as ma
d1 = numpy.genfromtxt("scipy_medfilt_19.csv", delimiter=',', dtype="int, f8", names=["EPIC_id", "sde"])
d2 = numpy.genfromtxt("scipy_medfilt_29.csv", delimiter=',', dtype="int, f8", names=["EPIC_id", "sde"])
d3 = numpy.genfromtxt("scipy_medfilt_39.csv", delimiter=',', dtype="int, f8", names=["EPIC_id", "sde"])
d4 = numpy.genfromtxt("scipy_medfilt_49.csv", delimiter=',', dtype="int, f8", names=["EPIC_id", "sde"])
summe = d1["sde"] + d2["sde"] + d3["sde"] + d4["sde"]
idxs = numpy.where(summe > 0)
d1 = d1["sde"][idxs]
d2 = d2["sde"][idxs]
d3 = d3["sde"][idxs]
d4 = d4["sde"][idxs]


s1 = numpy.genfromtxt("savgol_49.csv", delimiter=',', dtype="int, f8", names=["EPIC_id", "sde"])
s2 = numpy.genfromtxt("savgol_101.csv", delimiter=',', dtype="int, f8", names=["EPIC_id", "sde"])
s3 = numpy.genfromtxt("savgol_151.csv", delimiter=',', dtype="int, f8", names=["EPIC_id", "sde"])
s4 = numpy.genfromtxt("savgol_201.csv", delimiter=',', dtype="int, f8", names=["EPIC_id", "sde"])
summe = s1["sde"] + s2["sde"] + s3["sde"] + s4["sde"]
idxs = numpy.where(summe > 0)
s1 = s1["sde"][idxs]
s2 = s2["sde"][idxs]
s3 = s3["sde"][idxs]
s4 = s4["sde"][idxs]


all_data = [d1, d2, d3, d4, s1, s2, s3, s4]

a = len(d1) - numpy.count_nonzero(d1)
b = len(d2) - numpy.count_nonzero(d2)
c = len(d3) - numpy.count_nonzero(d3)
d = len(d4) - numpy.count_nonzero(d4)

e = len(s1) - numpy.count_nonzero(s1)
f = len(s2) - numpy.count_nonzero(s2)
g = len(s3) - numpy.count_nonzero(s3)
h = len(s4) - numpy.count_nonzero(s4)


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

axes.text(1, 2, str(a), ha='center')
axes.text(2, 2, str(b), ha='center')
axes.text(3, 2, str(c), ha='center')
axes.text(4, 2, str(d), ha='center')

axes.text(5, 2, str(e), ha='center')
axes.text(6, 2, str(f), ha='center')
axes.text(7, 2, str(g), ha='center')
axes.text(8, 2, str(h), ha='center')

# plot box plot
axes.boxplot(all_data, notch=True, sym='None', meanline=True, showmeans=True)
axes.plot((4.5, 4.5), (0, 40))
axes.set_ylim(0, 40)

# adding horizontal grid lines
axes.yaxis.grid(True)
axes.set_xticks([y+1 for y in range(len(all_data))])
axes.set_xlabel('xlabel')
axes.set_ylabel('ylabel')

# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
         xticklabels=['19', '29', '39', '49', '51', '101', '151', '201'])
plt.show()
