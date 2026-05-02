# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

file_new2 = h5py.File("/home/plotnikovgp/baikal/data/baikal_mc_merged_m50-na15-nue15_norm.h5", "r")

print(file_new2.keys())

# %%
file_new2["test"].keys()
# %%
ev_ids = np.array(file_new2["test/ev_ids/data"])

ev_ids_sample = ev_ids[::100]
print(ev_ids_sample.shape)

plt.hist(ev_ids_sample, bins=100)
plt.show()
