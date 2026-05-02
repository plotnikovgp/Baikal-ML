import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
hfile_r = h5.File("/home/plotnikovgp/baikal/data/baikal_reco_reco_h8s2_norm.h5", "r")
hfile_m = h5.File(
    "/home/plotnikovgp/baikal/data/baikal_multi_0324_flat_treck-cascade_h8_s2_pureMC.h5", "r"
)
hfile_n = h5.File(
    "/home/plotnikovgp/baikal/data/baikal_mc2020_multi_split_MChits-8-2_track-caskade-nue2_normed.h5",
    "r",
)

# Extract theta and phi from all datasets
theta_r = hfile_r["test/prime_prty/data"][:, 0]
phi_r = hfile_r["test/prime_prty/data"][:, 1]
theta_m = hfile_m["test/prime_prty/data"][:, 0]
phi_m = hfile_m["test/prime_prty/data"][:, 1]
theta_n = hfile_n["test/prime_prty/data"][:, 0]
phi_n = hfile_n["test/prime_prty/data"][:, 1]

# Create figures for theta
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

# Plot theta histograms separately
axes1[0].hist(theta_r, bins=50, color="blue")
axes1[0].set_xlabel("Theta")
axes1[0].set_ylabel("Count")
axes1[0].set_title("Theta Distribution (reco)")

axes1[1].hist(theta_m, bins=50, color="red")
axes1[1].set_xlabel("Theta")
axes1[1].set_ylabel("Count")
axes1[1].set_title("Theta Distribution (mc2020)")

axes1[2].hist(theta_n, bins=50, color="green")
axes1[2].set_xlabel("Theta")
axes1[2].set_ylabel("Count")
axes1[2].set_title("Theta Distribution (mc2020 neutrino)")

plt.tight_layout()
plt.savefig("theta_comparison.png", dpi=300)

# Create a new figure for phi
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# Plot phi histograms separately
axes2[0].hist(phi_r, bins=50, color="blue")
axes2[0].set_xlabel("Phi")
axes2[0].set_ylabel("Count")
axes2[0].set_title("Phi Distribution (reco)")

axes2[1].hist(phi_m, bins=50, color="red")
axes2[1].set_xlabel("Phi")
axes2[1].set_ylabel("Count")
axes2[1].set_title("Phi Distribution (mc2020)")

axes2[2].hist(phi_n, bins=50, color="green")
axes2[2].set_xlabel("Phi")
axes2[2].set_ylabel("Count")
axes2[2].set_title("Phi Distribution (mc2020 neutrino)")

plt.tight_layout()
plt.savefig("phi_comparison.png", dpi=300)

print("Plots saved as theta_comparison.png and phi_comparison.png")

# Calculate and print statistics
print("Theta statistics:")
print(
    f"reco: mean={np.mean(theta_r):.4f}, std={np.std(theta_r):.4f}, min={np.min(theta_r):.4f}, max={np.max(theta_r):.4f}"
)
print(
    f"mc2020: mean={np.mean(theta_m):.4f}, std={np.std(theta_m):.4f}, min={np.min(theta_m):.4f}, max={np.max(theta_m):.4f}"
)
print(
    f"mc2020 neutrino: mean={np.mean(theta_n):.4f}, std={np.std(theta_n):.4f}, min={np.min(theta_n):.4f}, max={np.max(theta_n):.4f}"
)

print("\nPhi statistics:")
print(
    f"reco: mean={np.mean(phi_r):.4f}, std={np.std(phi_r):.4f}, min={np.min(phi_r):.4f}, max={np.max(phi_r):.4f}"
)
print(
    f"mc2020: mean={np.mean(phi_m):.4f}, std={np.std(phi_m):.4f}, min={np.min(phi_m):.4f}, max={np.max(phi_m):.4f}"
)
print(
    f"mc2020 neutrino: mean={np.mean(phi_n):.4f}, std={np.std(phi_n):.4f}, min={np.min(phi_n):.4f}, max={np.max(phi_n):.4f}"
)

# Close the files
hfile_r.close()
hfile_m.close()
hfile_n.close()
