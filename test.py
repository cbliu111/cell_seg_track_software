import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("neg_pcb_bf_488.csv", header=0)
df10 = pd.read_csv("660-10-750-50.csv", header=0)
df20 = pd.read_csv("660-20-750-50.csv", header=0)
df40 = pd.read_csv("660-40-750-50.csv", header=0)
df60 = pd.read_csv("660-60-750-50.csv", header=0)
df80 = pd.read_csv("660-80-750-50.csv", header=0)


def collect_trajectories(df):
    # collect trajectories, (time, fl, fov, lv) in traj
    traj = []
    for fov in np.unique(df["Fov"]):
        for lv in np.unique(df["Label"]):
            df_fov_label = df.loc[(df["Label"] == lv) & (df["Fov"] == fov), :]
            t = df_fov_label.loc[:, "Time"] / 60000
            fl = df_fov_label.loc[:, "Mean_intensity_of_lcb_488"].to_numpy()
            if fl.size == 0 or fl[-1]-fl[0] < 100:
                continue
            traj.append((t.to_numpy(), fl, fov, lv))
    return traj


def compute_trajectory_stat(df):
    trajs = collect_trajectories(df)
    times = []
    fl_means = []
    fl_stds = []
    for time in np.unique(df["Time"]):
        time_point_fls = []
        for traj in trajs:
            ts, fls, fov, lv = traj
            for t, fl in zip(ts, fls):
                if t == time / 60000:
                    time_point_fls.append(fl)
        time_point_fls = np.array(time_point_fls)
        if time_point_fls.size == 0:
            continue
        times.append(time)
        fl_means.append(time_point_fls.mean())
        fl_stds.append(time_point_fls.std())

    times = np.array(times)
    traj_means = np.array(fl_means)
    traj_stds = np.array(fl_stds)

    return times, traj_means, traj_stds

plt.figure(figsize=(10,8))

times, traj_means, traj_stds = compute_trajectory_stat(df)
plt.plot(times, traj_means, color="blue", label="Negative", lw=2)
plt.fill_between(times, traj_means - traj_stds, traj_means + traj_stds, facecolor='blue', alpha=0.1)

times, traj_means, traj_stds = compute_trajectory_stat(df10)
plt.plot(times, traj_means, color="green", label="660-10", lw=2)
plt.fill_between(times, traj_means - traj_stds, traj_means + traj_stds, facecolor='green', alpha=0.1)

times, traj_means, traj_stds = compute_trajectory_stat(df20)
plt.plot(times, traj_means, color="orange", label="660-20", lw=2)
plt.fill_between(times, traj_means - traj_stds, traj_means + traj_stds, facecolor='orange', alpha=0.1)

times, traj_means, traj_stds = compute_trajectory_stat(df40)
plt.plot(times, traj_means, color="purple", label="660-40", lw=2)
plt.fill_between(times, traj_means - traj_stds, traj_means + traj_stds, facecolor='purple', alpha=0.1)

times, traj_means, traj_stds = compute_trajectory_stat(df60)
plt.plot(times, traj_means, color="cyan", label="660-60", lw=2)
plt.fill_between(times, traj_means - traj_stds, traj_means + traj_stds, facecolor='cyan', alpha=0.1)


times, traj_means, traj_stds = compute_trajectory_stat(df80)
plt.plot(times, traj_means, color="red", label="660-80", lw=2)
plt.fill_between(times, traj_means - traj_stds, traj_means + traj_stds, facecolor='red', alpha=0.1)

plt.legend()
plt.xlabel("Time (min)", fontsize=20)
plt.ylabel("Fluoresence (a.u.)", fontsize=20)
plt.tight_layout()
plt.show()