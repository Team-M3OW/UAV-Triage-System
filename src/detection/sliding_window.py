import numpy as np
import matplotlib.pyplot as plt

THRESH_PATH_NOMOVE = 10.0
WINDOW_SIZE = 15
STEP_SIZE = 5
THRESH_MAX_DR_TWITCH = 10.0
THRESH_MEAN_DR_WALK = 2.5
EPSILON = 1e-6


def _find_longest_contig_segment(bool_arr):
    max_len = 0
    best_start = -1
    current_len = 0
    current_start = -1
    for i, val in enumerate(bool_arr):
        if val:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                best_start = current_start
            current_len = 0
    if current_len > max_len:
        max_len = current_len
        best_start = current_start

    if best_start == -1:
        return 0, 0
    return best_start, best_start + max_len


def compute_motion_metrics(trajs_np, visibs_np, framerate=30):
    N, T, _ = trajs_np.shape
    analysis_results = []

    for n in range(N):
        vis_arr = visibs_np[n]
        start, end = _find_longest_contig_segment(vis_arr)

        if (end - start) < WINDOW_SIZE:
            analysis_results.append({'id': n, 'label': 'Invalid', 'global_path_len': np.nan})
            continue

        traj_vis = trajs_np[n, start:end]
        global_path_len = np.sum(np.linalg.norm(np.diff(traj_vis, axis=0), axis=1))

        if global_path_len < THRESH_PATH_NOMOVE:
            analysis_results.append({
                'id': n, 'label': 'No Movement', 'global_path_len': global_path_len,
                'mean_dr': 1.0, 'max_dr': 1.0, 'std_dr': 0.0,
            })
            continue

        local_directness_ratios = []
        num_windows = (len(traj_vis) - WINDOW_SIZE) // STEP_SIZE + 1

        for i in range(num_windows):
            win_start = i * STEP_SIZE
            win_end = win_start + WINDOW_SIZE
            window_traj = traj_vis[win_start:win_end]

            if len(window_traj) < 2:
                continue

            window_path_len = np.sum(np.linalg.norm(np.diff(window_traj, axis=0), axis=1))
            window_net_disp = np.linalg.norm(window_traj[-1] - window_traj[0])

            if window_path_len < EPSILON:
                local_directness_ratios.append(1.0)
            else:
                local_directness_ratios.append(window_path_len / (window_net_disp + EPSILON))

        if not local_directness_ratios:
            analysis_results.append({'id': n, 'label': 'Invalid', 'global_path_len': global_path_len})
            continue

        mean_dr = np.mean(local_directness_ratios)
        max_dr = np.max(local_directness_ratios)
        std_dr = np.std(local_directness_ratios)

        if max_dr > THRESH_MAX_DR_TWITCH and mean_dr < (max_dr / 2.0):
            label = 'Twitching'
        elif mean_dr < THRESH_MEAN_DR_WALK:
            label = 'Walking'
        else:
            label = 'General Movement'

        analysis_results.append({
            'id': n, 'label': label, 'global_path_len': global_path_len,
            'mean_dr': mean_dr, 'max_dr': max_dr, 'std_dr': std_dr,
        })

    return analysis_results


def plot_motion_analysis(results, save_path="motion_analysis.png"):
    if not results:
        print("No results to plot.")
        return

    valid_results = [r for r in results if r['label'] != 'Invalid']
    if not valid_results:
        print("No valid points to plot.")
        return

    ids = [r['id'] for r in valid_results]
    labels = np.array([r['label'] for r in valid_results])
    mean_drs = np.array([r['mean_dr'] for r in valid_results])
    max_drs = np.array([r['max_dr'] for r in valid_results])

    color_map = {
        'No Movement': 'blue',
        'Walking': 'green',
        'Twitching': 'red',
        'General Movement': 'purple',
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Sliding Window Motion Analysis", fontsize=16)

    ax1 = axes[0]
    for label, color in color_map.items():
        mask = labels == label
        if np.any(mask):
            ax1.scatter(mean_drs[mask], max_drs[mask], c=color, label=label, s=50, alpha=0.7, edgecolors='black')

    ax1.set_title("Per-Point Classification (Mean vs. Max Directness Ratio)")
    ax1.set_xlabel("Mean Directness Ratio (across all windows)")
    ax1.set_ylabel("Max Directness Ratio (in any single window)")
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.axhline(y=THRESH_MAX_DR_TWITCH, color='red', linestyle='--')
    ax1.axvline(x=THRESH_MEAN_DR_WALK, color='green', linestyle='--')
    ax1.legend()

    ax2 = axes[1]
    unique_labels = sorted(list(color_map.keys()))
    bar_width = 0.8
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if np.any(mask):
            ax2.bar(np.array(ids)[mask], 1, color=color_map[label], label=label, width=bar_width)

    ax2.set_title("Final Classification per Keypoint ID")
    ax2.set_xlabel("Keypoint ID")
    ax2.set_yticks([])
    ax2.set_ylabel("Classification")
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Motion analysis plot at {save_path}")
