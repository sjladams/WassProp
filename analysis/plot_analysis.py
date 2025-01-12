import plot
import json
import os
from configs import FOLDER, NUM_LOCS_CHOICES, W_P__Q_CHOICES, W_P__Q_CHOICES_OPTIMIZE_LOCS

folder = FOLDER

def plot_increase_num_locs_analysis():

    file_path = os.path.join(folder, f"increase_num_locs_analysis_0.0.json")
    with open(file_path, 'r') as file:
        dict_1 = json.load(file)

    file_path = os.path.join(folder, f"increase_num_locs_analysis_0.1.json")
    with open(file_path, 'r') as file:
        dict_2 = json.load(file)

    plot.plot_analysis(
        "increase_num_locs_analysis.pdf",
        dict_1,
        dict_2,
        NUM_LOCS_CHOICES,
        log_scale = True
    )

def plot_wass_ball_radius_analysis():

    file_path = os.path.join(folder, f"wass_ball_radius_analysis_lagrangian.json")
    with open(file_path, 'r') as file:
        dict_1 = json.load(file)

    file_path = os.path.join(folder, f"wass_ball_radius_analysis_difference.json")
    with open(file_path, 'r') as file:
        dict_2 = json.load(file)


    plot.plot_analysis(
        "wass_ball_radius_analysis.pdf",
        dict_1,
        dict_2,
        W_P__Q_CHOICES,
        x_label=r'$\rho$-Wasserstein ball radius $\theta$',
        log_scale=False,
        different_y_axis=True
    )

def optimize_locs_analysis():

    file_path = os.path.join(folder, f"optimize_locs_lipschitz.json")
    with open(file_path, 'r') as file:
        dict_lipschitz = json.load(file)

    file_path = os.path.join(folder, f"optimize_locs_duality_false.json")
    with open(file_path, 'r') as file:
        dict_duality = json.load(file)

    file_path = os.path.join(folder, f"optimize_locs_duality_true.json")
    with open(file_path, 'r') as file:
        dict_duality_optimize = json.load(file)

    file_path = os.path.join(folder, f"optimize_locs_random.json")
    with open(file_path, 'r') as file:
        dict_random = json.load(file)

        plot.plot_optimize_locs(
            "optimize_locs_analysis.pdf",
            dict_lipschitz,
            dict_duality,
            dict_duality_optimize,
            dict_random,
            W_P__Q_CHOICES_OPTIMIZE_LOCS,
            dict_lipschitz.keys(),
            x_label=r"$\theta$",
            y_label=r"$\sup_{ \mathbb{Q} \in \mathbb{B}_{\theta}(\mathbb{P})}  \mathbb{W}_{\rho}(f\#\mathbb{Q}, f\#\Delta_{\mathcal{R}, \mathcal{C}}\#\mathbb{P})$",
            log_scale=False
        )
if __name__ == '__main__':
    #plot_increase_num_locs_analysis()
    #plot_wass_ball_radius_analysis()
    optimize_locs_analysis()