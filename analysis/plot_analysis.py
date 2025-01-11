import plot
import json
import os
from configs import FOLDER, NUM_LOCS_CHOICES, W_P__Q_CHOICES

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

if __name__ == '__main__':
    plot_increase_num_locs_analysis()
    #plot_wass_ball_radius_analysis()