import matplotlib.pyplot as plt


def plot_multi_step(trajectories, num_time_steps, num_samples):

    fig, ax = plt.subplots()

    colors = ['Blues', 'BuPu', 'PuRd', 'Greens', 'Oranges', 'Reds', 'Greys', 'Purples',
              'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
              'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    trajectories_np = trajectories.detach().numpy()

    for t in range(num_time_steps):
        samples = trajectories_np[t * num_samples : (t+1) * num_samples]

        # Plot using hist2d with color intensity indicating the density
        plt.hist2d([state[0] for state in samples], [state[1] for state in samples], bins=100, cmap=colors[t], alpha=0.8,
                   cmin=0.1)

    # cb = plt.colorbar(label='Point Density')
    #plt.legend(loc='lower right')

    plt.xlabel('State[0]')
    plt.ylabel('State[1]')
    plt.grid(True)
    plt.axis('equal')
    plt.show()



def plot_single_step(dynamics, w2_bounds, tag, w2_p__q_options):
    fig_w2_bounds = plt.figure()
    plt.plot(w2_p__q_options, w2_bounds['gl'], label='Global Lipschitz')
    if 'independent_coupling' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['independent_coupling'], label='Independent Coupling')
    if 'local_linear' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['local_linear'], label='Local Linear')
    if 'local_constant' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['local_constant'], label='Local Constant')
    if 'local_linear_or_constant' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['local_linear_or_constant'], label='Local Linear or Constant')
    if 'local_affine' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['local_affine'], label='Local Affine')
    if 'together' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['together'], label='Together')
    if 'triangle_type1' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['triangle_type1'], label=r'Triangle (Budget Term 2 = $W_2(\Delta p,\Delta q)$)')
    if 'triangle_type2' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['triangle_type2'], label=r'Triangle (Budget Term 2 = $W_2(p,\Delta q)$)')

    plt.legend()
    plt.title(tag)
    plt.xlabel('$W_2(p,q)$')
    plt.xticks(w2_p__q_options)
    plt.ylabel(r'$W_2(f p, f \Delta q)$')
    plt.xlim(min(w2_p__q_options), max(w2_p__q_options))
    plt.show()
