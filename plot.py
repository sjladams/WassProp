import matplotlib.pyplot as plt


def plot_multi_step(dynamics, w2_bounds, tag):
    K = w2_bounds['gl'].shape[0] - 1

    fig_w2_bounds = plt.figure()
    plt.plot(range(K+1), w2_bounds['emp'], label='Empirical')
    plt.plot(range(K+1), w2_bounds['gl'], label='Global Lipschitz')
    if 'type1' in w2_bounds:
        plt.plot(range(K+1), w2_bounds['type1'],
                 label=r'Own (Budget Term 2 = $W_2(\Delta p,\Delta q)$)')
    plt.plot(range(K+1), w2_bounds['type2'],
             label=r'Own (Budget Term 2 = $W_2(p,\Delta q)$)')
    plt.legend()
    plt.title(tag)
    plt.xticks(range(K))
    plt.xlabel('k')
    plt.ylabel(r'$W_2(p_k, q_k)$')

    if dynamics.__class__.__name__ == 'ChaoticDynamics':
        plt.yscale('log')
        plt.xlim(1, K)
    else:
        plt.xlim(0, K)

    plt.show()

    if dynamics.__class__.__name__ in ['ChaoticDynamics', 'LinearDynamics']:
        fig_our_w2_bounds = plt.figure()
        if 'type1' in w2_bounds:
            plt.plot(range(K+1), w2_bounds['type1'],
                     label=r'Own (Budget Term 2 = $W_2(\Delta p,\Delta q)$)')
        plt.plot(range(K+1), w2_bounds['type2'],
                 label=r'Own (Budget Term 2 = $W_2(p,\Delta q)$)')
        plt.legend()
        plt.title(tag)
        plt.xticks(range(K+1))
        plt.xlabel('k')
        plt.ylabel(r'$W_2(p_k, q_k)$')

        plt.xlim(0, K)

        plt.show()

@torch.no_grad()
def plot_single_step(dynamics, w2_bounds, tag, w2_p__q_options, num_locs):

    tag = f"{dynamics.__class__.__name__} (Lipschitz={dynamics.global_lipschitz:.2f}, |C|={num_locs})"

    fig_w2_bounds = plt.figure()
    plt.plot(w2_p__q_options, w2_bounds['gl'], label='Global Lipschitz')
    if 'independent_coupling' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['independent_coupling'], label='Independent Coupling')
    if 'local_linear' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['local_linear'], label='Local Linear')
    if 'local_constant' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['local_constant'], label='Local Constant')
    if 'lagrangian_duality' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['lagrangian_duality'], label='Lagrangian Duality')
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
