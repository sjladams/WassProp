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




def plot_single_step(dynamics, w2_bounds, tag, w2_p__q_options):
    fig_w2_bounds = plt.figure()
    plt.plot(w2_p__q_options, w2_bounds['gl'], label='Global Lipschitz')
    plt.plot(w2_p__q_options, w2_bounds['independent_coupling'], label='Independent Coupling')
    plt.plot(w2_p__q_options, w2_bounds['local_linear'], label='Local Linearization')
    plt.plot(w2_p__q_options, w2_bounds['together'], label='Together')
    if 'triangle_type1' in w2_bounds:
        plt.plot(w2_p__q_options, w2_bounds['triangle_type1'], label=r'Triangle (Budget Term 2 = $W_2(\Delta p,\Delta q)$)')
    plt.plot(w2_p__q_options, w2_bounds['triangle_type2'], label=r'Triangle (Budget Term 2 = $W_2(p,\Delta q)$)')

    plt.legend()
    plt.title(tag)
    plt.xlabel('$W_2(p,q)$')
    plt.xticks(w2_p__q_options)
    plt.ylabel(r'$W_2(f p, f \Delta q)$')
    plt.xlim(min(w2_p__q_options), max(w2_p__q_options))
    plt.show()
