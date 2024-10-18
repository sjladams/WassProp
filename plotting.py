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




def plot_single_step(dynamics, w2_bounds, tag, initial_budget_options):
    fig_w2_bounds = plt.figure()
    plt.plot(initial_budget_options, w2_bounds['gl'], label='Global Lipschitz')
    if 'type1' in w2_bounds:
        plt.plot(initial_budget_options, w2_bounds['type1'],
                 label=r'Own (Budget Term 2 = $W_2(\Delta p,\Delta q)$)')
    plt.plot(initial_budget_options, w2_bounds['type2'],
             label=r'Own (Budget Term 2 = $W_2(p,\Delta q)$)')
    plt.legend()
    plt.title(tag)
    plt.xlabel('$W_2(p,q)$')
    plt.xticks(initial_budget_options)
    plt.ylabel(r'$W_2(f p, f \Delta q)$')
    plt.show()
