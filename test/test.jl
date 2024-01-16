using RidgeBootstrapStateEvolution

sampling_ratio = 1.0
regularisation = 1e-4
noise_variance = 1.0

RidgeBootstrapStateEvolution.state_evolution_bootstrap_bootstrap(sampling_ratio, regularisation, noise_variance)
RidgeBootstrapStateEvolution.state_evolution_bootstrap_bootstrap_full(sampling_ratio, regularisation, noise_variance; relative_tolerance=1e-4, max_iteration=100, max_weight=8)