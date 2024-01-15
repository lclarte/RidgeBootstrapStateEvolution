module RidgeBootstrapStateEvolution

export state_evolution

using Distributions
using ForwardDiff
using Integrals
using LinearAlgebra
using Optim
using QuadGK
using SpecialFunctions
using StaticArrays

# 5.0 seems good enough for now :) and is twice faster than Inf 
const Bound = 7.5
const LogisticProbitFactor = 0.5875651988237005

function weights_proba_function_bootstrap(w1::Number, w2::Number)
    return pdf(Poisson(1), w1) * pdf(Poisson(1), w2)
end

## 

function update_hatoverlaps(m::AbstractVector, q::AbstractMatrix, v::AbstractVector, vstar::Number, noise_variance::Number, weight_range1, weight_range2, weight_distribution::Function)
    mhat = MVector{2}([0.0, 0.0])
    qhat = MMatrix{2, 2}(zeros((2, 2)))
    vhat = MVector{2}([0.0, 0.0])

    q_inv = inv(q)

    # bias_mat    = np.vstack([m_vec.reshape((1, 2)), m_vec.reshape((1, 2))]) @ inv_q_mat - I
    bias_mat = vcat(m', m') * q_inv - I
    
    for weight1 in weight_range1
        for weight2 in weight_range2
            gout_vec      = SVector{2}([ weight1 / (1.0 + weight1 * v[1]), weight2 / (1.0 + weight2 * v[2])])
            gout_mat      = SMatrix{2, 2}(diagm(gout_vec))
            proba::Number = weight_distribution(weight1, weight2)

            mhat += gout_vec * proba
            qhat += (gout_mat * ((vstar + noise_variance) .* SMatrix{2, 2}(ones((2, 2))) + bias_mat * q * bias_mat') * gout_mat') * proba
            vhat += gout_vec * proba
        end
    end

    return mhat, qhat, vhat
end

function update_hatoverlaps_y_resampling(m::AbstractVector, q::AbstractMatrix, v::AbstractVector, vstar::Number, noise_variance::Number)
    mhat = MVector{2}([0.0, 0.0])
    qhat = MMatrix{2, 2}(zeros((2, 2)))
    vhat = MVector{2}([0.0, 0.0])

    q_inv = inv(q)

    # bias_mat    = np.vstack([m_vec.reshape((1, 2)), m_vec.reshape((1, 2))]) @ inv_q_mat - I
    bias_mat = vcat(m', m') * q_inv - I
    
    gout_vec      = SVector{2}([ 1.0 / (1.0 + v[1]), 1.0 / (1.0 + v[2])])
    gout_mat      = SMatrix{2, 2}(diagm(gout_vec))

    mhat += gout_vec
    qhat += (gout_mat * (vstar .* SMatrix{2, 2}(ones((2, 2))) + noise_variance .* I + bias_mat * q * bias_mat') * gout_mat)
    vhat += gout_vec

    return mhat, qhat, vhat
end

function update_overlaps(mhat::AbstractVector, qhat::AbstractMatrix, vhat::AbstractVector, regularisation::Number)
    mat = SMatrix{2, 2}(diagm(1.0 ./ (regularisation .+ vhat)))
    
    m = mhat ./ (regularisation .+ vhat)
    q = mat * (mhat * mhat' + qhat) * mat
    v = 1.0 ./ (regularisation .+ vhat)

    return m, q, v

end

function state_evolution_bootstrap_bootstrap(sampling_ratio, regularisation, noise_variance; max_weight=8, relative_tolerance=1e-4, max_iteration=100)
    rho = 1.0
    
    old_m = SVector(0.0, 0.0);
    m     = SVector(0.0, 0.0);
    q     = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
    v     = SVector{2}([1.0 1.0]);

    mhat  = SVector(0.0, 0.0);
    qhat  = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
    vhat  = SVector{2}([1.0 1.0]);

    for i in 0:max_iteration
        # copy m into old_m to compute the difference at the end of the loop
        old_m = copy(m)

        vstar = rho - m' * inv(q) * m
        tmp = update_hatoverlaps(m, q, v, vstar, noise_variance, 0:max_weight, 0:max_weight, weights_proba_function_bootstrap)
        mhat = sampling_ratio * tmp[1]
        qhat = sampling_ratio * tmp[2]
        vhat = sampling_ratio * tmp[3]

        m, q, v = update_overlaps(mhat, qhat, vhat, regularisation)

        # compute the relative difference between old and new m 
        difference = norm(m - old_m) / norm(m)
        if difference < relative_tolerance
            return Dict([
                "m" => m, 
                "q" => q,
                "v" => v,
                "mhat" => mhat,
                "qhat" => qhat,
                "vhat" => vhat
            ])
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return Dict([
        "m" => m, 
        "q" => q,
        "v" => v,
        "mhat" => mhat,
        "qhat" => qhat,
        "vhat" => vhat
    ])
end

function state_evolution_bootstrap_full(sampling_ratio, regularisation, noise_variance, max_weight=8; relative_tolerance=1e-4, max_iteration=100)
    rho = 1.0
    
    old_m = SVector(0.0, 0.0);
    m     = SVector(0.0, 0.0);
    q     = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
    v     = SVector{2}([1.0 1.0]);

    mhat  = SVector(0.0, 0.0);
    qhat  = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
    vhat  = SVector{2}([1.0 1.0]);

    weight_function = (w1, w2) -> pdf(Poisson(1), w1) * (w2 == 1.0)

    for i in 0:max_iteration
        # copy m into old_m to compute the difference at the end of the loop
        old_m = copy(m)

        vstar = rho - m' * inv(q) * m
        tmp = update_hatoverlaps(m, q, v, vstar, noise_variance, 0:max_weight, [1], weight_function)
        mhat = sampling_ratio * tmp[1]
        qhat = sampling_ratio * tmp[2]
        vhat = sampling_ratio * tmp[3]

        m, q, v = update_overlaps(mhat, qhat, vhat, regularisation)

        # compute the relative difference between old and new m 
        difference = norm(m - old_m) / norm(m)
        if difference < relative_tolerance
            return Dict([
                "m" => m, 
                "q" => q,
                "v" => v,
                "mhat" => mhat,
                "qhat" => qhat,
                "vhat" => vhat
            ])
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return Dict([
        "m" => m, 
        "q" => q,
        "v" => v,
        "mhat" => mhat,
        "qhat" => qhat,
        "vhat" => vhat
    ])
end


function state_evolution_full_full(sampling_ratio, regularisation, noise_variance; relative_tolerance=1e-4, max_iteration=100)
    rho = 1.0
    
    old_m = SVector(0.0, 0.0);
    m     = SVector(0.0, 0.0);
    q     = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
    v     = SVector{2}([1.0 1.0]);

    mhat  = SVector(0.0, 0.0);
    qhat  = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
    vhat  = SVector{2}([1.0 1.0]);

    weight_function = (w1, w2) -> 0.5 * ((w1 == 1.0 && w2 == 0.0) || (w1 == 0.0 && w2 == 1.0))

    for i in 0:max_iteration
        # copy m into old_m to compute the difference at the end of the loop
        old_m = copy(m)

        vstar = rho - m' * inv(q) * m
        tmp = update_hatoverlaps(m, q, v, vstar, noise_variance, 0:1, 0:1, weight_function)
        mhat = 2*sampling_ratio * tmp[1]
        qhat = 2*sampling_ratio * tmp[2]
        vhat = 2*sampling_ratio * tmp[3]

        m, q, v = update_overlaps(mhat, qhat, vhat, regularisation)

        # compute the relative difference between old and new m 
        difference = norm(m - old_m) / norm(m)
        if difference < relative_tolerance
            return Dict([
                "m" => m, 
                "q" => q,
                "v" => v,
                "mhat" => mhat,
                "qhat" => qhat,
                "vhat" => vhat
            ])
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return Dict([
        "m" => m, 
        "q" => q,
        "v" => v,
        "mhat" => mhat,
        "qhat" => qhat,
        "vhat" => vhat
    ])
end

function state_evolution_y_resampling(sampling_ratio, regularisation, noise_variance; relative_tolerance=1e-4, max_iteration=100)
    rho = 1.0
    
    old_m = SVector(0.0, 0.0);
    m     = SVector(0.0, 0.0);
    q     = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
    v     = SVector{2}([1.0 1.0]);

    mhat  = SVector(0.0, 0.0);
    qhat  = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
    vhat  = SVector{2}([1.0 1.0]);

    for i in 0:max_iteration
        # copy m into old_m to compute the difference at the end of the loop
        old_m = copy(m)

        vstar = rho - m' * inv(q) * m
        tmp = update_hatoverlaps_y_resampling(m, q, v, vstar, noise_variance)
        mhat = sampling_ratio * tmp[1]
        qhat = sampling_ratio * tmp[2]
        vhat = sampling_ratio * tmp[3]

        m, q, v = update_overlaps(mhat, qhat, vhat, regularisation)

        # compute the relative difference between old and new m 
        difference = norm(m - old_m) / norm(m)
        if difference < relative_tolerance
            return Dict([
                "m" => m, 
                "q" => q,
                "v" => v,
                "mhat" => mhat,
                "qhat" => qhat,
                "vhat" => vhat
            ])
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return Dict([
        "m" => m, 
        "q" => q,
        "v" => v,
        "mhat" => mhat,
        "qhat" => qhat,
        "vhat" => vhat
    ])
end

end
