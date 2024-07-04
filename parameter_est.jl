
#=
We want to use Julia to estimate the parameters Tau and D as
described in the paper: arxiv:2302.04508 (Carrara 2024). I use
Julia because they use the package: DynamicalSystems.jl


First we read the EEG data from the H5 file (we generated the H5
file from Python). Important to note that dimensions got swapped,
in Julia it's (161, 64, 4770) in Python it's (4770, 64, 161). This
is why I do permutedims(X, (3,2,1)).


Questions:
    - is the permutation of dims correct?


This code takes ~6.5hours to run. The output (17:29 July 4th 2024) is

Optimal τ: 1652.7023060796646
Optimal ψ: 287.6670859538784
=#

using HDF5
file_path = "eeg_data.h5"
X = h5read(file_path, "X")
X = permutedims(X, (3,2,1))
println("Shape of the loaded matrix: ", size(X))

using DelayEmbeddings


function MDOP_for_epochs(X)
    # Initialize τ and ψ
    τ = 0
    ψ = 0
    n_epochs = size(X, 1)
    println("N Epochs: ", n_epochs)

    # Iterate over each epoch
    for i in 1:n_epochs
        X_i = X[i,:,:]

        for channel in 1:size(X_i, 1)
            channel_data = X_i[channel, :]
            _, τ_vals, _, _, _ = mdop_embedding(channel_data)
            τ_i = τ_vals[end]
            # Update τ and ψ
            τ += τ_i
            ψ += length(τ_vals)
        end
    end

    # Calculate average τ and ψ
    τ_epoch = τ / n_epochs
    ψ_epoch = ψ / n_epochs

    return τ_epoch, ψ_epoch
end

τ_epoch, ψ_epoch = MDOP_for_epochs(X)

println("Optimal τ: ", τ_epoch)
println("Optimal ψ: ", ψ_epoch)

