"""
    julia SDP-min-entanglement.jl 𝐧 θ P₃ [file]

Script used in [arXiv:2210.10357](https://arxiv.org/abs/2210.10357) for minimizing
entanglement over positive partial transpose states, to find the threshold for a
dynamic-based entanglement witness. See the paper for more detail.

# Arguments
- `𝐧::Integer`: Hilbert space truncation; energy constraints on the normal modes
- `θ::Float`: Angle, in radians, of passive transformation that relates physical
  modes to normal modes
- `P₃::Vector{Float}`: Observed violation of precession protocol
- `file::String`: File to save the data. Returns to standard output if none specified

# Package versions
For the reported results, version 1.6.6 of `julia` was used, with the packages
- `Convex`: 0.15.1
- `SCS`: 1.1.2
- `Memoization`: 0.1.14
"""
𝐧  = parse(Int,ARGS[1])
θ  = parse(Float64,ARGS[2])
P₃ = parse.(Float64,split(ARGS[3],","))
file = length(ARGS) ≥ 4 ? ARGS[4] : false

# Number of threads for parallel computation
numThreads = Threads.nthreads()

using LinearAlgebra, SparseArrays, Convex, SCS, Memoization, Dates

"""
    binCoef(x)

Returns 2⁻ˣ × choose(x,⌊x/2⌋). For `x` ≤ 62, the exact value is returned.
For x > 62, a series where log(binCoef(x)) is of order O(1/x²¹) is used.
"""
function binCoef(x)
    x = 2*(x÷2)
    if x ≤ 62
        return binomial(BigInt(x),BigInt(x÷2))/2^x
    else
        # We use 2⁻ˣ × choose(x,⌊x/2⌋) = exp(f(x) + O(1/x²¹)) × √(2/(π*x))
        x = BigFloat(x)
        fx = exp(
                -1/4x + 1/24x^3 - 1/20x^5 + 17/112x^7 - 31/36x^9 +
                691/88x^11 - 5461/52x^13 + 929569/480x^15 - 3202291/68x^17 + 221930581/152x^19
            )
        return fx/√(π*x/2)
    end
end

"""
    sgnX(n₁,n₂)

Returns ⟨n₁|sgn(X)|n₂⟩; see Eq. (D11) of [arXiv:2204.10498](https://arxiv.org/abs/2204.10498)
"""
function sgnX(n₁,n₂)
    if (n₁-n₂)%2 == 0
        return 0
    else
        if n₂%2 == 0
            n₁,n₂ = n₂,n₁
        end
        return (-1.0)^((n₂-n₁-1)÷2)/(n₂-n₁)*√(n₂*binCoef(n₁)*binCoef(n₂-1)*2/π)
    end
end

"""
    BS(θ=π/4;ϕ₀=0,ϕ₁=0,𝐧=1)

Returns the passive transformation (a₊,a₋) → (a₁,a₂) in the Fock space basis,
where θ is the angle, ϕ₀ and ϕ₁ are the phases, and 𝐧 is the truncation for each
space. Refer to Eq. (6) of [arXiv:2210.10357](https://arxiv.org/abs/2210.10357) for the passive transformation.
"""
function BS(θ=π/4;ϕ₀=0,ϕ₁=0,𝐧=1)
    U = spzeros(ComplexF64,(2𝐧+1)^2,(𝐧+1)^2)
    for n ∈ 0:𝐧
        for m ∈ 0:𝐧
            for s ∈ 0:m+n
                U[(2𝐧+1)*s+m+n-s+1,(𝐧+1)*m+n+1] = sum(
                    √(
                        binomial(BigInt(s),BigInt(j))*
                        binomial(BigInt(m+n-s),BigInt(m-j))*
                        binomial(BigInt(m),BigInt(j))*
                        binomial(BigInt(n),BigInt(s-j))
                    )*
                    cos(big(θ))^(2j+n-s)*
                    sin(big(θ))^(m+s-2j)*
                    exp(-im*ϕ₀*(s-n))*
                    exp(-im*ϕ₁*(m-s))*
                    (-1)^(s-j)
                    for j ∈ max(s-n,0):min(s,m)
                ) |> ComplexF64
            end
        end
    end
    return U
end

"""
    GellMann_f(k,j,d)

Defines the generalised GellMann matrices (GMM) fₖⱼ of dimension `d`, which forms
part of the Hermitian orthonormal basis operators as defined in Eqs. (3)—(5) of
[arXiv:0806.1174](https://arxiv.org/abs/0806.1174). When `k` = `j`, the diagonal GMMs are returned;
the symmetric GMM when `k` < `j`; and the antisymmetric GMM when `k` > `j`.
"""
function GellMann_f(k,j,d)
    if k == j
        return GellMann_h(k,d)
    elseif k < j
        M = spzeros(ComplexF64,d,d)
        M[j,k] = M[k,j] = 1/√2
        return M
    elseif k > j
        M = spzeros(ComplexF64,d,d)
        M[j,k] = -im/√2
        M[k,j] =  im/√2
        return M
    end
end

"""
    GellMann_h(k,d)

Defines the diagonal generalised GellMann matrices (GMM) hₖ of dimension `d`.
Since `GellMann_h(k,d)` is defined recursively, `@memoize` is used to remember
the output of a previously-called instance
"""
@memoize function GellMann_h(k,d)
    if k==1
        return sparse(I(d)/√d)
    elseif k < d
        return [GellMann_h(k,d-1) spzeros(d-1); spzeros(d)']
    elseif k == d
        return √(1/d/(d-1))*[sparse(I(d-1)) spzeros(d-1); spzeros(d-1)' 1-d]
    end
end


"""
    pTranspose(ρ,d₁::Int=Int(√size(ρ)[1]),d₂::Int=Int(√size(ρ)[1]))

Returns the partial transpose of `ρ`, defined by ⟨n₁,n₂|ρᵀ²|m₁,m₂⟩ = ⟨n₁,m₂|ρ|m₁,n₂⟩
"""
function pTranspose(ρ,d₁::Int=Int(√size(ρ)[1]),d₂::Int=Int(√size(ρ)[1]))
    ρᵀ² = copy(ρ)*0
    for n₁ ∈ 0:d₁-1, n₂ ∈ 0:d₂-1, m₁ ∈ 0:d₁-1, m₂ ∈ 0:d₂-1
        # ⟨n₁,n₂|ρᵀ²|m₁,m₂⟩ = ⟨n₁,m₂|ρ|m₁,n₂⟩
        ρᵀ²[n₁*d₂ + n₂ + 1, m₁*d₂ + m₂ + 1] = ρ[n₁*d₂ + m₂ + 1, m₁*d₂ + n₂ + 1]
    end
    return ρᵀ²
end

# The passive transformation for the specified angle and truncation
U = BS(θ,𝐧=𝐧)

# The work is split amongst the different threads, so these
# arrays keep track of which indices each thread handles.
# For multithreading reasons, it's easier to use the linearised index
# ((𝐧+1)² × j + k) instead of the two indices (j,k)
ind₁₂ = round.(Int,range(1, (𝐧+1)^4,length=numThreads+1))
ind₊₋ = round.(Int,range(1,(2𝐧+1)^4,length=numThreads+1))

# These will be exactly the B⃗B⃗, B⃗B⃗ᵀ² and A⃗A⃗ defined in the supplementary. B⃗B⃗
# is written in the {a₊,a₋} basis, while B⃗B⃗ᵀ² and A⃗A⃗ is in the {a₁,a₂} basis.
# Also, AA₀₀ = A₀ ⊗ A₀, BB₀₀ = B₀ ⊗ B₀. Here, they are split by the thread
BB   = [spzeros(ComplexF64, (𝐧+1)^4,ind₁₂[i+1]-ind₁₂[i]) for i ∈ 1:numThreads]
BBᵀ² = [spzeros(ComplexF64,(2𝐧+1)^4,ind₁₂[i+1]-ind₁₂[i]) for i ∈ 1:numThreads]
AA   = [spzeros(ComplexF64,(2𝐧+1)^4,ind₊₋[i+1]-ind₊₋[i]) for i ∈ 1:numThreads]
Threads.@threads for i ∈ 1:numThreads
    # Get the first and last index that this thread will handle
    jk₀ = ind₁₂[i]+1
    jk₁ = ind₁₂[i+1]
    for jk ∈ jk₀:jk₁
        # Convert linearlised index back into the index pair
        j = (jk-1)÷((𝐧+1)^2)+1
        k = jk-((𝐧+1)^2)*(j-1)

        # X is Hermitian by definition, but sometimes Julia needs a bit of a reminder
        X = GellMann_f(j,k,(𝐧+1)^2)
        BB[i][:,jk-jk₀+1] = sparse(X+X')[:]/2

        # Now get its partial transpose in the {a₁,a₂} basis
        X = pTranspose(U*X*U')
        BBᵀ²[i][:,jk-jk₀+1] = sparse(X+X')[:]/2

        # For garbage collection
        X = nothing
    end
    # Analogous steps for AA
    jk₀ = ind₊₋[i]+1
    jk₁ = ind₊₋[i+1]
    for jk ∈ jk₀:jk₁
        j = (jk-1)÷((2𝐧+1)^2)+1
        k = jk-((2𝐧+1)^2)*(j-1)
        X = GellMann_f(j,k,(2𝐧+1)^2)
        AA[i][:,jk-jk₀+1] = sparse(X+X')[:]/2
        X = nothing
    end
end

# Bring the results from each thread together, define B₀ ⊗ B₀
BB = hcat(BB...)
BB₀₀ = GellMann_f(1,1,(𝐧+1)^2)
BBᵀ² = hcat(BBᵀ²...)
BBᵀ²₀₀ = (pTranspose(U*BB₀₀*U') + pTranspose(U*BB₀₀*U')')/2
AA = hcat(AA...)
AA₀₀ = GellMann_f(1,1,(2𝐧+1)^2)

# Define Q₃ − ½𝟙 in the a₊ mode
Q₃₊ = spzeros(𝐧+1,𝐧+1)
for n ∈ 0:𝐧
    for m ∈ n+3:6:𝐧
        Q₃₊[m+1,n+1] = Q₃₊[n+1,m+1] = sgnX(n,m)/2
    end
end
# Now in the {a₊,a₋} space
Q₃₊₋ = kron(Q₃₊,I(𝐧+1))
# And the vector q⃗
q⃗ = real.(BB'*(Q₃₊₋ + Q₃₊₋')[:])/2

# Define the variables needed for the SDP
z  = Variable()
x⃗  = Variable( (𝐧+1)^4-1)
y⃗₊ = Variable((2𝐧+1)^4-1)
y⃗₋ = Variable((2𝐧+1)^4-1)

# The operator ρ defined with x⃗
ρ = BB₀₀/tr(BB₀₀) + reshape(BB*x⃗,(𝐧+1)^2,(𝐧+1)^2)

# Collect the garbage
BB = BB₀₀ = nothing
GC.gc()

# The operators relating to the partial transpose of ρ
ρ₊ = z*AA₀₀/tr(AA₀₀) + reshape(AA*y⃗₊,(2𝐧+1)^2,(2𝐧+1)^2)
ρ₋ = (z-1)*AA₀₀/tr(AA₀₀) + reshape(AA*y⃗₋,(2𝐧+1)^2,(2𝐧+1)^2)
ρᵀ² = BBᵀ²₀₀/tr(BBᵀ²₀₀) + reshape(BBᵀ²*x⃗,(2𝐧+1)^2,(2𝐧+1)^2)

# Collect the garbage
AA₀₀ = AA = BBᵀ²₀₀ = BBᵀ² = nothing
GC.gc()

# Collect the garbage
Memoization.empty_cache!(GellMann_h)
GC.gc()

# Perform the SDP
outputStrings = ["" for _ in 1:length(P₃)]
startTime = now()
println("Start time: $startTime")
for iP in 1:length(P₃)
    println("Starting $(P₃[iP]), $iP of $(length(P₃))")
    constraints = [
        ρ  ⪰ 0,
        ρ₊ ⪰ 0,
        ρ₋ ⪰ 0,
        ρ₊-ρ₋ == ρᵀ²,
        P₃[iP] == x⃗⋅q⃗ + 1/2
    ]
    optimizer = SCS.Optimizer()
    Convex.MOI.set(optimizer, Convex.MOI.RawOptimizerAttribute("max_iters"), 1_000_000)
    problem = minimize(2z-1,constraints)
    solve!(problem, optimizer)

    # Evaluate tr(ρᵀ²), making sure to normalise it.
    tr_ρᵀ² = (evaluate(ρᵀ²)/tr(evaluate(ρᵀ²))) |> collect |> eigvals .|> abs |> sum
    outputStrings[iP] = "$𝐧,$θ,$(P₃[iP]),$tr_ρᵀ²"

    # If it was specified, append the result into the file.
    if file != false
        open(file,"a") do io
            println(io,outputStrings[iP])
        end
    end
end
println("End time: $(now()), time taken: $(now()-startTime)")
println("")

# If no file was specified, print out the results
if file == false
    println(join(outputStrings,"\n"))
end
