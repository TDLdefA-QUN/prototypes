ENV["GKSwstype"] = "nul"
using Pkg
Pkg.activate("/home/kaliam/NormalFormAE/.")
using NormalFormAE, Flux,CUDA, Zygote, Plots, DifferentialEquations, FileIO, Printf
device!(CUDA.CuDevice(1))
include("/home/kaliam/NormalFormAE/problems/nf.jl")
include("/home/kaliam/NormalFormAE/run/run_nf_old.jl")

x_model_name = :nf
z_model_name = :Hopf
x_dim = 128
par_dim = 1
z_dim = 2
tsize = 250 # previously 500 
tspan = [0.0,100.0] # previously 200
xVar = 0.1
aVar = 0.5
mean_ic_x = [0.0] 
mean_ic_a = [0.0]
x_rhs = dxdt_rhs
x_solve = dxdt_solve
machine = gpu
 

model_x = xModel(x_model_name, x_dim, par_dim, tsize,tspan,xVar,aVar,mean_ic_x,mean_ic_a,x_rhs,x_solve, args)
model_z = NormalForm(:Hopf,z_dim ,par_dim, dzdt_rhs, dzdt_solve)

state = AE(:State, 128,2, [64,32],:tanh,machine)
par = AE(:Par, 1,1,[16,16],:tanh,machine)

tscale_init = [1.4f0] |> machine

training_size = 1000
test_size = 20


# data_dir = nothing

P_reg = [0f0, 0.01f0, 0.0f0, 0.0001f0, 0.0f0, 0.001f0, 0.0f0 ]

data_dir = "/home/kaliam/NFAEdata/"
# data_dir = nothing

nfae = NFAE(:nf, :Hopf, model_x, model_z, training_size, test_size, state, par, nothing, tscale_init,
                       P_reg,machine, 20,20,0.1,data_dir)

# ---------------------------------- Training-----------------------------------------------------------
# nfae.data_dir = "home/kaliam/NFAEdata/"
load_data(nfae)
load_params(nfae)
# save_data(nfae) # Make sure to remove this after first run


# Sort test data by order of parameters
ind_ = sortperm(nfae.test_data["alpha"][1,:])
nfae.test_data["alpha"] = sort(nfae.test_data["alpha"],dims=2)
nfae.test_data["x"] = nfae.test_data["x"][:,:,ind_]
nfae.test_data["dx"] = nfae.test_data["dx"][:,:,ind_]

# Trim initial transients for Lorenz96 specifically
trim = 50
nfae.training_data["x"] = nfae.training_data["x"][:,trim+1:end,:]
nfae.training_data["dx"] = nfae.training_data["dx"][:,trim+1:end,:]
nfae.test_data["x"] = nfae.test_data["x"][:,trim+1:end,:]
nfae.test_data["dx"] = nfae.test_data["dx"][:,trim+1:end,:]
nfae.model_x.tsize = nfae.model_x.tsize - trim
nfae.model_x.tspan = [0.0, 80.0]

# Scale params
nfae.training_data["alpha"] = nfae.training_data["alpha"] 
nfae.test_data["alpha"] = nfae.test_data["alpha"]



# Load test aata
x_test = reduce(hcat,[nfae.test_data["x"][:,:,i] for i in 1:nfae.test_size]) |> nfae.machine
dx_test = reduce(hcat,[nfae.test_data["dx"][:,:,i] for i in 1:nfae.test_size]) |> nfae.machine
alpha_test = nfae.test_data["alpha"] |> nfae.machine

nEpochs = 0
batchsize = 250 # before 50
ctr = 1
p = gen_plot(nfae.model_z.z_dim, nfae.nPlots)
adamarg = 0.0001
opt = ADAM(adamarg)

# Helpers, should go into nfae at some point
loss_train_full = []
loss_test_full = []
last_loss = 1e10
loss_test = 0.0f0
rel_loss_test = 1f0

include("/home/kaliam/NormalFormAE/src/utils/pretrain.jl")
include("/home/kaliam/NormalFormAE/src/utils/train.jl")

if nfae.state.act == "id"
    act_ = nothing
else
    act_ = Symbol(nfae.state.act)
end

while ctr < nEpochs*(training_size/batchsize) 
   global adamarg
   ctr = 1 
   nfae.state = AE(:State, nfae.state.in_dim,nfae.state.out_dim,nfae.state.widths,act_,nfae.machine) # respawn
   nfae.par = AE(:Par, nfae.par.in_dim,nfae.par.out_dim,nfae.par.widths,act_,nfae.machine) # respawn
   pretrain(10000,batchsize,5e2,ADAM(0.001))
   #load_params(nfae)
   train(nfae,nEpochs,batchsize, x_test, dx_test, alpha_test)
   adamarg = 0.0001
end
