import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from qutip import *
from base64 import b64encode


# The aim of this exercise is to "squeeze" an observable so that its variance
# is less than the minimum uncertainty limit set by the Heisenberg uncertainity principle
# between two non-commuting operators. We will use the \hat{x} and \hat{p} of a quantum
# oscillator for this purpose

# The Quantum Harmonic Oscillator
# Parameters
N = 35
w = 1 * 2 * np.pi              # oscillator frequency
tlist = np.linspace(0, 4, 101) # periods


# operators
a = destroy(N)
n = num(N)
x = (a + a.dag())/np.sqrt(2)
p = -1j * (a - a.dag())/np.sqrt(2)

# the quantum harmonic oscillator Hamiltonian
H = w * a.dag() * a

c_ops = []

# uncomment to see how things change when dissipation is included
# c_ops = [np.sqrt(0.25) * a]

# A function to plot expectation value with variance

def plot_expect_with_variance(N, op_list, op_title, states):
    """
    Plot the expectation value of an operator (list of operators)
    with an envelope that describes the operators variance.
    """

    fig, axes = plt.subplots(1, len(op_list), figsize=(14, 3))

    for idx, op in enumerate(op_list):
        e_op = expect(op, states)
        v_op = variance(op, states)

        axes[idx].fill_between(tlist, e_op - np.sqrt(v_op), e_op + np.sqrt(v_op), color="green", alpha=0.5);
        axes[idx].plot(tlist, e_op, label="expectation")
        axes[idx].set_xlabel('Time')
        axes[idx].set_title(op_title[idx])

    return fig, axes

# Let's start with the "coherent" state, where the Quantum Harmonic oscillator most
# resembles the classical harmonic oscillator.

# A coherent state is formally defined as the (unique) eigenstate of the annihilation operator â with
# corresponding eigenvalue α.

# "coherent" function of qutip generates a coherent state with eigenvalue alpha.
# Constructed using displacement operator on vacuum state.
# input N: Number of Fock states in Hilbert space, Alpha: Eigenvalue of the coherent state
# output:  Qobj quantum object for coherent state

psi0 = coherent(N, 2.0)

# Mesolve:  Master equation evolution of a density matrix for a given Hamiltonian and set of collapse operators,
# or a Liouvillian. This evolves the state vector or density matrix (rho0) using a given Hamiltonian or Liouvillian (
# H) and an optional set of collapse operators (c_ops), by integrating the set of ordinary differential equations
# that define the system. In the absence of collapse operators the system is evolved according to the unitary
# evolution of the Hamiltonian.

result = mesolve(H, psi0, tlist, c_ops, [])

plot_expect_with_variance(N, [n, x, p], [r'$n$', r'$x$', r'$p$'], result.states);


fig, axes = plt.subplots(1, 2, figsize=(10,5))

plt.show()



# Since the uncertainty (and hence measurement noise) stays constant at 1⁄2 as the amplitude of the oscillation
# increases, the state behaves increasingly like a sinusoidal wave. Moreover, since the vacuum
# state | 0 ⟩ is just the coherent state with α=0, all coherent states have the
# same uncertainty as the vacuum. Therefore, one may interpret the quantum noise of a coherent state as being due to
# vacuum fluctuations.

#Squeezed vacuum

#General squeezing operator with parameter = 1.0 S(z)=exp(1/2(z^* a_1 a_2 - za_1\dagger a2_\dagger)
psi0 = squeeze(N, 1.0) * basis(N, 0)
result = mesolve(H, psi0, tlist, c_ops, [])
plot_expect_with_variance(N, [n, x, p], [r'$n$', r'$x$', r'$p$'], result.states);
plt.show()

#Let's see the effect of squeezed vacuum on a coherent state
#Squeezed coherent state
#(First squeeze vacuum and then
psi0 = displace(N, 2) * squeeze(N, 1.0) * basis(N, 0)  # first squeeze vacuum and then displace
result = mesolve(H, psi0, tlist, c_ops, [])
plot_expect_with_variance(N, [n, x, p], [r'$n$', r'$x$', r'$p$'], result.states);
plt.show()





