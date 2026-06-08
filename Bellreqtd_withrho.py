import numpy as np
import re

def parse_density_dirac(expr, num_qubits):
    dim = 2**num_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    # Split terms by +
    terms = expr.replace(" ", "").split('+')
    for term in terms:
        # Match coefficient and ket-bra
        match = re.match(r'([0-9./]*)\|([01]+)><([01]+)\|', term)
        if not match:
            raise ValueError(f"Invalid term: {term}")

        coeff_str, ket, bra = match.groups()

        # Coefficient handling
        if coeff_str == '':
            coeff = 1
        else:
            coeff = eval(coeff_str)

        i = int(ket, 2)
        j = int(bra, 2)

        rho[i, j] += coeff
    return rho

# ---------- Single-qubit gates ----------
def I():
    return np.eye(2, dtype=complex)

def H():
    return (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype=complex)

# ---------- Tensor product ----------
def kron(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def single(G,q,n):
    return kron(*[G if i==q else I() for i in range(n)])

# ---------- CNOT from definition ----------
def CNOT(control, target, n):
    dim = 2**n
    U = np.zeros((dim, dim))
    for i in range(dim):
        bits = list(format(i, f'0{n}b'))
        if bits[control] == '1':
            bits[target] = '0' if bits[target] == '1' else '1'
        j = int("".join(bits), 2)
        U[j, i] = 1
    return U

def get_rho_from_SE_density(num_qubits, sys_qubits):
    print("\n--- Enter SYSTEM density matrix ---")
    rho_S_str = input("ρ_S (e.g. 1/2|00><00| + 1/2|11><11|): ")

    print("\n--- Enter ENVIRONMENT density matrix ---")
    rho_E_str = input("ρ_E (e.g. 1/2|00><00| + 1/2|11><11|): ")

    # parse separately
    rho_S = parse_density_dirac(rho_S_str, sys_qubits)
    rho_E = parse_density_dirac(rho_E_str, num_qubits - sys_qubits)

    # tensor product
    rho = np.kron(rho_S, rho_E)

    return rho

def parse_superposed_state(input_str, num_qubits):
    state = np.zeros((2 ** num_qubits, 1), dtype=complex)
    input_str = input_str.replace(" ", "")
    input_str = input_str.replace("√", "sqrt")

    # Pattern:
    #   coefficient | bitstring >
    pattern = r'([+-]?[^|]*)\|([01]{%d})>' % num_qubits
    matches = re.findall(pattern, input_str)

    if not matches:
        raise ValueError(f"Could not parse state for {num_qubits} qubits.")

    for coeff_str, basis_str in matches:
        # Default coefficient = 1
        if coeff_str in ("", "+"):
            coeff = 1.0
        elif coeff_str == "-":
            coeff = -1.0
        else:
            coeff = eval(
                coeff_str,
                {"sqrt": np.sqrt, "np": np, "builtins": {}}
            )

        index = int(basis_str, 2)
        state[index] += coeff

    # Normalize
    norm = np.sqrt(np.sum(np.abs(state)**2))
    if norm > 0:
        state = state / norm

    return state

def partial_trace(rho, traced_qubits, num_qubits):
    traced_qubits = sorted(traced_qubits)
    retained_qubits = [i for i in range(num_qubits) if i not in traced_qubits]

    # reshape rho into tensor
    reshaped = rho.reshape([2] * (2 * num_qubits))

    # permutation: traced_out, retained_out, traced_in, retained_in
    perm = (
        traced_qubits
        + retained_qubits
        + [i + num_qubits for i in traced_qubits]
        + [i + num_qubits for i in retained_qubits]
    )
    reshaped = np.transpose(reshaped, perm)

    d_tr = 2 ** len(traced_qubits)
    d_ret = 2 ** len(retained_qubits)

    reshaped = reshaped.reshape(d_tr, d_ret, d_tr, d_ret)

    # correct partial trace
    rho_red = np.trace(reshaped, axis1=0, axis2=2)

    return rho_red

## Diagonal check
def is_diagonal(rho, tol=1e-12):
    off = rho - np.diag(np.diag(rho))
    return np.all(np.abs(off) < tol)


## Praggnya REQ Definition
def REQ(rho, tol=1e-12):
    """
    Definition:

    If rho is diagonal:
        REQ = 0

    Else:
        REQ = Tr(rho log2 rho) + log2(d)
            = log2(d) - S(rho)
    """
    # Make Hermitian + normalize
    rho = (rho + rho.conj().T)/2
    rho = rho / np.trace(rho)

    # Check diagonal
    off = rho - np.diag(np.diag(rho))
    if np.all(np.abs(off) < tol):
        return 0.0

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(rho)

    # Compute Tr(rho log2 rho)
    tr_rho_log_rho = 0.0

    for lam in eigvals:
        if lam > tol:
            tr_rho_log_rho += lam * np.log2(lam)

    # Dimension
    d = rho.shape[0]

    # Final REQ
    return tr_rho_log_rho + np.log2(d)


def trace_distance(rho1, rho2):
    diff = rho1 - rho2
    s = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(np.abs(s))

# ======================
# PRODUCT CHECK
# ======================

def is_product_state(psi, d_sys, d_env, tol=1e-10):
    psi_tensor = psi.reshape(d_sys, d_env)
    _, s, _ = np.linalg.svd(psi_tensor)
    return s[1] < tol

def partial_trace_kraus(rho, keep_qubits, num_qubits):

    keep_qubits = sorted(keep_qubits)
    traced_qubits = [i for i in range(num_qubits) if i not in keep_qubits]

    dim_keep = 2 ** len(keep_qubits)
    dim_trace = 2 ** len(traced_qubits)

    rho_tensor = rho.reshape([2]*num_qubits*2)

    perm = (
        keep_qubits
        + traced_qubits
        + [i + num_qubits for i in keep_qubits]
        + [i + num_qubits for i in traced_qubits]
    )

    rho_perm = np.transpose(rho_tensor, perm)
    rho_perm = rho_perm.reshape(dim_keep, dim_trace, dim_keep, dim_trace)

    rho_red = np.zeros((dim_keep, dim_keep), dtype=complex)

    for i in range(dim_trace):
        rho_red += rho_perm[:, i, :, i]

    return rho_red

#### ACTUAL CORRECT
def kraus_system(U, rho_in, num_qubits, sys_qubits):

    dim_sys = 2**sys_qubits
    dim_env = 2**(num_qubits - sys_qubits)

    # --- STEP 1: get environment state ---
    env_idx = list(range(sys_qubits, num_qubits))
    rho_env = partial_trace_kraus(rho_in, env_idx, num_qubits)

    eigvals, eigvecs = np.linalg.eigh(rho_env)

    # --- STEP 2: reshape unitary ---
    # U[sys_out, env_out, sys_in, env_in]
    U4 = U.reshape(dim_sys, dim_env, dim_sys, dim_env)

    K_list = []

    # --- STEP 3: build Kraus ---
    for k in range(dim_env):   # eigenstates of environment

        lam = eigvals[k]
        if lam < 1e-12:
            continue

        phi_k = eigvecs[:, k]   # env eigenvector

        for j in range(dim_env):   # environment output index

            K = np.zeros((dim_sys, dim_sys), dtype=complex)

            for beta in range(dim_env):   # env input basis
                K += phi_k[beta] * U4[:, j, :, beta]

            K_list.append(np.sqrt(lam) * K)

    return K_list

def kraus_environment(U, rho_in, num_qubits, sys_qubits):
    dim_sys = 2**sys_qubits
    dim_env = 2**(num_qubits - sys_qubits)

    sys_qubits_idx = list(range(sys_qubits))

    rho_sys = partial_trace_kraus(rho_in, sys_qubits_idx, num_qubits)  # keeps sys

    eigvals, eigvecs = np.linalg.eigh(rho_sys)
    psi_sys = eigvecs[:, np.argmax(eigvals)]  # dominant eigenvector

    U4 = U.reshape(dim_sys, dim_env, dim_sys, dim_env)

    K_list = []
    for i in range(dim_sys):
        K = np.zeros((dim_env, dim_env), dtype=complex)
        for alpha in range(dim_sys):
            K += psi_sys[alpha] * U4[i, :, alpha, :]
        K_list.append(K)

    return K_list

def check_kraus(K_list, name="Kraus"):
    dim = K_list[0].shape[0]
    I = np.eye(dim, dtype=complex)

    sum_K = np.zeros((dim, dim), dtype=complex)
    for K in K_list:
        sum_K += K.conj().T @ K

    print(f"\n--- Checking {name} ---")
    print("Sum K†K =\n", np.round(sum_K, 5))

    if np.allclose(sum_K, I, atol=1e-6):
        print("CPTP (Trace Preserving)")
    else:
        print("Not CPTP")

    # Unital check
    sum_KK = np.zeros((dim, dim), dtype=complex)
    for K in K_list:
        sum_KK += K @ K.conj().T
    
    if np.allclose(sum_KK, I, atol=1e-6):
        print("Unital")
    else:
        print("Non Unital")

    print("Sum KK† =\n", np.round(sum_KK, 5))

def print_kraus(K_list, name="Kraus"):
    print(f"\n--- {name} ---")
    print(f"Total number: {len(K_list)}\n")

    for i, K in enumerate(K_list):
        print(f"{name}_{i} =")
        print(np.round(K.real, 4) + 1j*np.round(K.imag, 4))
        print("-" * 40)

def verify_kraus_action(K_list, rho_in, rho_out, name="Check"):

    if len(K_list) == 0:
        print(f"\n❌ {name}: No Kraus operators")
        return

    # compute via Kraus
    rho_kraus = sum(K @ rho_in @ K.conj().T for K in K_list)

    print(f"\n--- {name} ---")
    print("Direct rho_out:\n", np.round(rho_out, 5))
    print("\nKraus rho_out:\n", np.round(rho_kraus, 5))

    diff = np.linalg.norm(rho_out - rho_kraus)

    print("\nDifference ρ_out - Kraus =", diff)

    if np.allclose(rho_out, rho_kraus, atol=1e-6):
        print("✅ MATCH: Kraus representation is correct")
    else:
        print("❌ NOT MATCHING")



def main():
    num_qubits = int(input("Total qubits: "))
    sys_qubits = int(input("Number of system qubits: "))
    env_qubits = num_qubits - sys_qubits
    print(f"Environment qubits: {env_qubits}")

    system_qubits = list(range(sys_qubits))
    environ_qubits = list(range(sys_qubits, num_qubits))

    #env_qubits = int(input("Enter number of environment qubits: "))
    #trace_choice = input("Trace over environment or system? (e/s): ").lower().strip()
    trace_choice = 'e'
    
    if trace_choice == 'e':
        traced_qubits = list(range(num_qubits - env_qubits, num_qubits))
    else:
        traced_qubits = list(range(0, num_qubits - env_qubits))
        
    retained_qubits = [i for i in range(num_qubits) if i not in traced_qubits]

    print("Traced Qubits:", traced_qubits)
    print("Retained Qubits:", retained_qubits)

    traced_env = list(range(sys_qubits, num_qubits))
    traced_sys = list(range(sys_qubits))

    mode = input("\nEnter input type (psi / rho / tensor): ").strip().lower()
    if mode == "psi":
        input1 = input("\nEnter input state 1 (e.g 1/sqrt(2)|100>): ")
        input2 = input("Enter input state 2 : ")

        psi1 = parse_superposed_state(input1, num_qubits)
        psi2 = parse_superposed_state(input2, num_qubits)

        print(f"psi1 shape: {psi1.shape} | psi2 shape: {psi2.shape}")

        rho1_in = np.dot(psi1, psi1.conj().T)
        rho2_in = np.dot(psi2, psi2.conj().T)
    
    elif mode == "rho":
        input1 = input("\nEnter density matrix 1: ")
        input2 = input("Enter density matrix 2: ")

        rho1_in = parse_density_dirac(input1, num_qubits)
        rho2_in = parse_density_dirac(input2, num_qubits)

        print(f"rho1 shape: {rho1_in.shape} | rho2 shape: {rho2_in.shape}")

    elif mode == 'tensor':
        print("\n--- STATE 1 ---")
        rho1_in = get_rho_from_SE_density(num_qubits, sys_qubits)
        print("\n Rho 1 =" , rho1_in)

        print("\n--- STATE 2 ---")
        rho2_in = get_rho_from_SE_density(num_qubits, sys_qubits)
        print("\n Rho 2 =" , rho2_in)

    else:
        raise ValueError("Invalid input type. Choose 'psi' or 'rho'")

    '''
    a = 1/np.sqrt(2)
    U = np.array([[a,0,0,a],
              [0,a,a,0],
              [a,0,0,-a],
              [0,a,-a,0]])
    '''

    # ---- Bell-state unitary ----

    U = single(H(), 0, num_qubits)
    U = CNOT(0, 1, num_qubits) @ U

    print("\nUnitary U:\n", U)

    
    if mode == 'psi' :
        # Apply unitary evolution
        out1 = U @ psi1
        out2 = U @ psi2
        print("1st output state :",out1)
        print("2nd output state :",out2)

        rho1_out = out1 @ out1.conj().T
        rho2_out = out2 @ out2.conj().T

    elif mode == 'rho' or mode == 'tensor' :
        rho1_out = np.dot(np.dot(U, rho1_in), U.conj().T)
        rho2_out = np.dot(np.dot(U, rho2_in), U.conj().T)

    else :
        raise ValueError("Invalid input type. Choose 'psi' or 'rho'")

    print(f"\nOutput rho1 shape: {rho1_out.shape}")
    print(f"Output rho2 shape: {rho2_out.shape}")

    #print(f"\nOutput rho1 : {rho1_out}")
    #print(f"Output rho2 : {rho2_out}")
    
    rho1_in_red = partial_trace(rho1_in, traced_qubits, num_qubits)
    rho1_out_red = partial_trace(rho1_out, traced_qubits, num_qubits)

    print("\nReduced first input system density matrix:\n", np.round(rho1_in_red.real, 4) + 1j * np.round(rho1_in_red.imag, 4))
    print("\nReduced first output system density matrix:\n", np.round(rho1_out_red.real, 4) + 1j * np.round(rho1_out_red.imag, 4))
    
    rho1_in_traced = partial_trace(rho1_in, retained_qubits, num_qubits)
    rho1_out_traced = partial_trace(rho1_out, retained_qubits, num_qubits)
    
    print("\nReduced first input environment density matrix:\n", np.round(rho1_in_traced.real, 4) + 1j * np.round(rho1_in_traced.imag, 4))
    print("\nReduced first output environment density matrix:\n", np.round(rho1_out_traced.real, 4) + 1j * np.round(rho1_out_traced.imag, 4))

    rho2_in_red = partial_trace(rho2_in, traced_qubits, num_qubits)
    rho2_out_red = partial_trace(rho2_out, traced_qubits, num_qubits)

    print("\nReduced second input system density matrix:\n", np.round(rho2_in_red.real, 4) + 1j * np.round(rho2_in_red.imag, 4))
    print("\nReduced second output system density matrix:\n", np.round(rho2_out_red.real, 4) + 1j * np.round(rho2_out_red.imag, 4))
    
    rho2_in_traced = partial_trace(rho2_in, retained_qubits, num_qubits)
    rho2_out_traced = partial_trace(rho2_out, retained_qubits, num_qubits)

    print("\nReduced second input environment density matrix:\n", np.round(rho2_in_traced.real, 4) + 1j * np.round(rho2_in_traced.imag, 4))
    print("\nReduced second output environment density matrix:\n", np.round(rho2_out_traced.real, 4) + 1j * np.round(rho2_out_traced.imag, 4))

    '''
    d1 = trace_distance(rho1_in_red, rho2_in_red)
    d2 = trace_distance(rho1_in_traced, rho2_in_traced)
    d3 = trace_distance(rho1_out_red, rho2_out_red)
    d4 = trace_distance(rho1_out_traced, rho2_out_traced)
    d5 = trace_distance(rho1_in, rho2_in)
    d6 = trace_distance(rho1_out, rho2_out)
    print(f"\nTrace distance between input reduced system states: {d1:.6f}")
    print(f"\nTrace distance between input reduced environment states: {d2:.6f}")
    print(f"\nTrace distance between output reduced system states: {d3:.6f}")
    print(f"\nTrace distance between output reduced environment states: {d4:.6f}")
    print(f"\nTrace distance between input overall states: {d5:.6f}")
    print(f"\nTrace distance between output overall states: {d6:.6f}")

    # NEW RE
    print("\nRelative Entropy:")
    print("System input 1:", REQ(rho1_in_red))
    print("System output 1:", REQ(rho1_out_red))
    print("System input 2:", REQ(rho2_in_red))
    print("System output 2:", REQ(rho2_out_red))

    print("Environment input 1:", REQ(rho1_in_traced))
    print("Environment output 1:", REQ(rho1_out_traced))
    print("Environment input 2:", REQ(rho2_in_traced))
    print("Environment output 2:", REQ(rho2_out_traced))

    '''
    dim = U.shape[0]
    Iden = np.eye(dim , dtype = complex)

    UdagU = U.conj().T @ U
    UUdag = U @ U.conj().T

    err1 = np.linalg.norm(UdagU - Iden)
    err2 = np.linalg.norm(UUdag - Iden)

    print(" U†U - I  =", err1)
    print(" UU† - I  =", err2)

    tol = 1e-9
    print("U†U = I ?", err1 < tol)
    print("UU† = I ?", err2 < tol)

    sys_idx = list(range(sys_qubits))
    env_idx = list(range(sys_qubits, num_qubits))

    rho1_sys_in  = partial_trace_kraus(rho1_in,  sys_idx, num_qubits)
    rho1_sys_out = partial_trace_kraus(rho1_out, sys_idx, num_qubits)
    rho2_sys_in  = partial_trace_kraus(rho2_in,  sys_idx, num_qubits)
    rho2_sys_out = partial_trace_kraus(rho2_out, sys_idx, num_qubits)

    rho1_env_in  = partial_trace_kraus(rho1_in,  env_idx, num_qubits)
    rho1_env_out = partial_trace_kraus(rho1_out, env_idx, num_qubits)
    rho2_env_in  = partial_trace_kraus(rho2_in,  env_idx, num_qubits)
    rho2_env_out = partial_trace_kraus(rho2_out, env_idx, num_qubits)

     # ---- STATE 1 ----

    K1_sys = kraus_system(U, rho1_in, num_qubits, sys_qubits)
    print_kraus(K1_sys, "State 1 System Kraus")
    check_kraus(K1_sys, "State 1 System Kraus")
    verify_kraus_action(K1_sys, rho1_sys_in, rho1_sys_out, "State 1 System")

    K1_env = kraus_environment(U, rho1_in, num_qubits, sys_qubits)
    print_kraus(K1_env, "State 1 Environment Kraus")
    check_kraus(K1_env, "State 1 Environment Kraus")
    verify_kraus_action(K1_env, rho1_env_in, rho1_env_out, "State 1 Environment")

     # ---- STATE 2 ----

    K2_sys = kraus_system(U, rho2_in, num_qubits, sys_qubits)
    print_kraus(K2_sys, "State 2 System Kraus")
    check_kraus(K2_sys, "State 2 System Kraus")
    verify_kraus_action(K2_sys, rho2_sys_in, rho2_sys_out, "State 2 System")

    K2_env = kraus_environment(U, rho2_in, num_qubits, sys_qubits)
    print_kraus(K2_env, "State 2 Environment Kraus")
    check_kraus(K2_env, "State 2 Environment Kraus")
    verify_kraus_action(K2_env, rho2_env_in, rho2_env_out, "State 2 environment")

    psi1_curr = psi1.copy()
    psi2_curr = psi2.copy()

    
    def diagnostics_two(label, p1_in, p2_in, G):

        print("\n==============================")
        print(label)
        print("==============================")

        print("\npsi1 IN:\n", np.round(p1_in,4))
        print("\npsi2 IN:\n", np.round(p2_in,4))

        # Apply gate
        p1_out = G @ p1_in
        p2_out = G @ p2_in

        print("\npsi1 OUT:\n", np.round(p1_out,4))
        print("\npsi2 OUT:\n", np.round(p2_out,4))

        # Density matrices
        rho1I = p1_in @ p1_in.conj().T
        rho2I = p2_in @ p2_in.conj().T
        rho1O = p1_out @ p1_out.conj().T
        rho2O = p2_out @ p2_out.conj().T
        
        r1_sys_in  = partial_trace_kraus(rho1I,  sys_idx, num_qubits)
        r1_sys_out = partial_trace_kraus(rho1O, sys_idx, num_qubits)
        r2_sys_in  = partial_trace_kraus(rho2I,  sys_idx, num_qubits)
        r2_sys_out = partial_trace_kraus(rho2O, sys_idx, num_qubits)

        r1_env_in  = partial_trace_kraus(rho1I,  env_idx, num_qubits)
        r1_env_out = partial_trace_kraus(rho1O, env_idx, num_qubits)
        r2_env_in  = partial_trace_kraus(rho2I,  env_idx, num_qubits)
        r2_env_out = partial_trace_kraus(rho2O, env_idx, num_qubits)

         # ---- STATE 1 ----

        K1_sys = kraus_system(G, rho1I, num_qubits, sys_qubits)
        print_kraus(K1_sys, "State 1 System Kraus")
        check_kraus(K1_sys, "State 1 System Kraus")
        verify_kraus_action(K1_sys, r1_sys_in, r1_sys_out, "State 1 System")

        K1_env = kraus_environment(G, rho1I, num_qubits, sys_qubits)
        print_kraus(K1_env, "State 1 Environment Kraus")
        check_kraus(K1_env, "State 1 Environment Kraus")
        verify_kraus_action(K1_env, r1_env_in, r1_env_out, "State 1 Environment")

         # ---- STATE 2 ----

        K2_sys = kraus_system(G, rho2I, num_qubits, sys_qubits)
        print_kraus(K2_sys, "State 2 System Kraus")
        check_kraus(K2_sys, "State 2 System Kraus")
        verify_kraus_action(K2_sys, r2_sys_in, r2_sys_out, "State 2 System")

        K2_env = kraus_environment(G, rho2I, num_qubits, sys_qubits)
        print_kraus(K2_env, "State 2 Environment Kraus")
        check_kraus(K2_env, "State 2 Environment Kraus")
        verify_kraus_action(K2_env, r2_env_in, r2_env_out, "State 2 environment")
        

        return p1_out, p2_out

    # ---- H on qubit 0 ----
    G = single(H(), 0, num_qubits)

    psi1_curr, psi2_curr = diagnostics_two("After H(0)",psi1_curr,psi2_curr,G)

    # ---- Bell-state entangling CNOT ----
    G = CNOT(0, 1, num_qubits)

    psi1_curr, psi2_curr = diagnostics_two("After CNOT(0->1)",psi1_curr,psi2_curr,G)


    AdagA = G.conj().T @ G
    AAdag = G @ G.conj().T

    err3 = np.linalg.norm(AdagA - Iden)
    err4= np.linalg.norm(AAdag - Iden)

    print(" U†U - I  =", err3)
    print(" UU† - I  =", err4)

    tol = 1e-9
    print("U†U = I ?", err3 < tol)
    print("UU† = I ?", err4 < tol)


if __name__ == "__main__":
    main()

#  1/2|00><00| + 1/2|00><11| + 1/2|11><00| + 1/2|11><11|