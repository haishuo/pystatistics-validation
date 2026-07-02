"""Wood (2011) stable method: solve the penalized WLS via QR of the AUGMENTED
matrix [sqrt(W) X ; B] where B'B = sum lam_j S_j, instead of the normal
equations A = X'WX + sum lam_j S_j. Claim: cond(augmented) ~ sqrt(cond(A)), so a
regime that is cond~4e8 (fp32-unsafe) on the normal equations becomes cond~2e4
(fp32-safe) on the augmented QR -> an fp32 GPU path becomes numerically viable."""
import numpy as np
from scipy.linalg import sqrtm

def near_rank_deficient(n, p=40, decay=0.55, seed=1):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, p)))
    Vt, _ = np.linalg.qr(rng.standard_normal((p, p)))
    X = (U * (decay ** np.arange(p))) @ Vt
    return X

def compare(n=5000, p=40, lam=1e-8, seed=1):
    X = near_rank_deficient(n, p, seed=seed)
    w = np.ones(n); z = np.random.default_rng(2).standard_normal(n)
    D = np.zeros((p-2, p))
    for i in range(p-2):
        D[i, i]=1.0; D[i, i+1]=-2.0; D[i, i+2]=1.0
    S = D.T @ D
    # --- normal equations (what the library does) ---
    A64 = (X.T*w)@X + lam*S
    cond_normal = np.linalg.cond(A64)
    beta_ref = np.linalg.solve(A64, (X.T*w)@z)          # fp64 reference
    # --- augmented matrix: penalty square root via eigh (psd) ---
    evals, evecs = np.linalg.eigh(lam*S)
    evals = np.clip(evals, 0, None)
    B = (evecs * np.sqrt(evals)) @ evecs.T              # B'B = lam*S, (p,p)
    Aug = np.vstack([np.sqrt(w)[:,None]*X, B])          # (n+p, p)
    cond_aug = np.linalg.cond(Aug)
    # fp32 QR solve of the augmented LS  min ||Aug beta - [sqrt(w)z; 0]||
    rhs = np.concatenate([np.sqrt(w)*z, np.zeros(p)])
    Aug32 = Aug.astype(np.float32); rhs32 = rhs.astype(np.float32)
    Q, R = np.linalg.qr(Aug32)
    beta_qr32 = np.linalg.solve(R.astype(np.float64),
                                (Q.T.astype(np.float64) @ rhs))
    # fp32 normal-equations solve (the unsafe path) for contrast
    A32 = ((X.astype(np.float32).T*w.astype(np.float32))@X.astype(np.float32)
           ).astype(np.float64) + lam*S
    beta_ne32 = np.linalg.solve(A32, (X.T*w)@z)
    err_qr = np.linalg.norm(beta_qr32-beta_ref)/np.linalg.norm(beta_ref)
    err_ne = np.linalg.norm(beta_ne32-beta_ref)/np.linalg.norm(beta_ref)
    return cond_normal, cond_aug, err_ne, err_qr

print(f"{'lam':>8} {'cond_normalEq':>14} {'cond_augQR':>12} "
      f"{'beta_relerr_fp32_NE':>20} {'beta_relerr_fp32_QR':>20}")
for lam in [1e-4, 1e-6, 1e-8]:
    cn, ca, ene, eqr = compare(lam=lam)
    print(f"{lam:>8.0e} {cn:>14.2e} {ca:>12.2e} {ene:>20.2e} {eqr:>20.2e}")
