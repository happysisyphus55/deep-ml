import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    # check square matrices condition
    if T_t.shape[0] != T_t.shape[1] or S_t.shape[0] != S_t.shape[1]:
        return torch.tensor(-1.)
    if S_t.ndim != 2 or S_t.shape[0] != S_t.shape[1]:
        return torch.tensor(-1.)
    
    # validate dimensional
    if T_t.shape[1] != A_t.shape[0] or A_t.shape[1] != S_t.shape[0]:
        return torch.tensor(-1.)

    # check invertibility first to ensure determinants are not 0
    if torch.isclose(torch.linalg.det(T_t), torch.tensor(0.0), atol=1e-6) or \
       torch.isclose(torch.linalg.det(S_t), torch.tensor(0.0), atol=1e-6):
        return torch.tensor(-1.)
    
    try:
        # 4. Compute the inverse of T
        T_inv = torch.linalg.inv(T_t)
        
        # 5. Perform the transformation T⁻¹ A S
        result = T_inv @ A_t @ S_t
        
        # 6. Round the output matrix to 3 decimal places
        return torch.round(result, decimals=3)
        
    except RuntimeError:
        # Handles any unforeseen numerical issues during inversion
        return torch.tensor(-1.)  
