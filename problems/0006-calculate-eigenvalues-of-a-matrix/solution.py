import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    mat = torch.as_tensor(matrix,dtype=torch.float32) 
    
    a, b = mat[0,0], mat[0,1]
    c, d = mat[1,0], mat[1,1]

    trace = a+ d 
    det = a* d - b * c 
    disc = torch.sqrt(trace**2 - 4 * det)

    l1 = (trace + disc) / 2
    l2 = (trace - disc) / 2 
    return torch.stack([l1,l2])