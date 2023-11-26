"""Utils for scatter.

Adaptation of torch_scatter/scatter.py from:
https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py
"""

import torch



def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcasts `src` to the shape of `other`."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    num_edges: int | None = None,
) -> torch.Tensor:
    """Add all values from the `src` tensor into `out` at the indices."""
    index = broadcast(index, src, dim)
    # print('num_edges', num_edges)
    # print('scatter_sum')
    # print('index', index.shape)
    # print(index)
    # print('dim size', dim_size)
    
    if out is None:
        size = list(src.size())
        #print('size', size)
        if dim_size is not None:
            size[dim] = dim_size
            # print('1')
            # print('dim size', dim_size)
        elif index.numel() == 0:
            size[dim] = 0
            #print('2')
        
        #elif dim_size is None and int(index.max()) + 1:      
        else:
            max_index = int(index.max()) + 1
            size[dim] = max(num_edges, max_index) if num_edges is not None else max_index

            
        # print('size[0]', size[dim])
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        # print('out shape', out.shape)
        # print(out.scatter_add_(dim = dim, index = index, src = src).shape)
        
        
        return out.scatter_add_(dim=dim, index=index, src=src) 
    else:
        return out.scatter_add_(dim=dim, index=index, src=src)  # Pass num_edges



def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    num_edges: int | None = None,
) -> torch.Tensor:
    """Add all values from the `src` tensor into `out` at the indices."""
    
    #print('this function is called')
    return scatter_sum(src, index, dim, out, dim_size, num_edges)




def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Compute the mean value of all values from the `src` tensor into `out`."""
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


SCATTER_DICT = {"sum": scatter_sum, "mean": scatter_mean, "add": scatter_sum}


def scatter(scatter: str):
    """Return the scatter function."""
    if isinstance(scatter, str) and scatter in SCATTER_DICT:
        return SCATTER_DICT[scatter]
    else:
        raise ValueError(
            f"scatter must be callable or string: {list(SCATTER_DICT.keys())}"
        )
