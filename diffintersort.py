"""
Copyright (C) 2024  GSK plc - Mathieu Chevalley;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
from torch.optim import Adam
import numpy as np
from intersort import score_ordering
from scipy.optimize import linear_sum_assignment

def transitive_closure(matrix, depth):
    n = len(matrix)
    reach = np.array(matrix).astype(bool)
    
    for k in range(depth):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    
    return reach.astype(int)


def diffintersort(score_matrix, d, init_ordering = None, scaling=0.1, n_iter=100, lr=0.001, n_iter_sinkhorn = 500, t_sinkhorn = 0.05, eps=0.3):
    p_scale = 0.001
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    p = p_scale * torch.randn((d), device=device)
    if init_ordering is not None:
        _, indices = torch.sort(torch.tensor(init_ordering), descending=False)
        p_sorted, _ = torch.sort(p, descending=False)
        p = p_sorted[indices]
    p.requires_grad = True
    p_opt = Adam([p],
                lr=lr,
                betas=(0.9,0.99),
                )

    score_matrix[score_matrix > 0.0] -= eps
    transitive = transitive_closure(score_matrix > 0.0, depth=d)
    score_matrix[transitive > 0.0] = scaling * d 
    scoring_matrix = score_matrix.copy()
    score_matrix_torch = torch.tensor(score_matrix).to(device)
    p_cpu = p.detach().cpu().numpy()
    new_ordering = np.argsort(p_cpu)
    score_new_ordering = score_ordering(new_ordering, scoring_matrix, d, eps)
    best_score = score_new_ordering
    best = p_cpu
    best, perm = opt_step(scoring_matrix, d, n_iter, n_iter_sinkhorn, t_sinkhorn, p, p_opt, score_matrix_torch, best_score)
    return np.argsort(best), perm

def opt_step(score_matrix, d, n_iter, n_iter_sinkhorn, t_sinkhorn, p, p_opt, score_matrix_torch, best_score, early_stop_steps=500):
    best = p.detach().cpu().numpy()
    early_stop_count = early_stop_steps 
    for i in range(n_iter):
        p_opt.zero_grad()
        loss = torch.tensor(0, dtype=p.dtype).to(p.device)

        sig_p = (torch.sigmoid(p) * 2) - 1
        perm_matrix, perm = compute_perm_matrix(sig_p, d, n_iter_sinkhorn, t_sinkhorn)
        score = perm_matrix * score_matrix_torch
        loss += -torch.sum(torch.mean(score, dim=0)) 
        loss.backward()
        p_opt.step()
        
        p_cpu = p.detach().cpu().numpy()
        new_ordering = np.argsort(p_cpu)
        score_new_ordering = score_ordering(new_ordering, score_matrix, d, 0)
        if score_new_ordering > best_score:
            best = p_cpu.copy()
            best_score = score_new_ordering
            early_stop_count = early_stop_steps
        else:
            early_stop_count -= 1
        if score_new_ordering == best_score:
            best = p_cpu.copy()
        if early_stop_count <= 0:
            break

    return best, perm.detach().cpu()[0]

"""MIT License 

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE"""
def fill_triangular(vec: torch.Tensor, d, upper: bool = False,) -> torch.Tensor:
    """
    Args:
        vec: A tensor of shape (..., n(n-1)/2)
        upper: whether to fill the upper or lower triangle
    Returns:
        An array of shape (..., n, n), where the strictly upper (lower) triangle is filled from vec
        with zeros elsewhere
    """
    num_nodes = d
    if upper:
        idxs = torch.triu_indices(num_nodes, num_nodes, offset=1, device=vec.device)
    else:
        idxs = torch.tril_indices(num_nodes, num_nodes, offset=-1, device=vec.device)
    output = torch.zeros(vec.shape[:-1] + (num_nodes, num_nodes), device=vec.device)
    output[..., idxs[0, :], idxs[1, :]] = vec
    return output

def compute_perm_matrix(p: torch.Tensor, d: int, sinkhorn_n_iter: int = 500, t: float = 0.05):
    def log_sinkhorn_norm(log_alpha: torch.Tensor, tol= 1e-3):
        for i in range(sinkhorn_n_iter):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
            exp_log_alpha = log_alpha.exp()
            if torch.abs(1.-exp_log_alpha.sum(-1)).max()<tol and torch.abs(1.-exp_log_alpha.sum(-2)).max()<tol:
                #print(i)
                break
        return log_alpha.exp()
    
    o_scale = 1
    O = o_scale * torch.arange(1, d+1, dtype=p.dtype).expand(1, -1).to(p.device)
    X = torch.matmul(p.unsqueeze(-1), O.unsqueeze(-2))

    perm = log_sinkhorn_norm(X / t)

    perm_matrix = torch.zeros_like(perm)
    for i in range(perm.shape[0]):
        row_ind, col_ind = linear_sum_assignment(-perm[i].squeeze().cpu().detach().numpy())
        perm_indices = list(zip(row_ind, col_ind))            
        perm_indices = [(i,) + idx for idx in perm_indices]
        perm_indices = tuple(zip(*perm_indices))
        perm_matrix[perm_indices] = 1.0
    perm_matrix_hard = (perm_matrix - perm).detach() + perm

    full_lower = torch.ones(1, int((d - 1) * d / 2)).to(p.device)
    full_lower = fill_triangular(full_lower, d, upper=True)
    mask_matrix = full_lower
    adj_matrix = torch.matmul(
        torch.matmul(perm_matrix_hard, mask_matrix), perm_matrix_hard.transpose(-1, -2)
    )

    return adj_matrix, perm
 

    