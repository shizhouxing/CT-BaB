# Modified by Zhouxing Shi for training-time branch-and-bound.
# Adapted from https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/main/complete_verifier/input_split.
#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from typing import Union
import torch
from torch import Tensor
from typing import Union, Tuple
from tensor_storage import TensorStorage


class UnsortedInputDomainList:
    """Unsorted domain list for input split."""

    def __init__(self, storage_depth=1, sort_index=None, sort_descending=True,
                 use_split_idx=True):
        super(UnsortedInputDomainList, self).__init__()
        self.lb = None
        self.dm_l = None
        self.dm_u = None
        self.sort_index = sort_index
        self.split_idx = None
        self.storage_depth = storage_depth
        self.sort_descending = sort_descending
        self.use_split_idx = use_split_idx

    def __len__(self):
        if self.dm_l is None:
            return 0
        return self.dm_l.num_used

    def __getitem__(self, idx):
        return (
            self.lb._storage[idx],
            self.dm_l._storage[idx],
            self.dm_u._storage[idx],
        )

    def add(
        self,
        lb: Tensor,
        dm_l: Tensor,
        dm_u: Tensor,
        split_idx: Union[Tensor, None] = None,
    ) -> None:
        # check shape correctness
        batch = len(lb)
        if batch == 0:
            return
        if self.use_split_idx:
            assert split_idx is not None, "Cannot accept split_idx"
            assert len(split_idx) == batch
            assert split_idx.shape[1] == self.storage_depth
        else:
            assert split_idx is None, "Expected to receive split_idx"
        assert len(dm_l) == len(dm_u) == batch
        # initialize attributes using input shapes
        if self.lb is None:
            self.lb = TensorStorage(lb.shape)
        if self.dm_l is None:
            self.dm_l = TensorStorage(dm_l.shape)
        if self.dm_u is None:
            self.dm_u = TensorStorage(dm_u.shape)
        if self.split_idx is None and self.use_split_idx:
            self.split_idx = TensorStorage([None, self.storage_depth])
        # append the tensors
        self.lb.append(lb.type(self.lb.dtype).to(self.lb.device))

        self.dm_l.append(dm_l.type(self.dm_l.dtype).to(self.dm_l.device))
        self.dm_u.append(dm_u.type(self.dm_u.dtype).to(self.dm_u.device))
        if self.use_split_idx:
            self.split_idx.append(
                split_idx.type(self.split_idx.dtype).to(self.split_idx.device)
            )

    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch = min(len(self), batch)
        assert batch > 0, "List of InputDomain is empty; pop failed."
        lb = self.lb.pop(batch).to(device=device, non_blocking=True)
        dm_l = self.dm_l.pop(batch).to(device=device, non_blocking=True)
        dm_u = self.dm_u.pop(batch).to(device=device, non_blocking=True)
        if self.use_split_idx:
            split_idx = self.split_idx.pop(batch).to(device=device, non_blocking=True)
        else:
            split_idx = None
        return lb, dm_l, dm_u, split_idx

    def _get_sort_margin(self, margin):
        if self.sort_index is not None:
            return margin[..., self.sort_index]
        else:
            return margin.max(dim=1).values

    def get_topk_indices(self, k=1, largest=False, threshold=0.):
        assert k <= len(self), print("Asked indices more than domain length.")
        lb = self.lb._storage[: self.lb.num_used]
        indices = self._get_sort_margin(lb - threshold).topk(k, largest=largest).indices
        return indices

    def sort(self, threshold=0.):
        lb = self.lb._storage[: self.lb.num_used]
        indices = self._get_sort_margin(lb - threshold).argsort(
            descending=self.sort_descending)
        self.lb._storage[: self.lb.num_used] = self.lb._storage[indices]
        self.dm_l._storage[: self.dm_l.num_used] = self.dm_l._storage[indices]
        self.dm_u._storage[: self.dm_u.num_used] = self.dm_u._storage[indices]
        if self.use_split_idx:
            self.split_idx._storage[: self.split_idx.num_used] = self.split_idx._storage[indices]
