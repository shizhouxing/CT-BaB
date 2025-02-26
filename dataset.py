import math
import random
import torch
from tqdm import tqdm
from utils import logger
from tensor_storage import TensorStorage


class InputDomainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 lower_limit, upper_limit,  # region where we need to verify
                 hole_lower=None, hole_upper=None,    # region where we exclude from the verification
                 hole_size=0, border_size=None,
                 init_splits=1, max_init_size=1.0, scale_input=1.0,
                 use_even_splits=True, hole_mode=None,
                 max_in_hole_dim=None,
                 adv_size=None):
        self.lower_limit = torch.tensor(lower_limit) * scale_input
        self.upper_limit = torch.tensor(upper_limit) * scale_input
        self.box_size = self.upper_limit - self.lower_limit
        self.hole_size = hole_size
        if hole_lower is not None:
            self.hole_lower = torch.tensor(hole_lower)
            self.hole_upper = torch.tensor(hole_upper)
        else:
            self.hole_lower = self.lower_limit * hole_size
            self.hole_upper = self.upper_limit * hole_size
        self.hole_mode = hole_mode
        self.max_in_hole_dim = max_in_hole_dim
        self.border_size = border_size
        if self.border_size is None:
            self.border_lower = self.border_upper = None
        else:
            self.border_lower = self.lower_limit * self.border_size
            self.border_upper = self.upper_limit * self.border_size

        self.init_splits = init_splits
        self.x_dim = self.lower_limit.shape[0]
        if len(max_init_size) == 1:
            self.max_init_size = [max_init_size[0] * scale_input] * self.x_dim
        else:
            self.max_init_size = [s * scale_input for s in max_init_size]
        self.use_even_splits = use_even_splits

        self.dm_lb = TensorStorage((1, self.x_dim))
        self.dm_ub = TensorStorage((1, self.x_dim))
        self.dm_weight = TensorStorage((1,))
        self.dm_inside = TensorStorage((1,), dtype=torch.bool)
        self.splits = []

        self.seed_x_L, self.seed_x_U = [], []
        self._construct_initial_domains()
        self.seed_x_L = torch.concat(self.seed_x_L).view(-1, self.x_dim)
        self.seed_x_U = torch.concat(self.seed_x_U).view(-1, self.x_dim)

        self.adv_size = adv_size

    def __repr__(self):
        return (
            'InputDomainDataset(\n'
            f'  lower_limit={self.lower_limit},\n'
            f'  upper_limit={self.upper_limit},\n'
            f'  hole_lower={self.hole_lower},\n'
            f'  hole_upper={self.hole_upper}\n'
            f'  border_size={self.border_size}\n'
            f'  length={self.__len__()}'
            ')'
        )

    def _construct_initial_domains(self):
        if self.border_size is None:
            self._construct_subboxes(self.lower_limit, self.upper_limit,
                                     self.hole_lower, self.hole_upper)
        else:
            self._construct_subboxes(self.border_lower, self.border_upper,
                                     self.hole_lower, self.hole_upper)
            self._construct_subboxes(self.lower_limit, self.upper_limit,
                                     self.border_lower, self.border_upper,
                                     use_max_init_size=False)

    def _construct_subboxes(self, lower_limit, upper_limit, hole_lower, hole_upper,
                            use_max_init_size=True):
        if (hole_lower == 0).all() and (hole_upper == 0).all():
            self._construct_init_splits(lower_limit, upper_limit,
                                        use_max_init_size=use_max_init_size)
            return

        if self.hole_mode == 'non-overlapping':
            lower_cur = lower_limit.clone()
            upper_cur = upper_limit.clone()

            # TODO fix weight
            # weight = 1. / self.x_dim / 2 / lower.shape[0]

            for i in range(self.x_dim):
                lower = lower_cur.clone()
                upper = upper_cur.clone()
                lower[i] = hole_upper[i]
                self._construct_init_splits(lower, upper, use_max_init_size=use_max_init_size)

                lower = lower_cur.clone()
                upper = upper_cur.clone()
                upper[i] = hole_lower[i]
                self._construct_init_splits(lower, upper, use_max_init_size=use_max_init_size)

                lower_cur[i] = hole_lower[i]
                upper_cur[i] = hole_upper[i]
        elif self.hole_mode == 'equal':
            for i in range(self.x_dim):
                lower = lower_limit.clone()
                upper = upper_limit.clone()
                upper[i] = hole_lower[i]
                self._construct_init_splits(lower, upper, use_max_init_size=use_max_init_size)

                lower = lower_limit.clone()
                upper = upper_limit.clone()
                lower[i] = hole_upper[i]
                self._construct_init_splits(lower, upper, use_max_init_size=use_max_init_size)

        elif self.hole_mode == 'smart':
            self._construct_smart_hole(
                0, lower=[], upper=[],
                lower_limit=lower_limit.tolist(), upper_limit=upper_limit.tolist(),
                hole_lower=hole_lower.tolist(), hole_upper=hole_upper.tolist(),
                in_hole=0, max_in_hole=self.max_in_hole_dim)

        else:
            raise NotImplementedError

    def _construct_smart_hole(
            self, idx: int, lower: list, upper: list,
            lower_limit: torch.Tensor, upper_limit: torch.Tensor,
            hole_lower: torch.Tensor, hole_upper: torch.Tensor,
            in_hole: int, max_in_hole: int):

        if idx == self.x_dim:
            if in_hole > 0:
                self._construct_init_splits(torch.tensor(lower), torch.tensor(upper), verbose=False)
            if len(self) % 1000 == 0:
                logger.info('Dataset size: %d', len(self))
            return

        if in_hole == max_in_hole:
            # either in hole or not in hole
            lower_ = lower + [lower_limit[idx]]
            upper_ = upper + [upper_limit[idx]]
            self._construct_smart_hole(
                idx + 1, lower_, upper_, lower_limit, upper_limit,
                hole_lower, hole_upper, in_hole, max_in_hole)
        else:
            # in hole
            lower_ = lower + [hole_lower[idx]]
            upper_ = upper + [hole_upper[idx]]
            self._construct_smart_hole(
                idx + 1, lower_, upper_, lower_limit, upper_limit,
                hole_lower, hole_upper, in_hole + 1, max_in_hole)

            # not in hole
            lower_ = lower + [lower_limit[idx]]
            upper_ = upper + [hole_lower[idx]]
            self._construct_smart_hole(
                idx + 1, lower_, upper_, lower_limit, upper_limit,
                hole_lower, hole_upper, in_hole, max_in_hole)

            lower_ = lower + [hole_upper[idx]]
            upper_ = upper + [upper_limit[idx]]
            self._construct_smart_hole(
                idx + 1, lower_, upper_, lower_limit, upper_limit,
                hole_lower, hole_upper, in_hole, max_in_hole)

    def get_weight(self, lower, upper):
        self.box_size = self.box_size.to(lower.device)
        weight = ((upper - lower) / self.box_size.unsqueeze(0)).prod(dim=-1)
        return weight

    def _construct_init_splits(self, lower, upper, verbose=True, use_max_init_size=True):
        self.seed_x_L.append(lower)
        self.seed_x_U.append(upper)
        weight = self.get_weight(lower, upper)
        meshgrid = []
        box_size = []
        for i in range(self.x_dim):
            size_full = upper[i] - lower[i]
            if use_max_init_size:
                num_splits = math.ceil(size_full / self.max_init_size[i] - 1e-6)
            else:
                num_splits = 1
            if self.use_even_splits:
                if lower[i] < 0 and upper[i] > 0:
                    if num_splits % 2:
                        num_splits += 1
            num_splits = max(self.init_splits, num_splits)
            if verbose:
                logger.info(f'Input dimension {i}: '
                            f'size={size_full:.5f}, '
                            f'lower={lower[i]:.5f}, '
                            f'upper={upper[i]:.5f}, '
                            f'num_splits={num_splits}')
            ratios = torch.arange(0, num_splits, 1)
            meshgrid.append(ratios)
            box_size.append(size_full / num_splits)
        box_size = torch.tensor(box_size)
        meshgrid = torch.meshgrid(*meshgrid)
        lb = torch.concat(
            [torch.tensor(item.reshape(-1)) for item in meshgrid]
        ).reshape(self.x_dim, -1).t() * box_size + lower
        ub = lb + box_size
        if verbose:
            logger.info(f'Adding {lb.shape[0]} examples to the initial dataset.')
        weight = torch.ones(lb.shape[0], dtype=torch.float32) * weight
        self.append(lb, ub, weight)

    def append(self, lb, ub, weight=None):
        self.dm_lb.append(lb)
        self.dm_ub.append(ub)
        if weight is None:
            weight = self.get_weight(lb, ub)
        self.dm_weight.append(weight)
        self.dm_inside.append(torch.ones_like(weight).to(torch.bool))

    def __getitem__(self, index):
        return (self.dm_lb._storage[index],
                self.dm_ub._storage[index],
                self.dm_weight._storage[index],
                index)

    def __len__(self):
        return self.dm_lb.num_used

    def add_split(self, idx, x_L_right, x_U_left, weight):
        self.splits.append((idx, x_L_right, x_U_left, weight))

    def commit_split(self):
        for idx, x_L_right, x_U_left, weight in self.splits:
            x_U = self.dm_ub._storage[idx]
            self.append(x_L_right, x_U, weight)
            self.dm_ub._storage[idx] = x_U_left
            self.dm_weight._storage[idx] = weight
        self.splits = []

    def save(self, path):
        domains = (
            self.dm_lb._storage[:self.dm_lb.num_used],
            self.dm_ub._storage[:self.dm_ub.num_used],
            self.dm_weight._storage[:self.dm_weight.num_used],
        )
        torch.save(domains, path)

    def clear(self):
        self.dm_lb.pop(self.dm_lb.num_used)
        self.dm_ub.pop(self.dm_ub.num_used)
        self.dm_weight.pop(self.dm_weight.num_used)
        self.splits = []

    def load_checkpoint(self, path):
        self.clear()
        dm_lb, dm_ub, dm_weight = torch.load(path)
        self.dm_lb.append(dm_lb)
        self.dm_ub.append(dm_ub)
        self.dm_weight.append(dm_weight)
        self.dm_inside.append(torch.ones_like(dm_weight).to(torch.bool))
        logger.info(f'{dm_lb.shape[0]} domains loaded from {path}')

    def update_inside(self, idx, inside):
        self.dm_inside._storage[idx.cpu()] = inside.cpu()


def get_data_loader(dataset, batch_size, shuffle=True, num_workers=8, inside_only=False):
    if inside_only:
        indices = dataset.dm_inside._storage[:dataset.dm_inside.num_used].nonzero().view(-1).numpy()
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        logger.info('Subset sampler with %d examples', len(indices))
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            pin_memory=True, num_workers=num_workers)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
            pin_memory=True, num_workers=num_workers)


def prepare_dataset(args):
    if args.box_dim is not None:
        args.lower_limit = [-args.box_size] * args.box_dim
        args.upper_limit = [args.box_size] * args.box_dim
    if args.lower_limit is None:
        args.lower_limit = [-item for item in args.upper_limit]

    dataset = InputDomainDataset(
        lower_limit=args.lower_limit,
        upper_limit=args.upper_limit,
        hole_size=args.hole_size, border_size=args.border_size,
        max_init_size=args.max_init_size,
        init_splits=args.sample_splits,
        use_even_splits=args.use_even_splits,
        hole_mode=args.hole_mode,
        scale_input=args.scale_input,
        adv_size=args.batch_size,
        max_in_hole_dim=args.max_in_hole_dim,
    )
    return dataset
