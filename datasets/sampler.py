from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import math
import random
import numpy as np


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomCrossModalIdentitySampler(Sampler):
    """
    Randomly sample identities while forcing cross-modal positives when possible.

    For each selected identity, the sampler draws ``num_instances`` images. If an
    identity has both camera/modality 0 and 1, at least one instance from each
    modality is included. This keeps triplet and cross-modal auxiliary losses
    active on small opt/SAR ReID training sets.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if num_instances < 2:
            raise ValueError("RandomCrossModalIdentitySampler requires NUM_INSTANCE >= 2.")

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        self.modality_index_dic = defaultdict(lambda: defaultdict(list))
        for index, (_, pid, camid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            self.modality_index_dic[pid][camid].append(index)

        self.pids = [
            pid
            for pid in self.index_dic.keys()
            if 0 in self.modality_index_dic[pid] and 1 in self.modality_index_dic[pid]
        ]
        if not self.pids:
            raise ValueError(
                "RandomCrossModalIdentitySampler found no IDs with both modality 0 and 1."
            )

        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            num = max(num, self.num_instances)
            self.length += num - num % self.num_instances

    def _sample_pid_batch(self, pid):
        modal_indices = self.modality_index_dic[pid]
        selected = [
            int(np.random.choice(modal_indices[0])),
            int(np.random.choice(modal_indices[1])),
        ]

        if self.num_instances > 2:
            all_indices = self.index_dic[pid]
            extra = np.random.choice(
                all_indices,
                size=self.num_instances - 2,
                replace=len(all_indices) < self.num_instances - 2,
            )
            selected.extend(int(idx) for idx in extra)

        random.shuffle(selected)
        return selected

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            num_batches = max(1, math.ceil(len(self.index_dic[pid]) / self.num_instances))
            for _ in range(num_batches):
                batch_idxs_dict[pid].append(self._sample_pid_batch(pid))

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
