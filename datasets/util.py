from itertools import cycle, islice

from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from typing import Sequence


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


class RoundrobinSampler(Sampler[int]):
    indices: Sequence[int]

    def __init__(self, indices: Sequence[Sequence[int]], generator=None) -> None:
        self.indices = indices
        self.generator = generator

        self.n_classes = len(indices)
        self._current_idx = 0

        self.subsamplers = [SubsetRandomSampler(i) for i in indices]

    def __iter__(self):
        "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
        # Recipe credited to George Sakkis
        iterables = self.subsamplers
        pending = len(iterables)
        nexts = cycle(iter(it).__next__ for it in iterables)
        while pending:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                pending -= 1
                nexts = cycle(islice(nexts, pending))

    def __len__(self) -> int:
        return sum([len(i) for i in self.indices])
