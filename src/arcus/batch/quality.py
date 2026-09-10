"""Telling the user which of their results to trust.

The hard problem for someone holding 800 machine-made labels isn't
getting the labels, it's knowing which ones are wrong. They can't read
them all; that's why they used a tool.

Two different signals live here and they are not interchangeable:

- **Accuracy against a gold set** the user labelled themselves. This is
  ground truth and it is the only thing here that measures correctness.
- **Agreement between two models.** This is *not* accuracy. Two human
  coders agreeing means something because their mistakes are
  independent; two language models trained on overlapping text have
  correlated mistakes, so they can agree confidently and both be wrong.
  Reported as what it honestly is: a signal about which rows are
  *uncertain*, not evidence that the rest are right.
"""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class GoldSplit:
    """Hand-labelled rows, split so the accuracy figure means something.

    The temptation is to use every hand-labelled row twice: as few-shot
    examples in the prompt and as the set you score against. That
    measures the model on the very answers it was shown, which
    reliably reports a number far better than the truth. Splitting keeps
    the held-out half honest.
    """

    examples: list[tuple[str, str]]
    held_out: list[tuple[str, str]]


def split_gold(labelled: list[tuple[str, str]], seed: int | None = None) -> GoldSplit:
    if len(labelled) < 2:
        # too few to both teach and test with, so spend them on testing:
        # a weak accuracy estimate beats a contaminated one
        return GoldSplit(examples=[], held_out=list(labelled))

    shuffled = list(labelled)
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return GoldSplit(examples=shuffled[:half], held_out=shuffled[half:])


@dataclass(frozen=True)
class Accuracy:
    correct: int
    total: int
    misses: list[tuple[str, str, str]]  # text, expected, got

    @property
    def rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


def score_against_gold(predictions: dict[str, str], gold: list[tuple[str, str]]) -> Accuracy:
    correct = 0
    misses = []
    for text, expected in gold:
        got = predictions.get(text)
        if got is not None and got.strip().lower() == expected.strip().lower():
            correct += 1
        else:
            misses.append((text, expected, got or "(no answer)"))
    return Accuracy(correct=correct, total=len(gold), misses=misses)


@dataclass(frozen=True)
class Agreement:
    agreed: int
    compared: int
    disagreements: list[tuple[int, str, str]]  # row index, primary, second opinion

    @property
    def rate(self) -> float:
        return self.agreed / self.compared if self.compared else 0.0


def compare_opinions(primary: dict[int, str], second: dict[int, str]) -> Agreement:
    """Compares two models' answers on the rows both attempted."""
    shared = sorted(set(primary) & set(second))
    agreed = 0
    disagreements = []
    for index in shared:
        a, b = primary[index], second[index]
        if a.strip().lower() == b.strip().lower():
            agreed += 1
        else:
            disagreements.append((index, a, b))
    return Agreement(agreed=agreed, compared=len(shared), disagreements=disagreements)


def sample_indexes(total: int, fraction: float, seed: int | None = None) -> list[int]:
    """Picks rows to cross-check, at random rather than the first N.

    Real files arrive sorted, by date or group or whatever the export
    was ordered by, so checking the first slice would describe one
    corner of the data and call it the whole thing.
    """
    if total <= 0 or fraction <= 0:
        return []
    count = max(1, round(total * min(fraction, 1.0)))
    return sorted(random.Random(seed).sample(range(total), min(count, total)))
