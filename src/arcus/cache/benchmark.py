from dataclasses import dataclass

from sqlmodel import SQLModel, create_engine

from arcus.cache.semantic_cache import lookup, store

# true paraphrases of stable/conceptual questions, should hit the cache
_PARAPHRASE_PAIRS = [
    ("how does binary search work", "explain the binary search algorithm"),
    ("what is a hash table", "can you explain what a hash map is"),
    ("explain the difference between TCP and UDP", "what's the difference between TCP and UDP protocols"),
    ("what is recursion in programming", "can you explain recursion"),
    ("how does photosynthesis work", "explain the process of photosynthesis"),
    ("what caused World War 1", "what were the causes of the First World War"),
    ("explain the pythagorean theorem", "what is the pythagorean theorem"),
    ("how do neural networks learn", "explain how neural networks are trained"),
    ("what is object oriented programming", "can you explain OOP"),
    ("how does a car engine work", "explain how an internal combustion engine works"),
    ("what is the theory of relativity", "explain einstein's theory of relativity"),
    ("how do vaccines work", "explain how vaccines protect against disease"),
    ("what is dependency injection", "explain the dependency injection pattern"),
    ("how does DNS work", "explain how domain name resolution works"),
    ("what is a black hole", "explain what a black hole is"),
    ("how does compound interest work", "explain compound interest"),
]

# near-duplicate phrasing, different project/assignment number, should NOT hit
_PROJECT_NUMBER_PAIRS = [
    (f"when is project {n} due for CS 3214", f"when is project {n + 1} due for CS 3214") for n in range(1, 9)
]

# near-duplicate phrasing, different software version, should NOT hit
_VERSION_PAIRS = [
    ("what's new in python 3.10", "what's new in python 3.11"),
    ("what's new in python 3.11", "what's new in python 3.12"),
    ("what's new in node 18", "what's new in node 20"),
    ("what's new in java 11", "what's new in java 17"),
    ("what's new in react 17", "what's new in react 18"),
    ("what's new in ubuntu 22.04", "what's new in ubuntu 24.04"),
]

# near-duplicate phrasing, different year, should NOT hit
_YEAR_PAIRS = [
    (f"what major tech events happened in {y}", f"what major tech events happened in {y + 1}")
    for y in range(2018, 2026)
]

# near-duplicate phrasing, different named entity, should NOT hit
_CITY_PAIRS = [
    ("what's the population of Paris", "what's the population of Berlin"),
    ("what's the population of Tokyo", "what's the population of Seoul"),
    ("what's the population of London", "what's the population of Madrid"),
    ("what's the population of Toronto", "what's the population of Vancouver"),
    ("what's the population of Cairo", "what's the population of Nairobi"),
    ("what's the population of Boston", "what's the population of Chicago"),
]
_COMPANY_PAIRS = [
    ("who is the CEO of Google", "who is the CEO of Microsoft"),
    ("who is the CEO of Amazon", "who is the CEO of Apple"),
    ("who is the CEO of Tesla", "who is the CEO of Ford"),
    ("who is the CEO of Netflix", "who is the CEO of Disney"),
    ("who is the CEO of Spotify", "who is the CEO of Samsung"),
    ("who is the CEO of Sony", "who is the CEO of Nintendo"),
]

# near-duplicate phrasing, different quantity, should NOT hit
_QUANTITY_PAIRS = [
    (f"how many calories are in {n} slices of pizza", f"how many calories are in {n + 1} slices of pizza")
    for n in range(1, 7)
]

# clearly unrelated topics, easy negatives
_UNRELATED_PAIRS = [
    ("how does binary search work", "what's a good recipe for banana bread"),
    ("explain the pythagorean theorem", "who won the world cup in 2022"),
    ("how do vaccines work", "what's the best way to learn guitar"),
    ("what is a black hole", "how do I fix a flat tire"),
    ("how does DNS work", "what's the capital of Australia"),
    ("what is recursion in programming", "how do I make cold brew coffee"),
]

BENCHMARK_PAIRS: list[tuple[str, str, bool]] = (
    [(a, b, True) for a, b in _PARAPHRASE_PAIRS]
    + [
        (a, b, False)
        for a, b in (
            _PROJECT_NUMBER_PAIRS
            + _VERSION_PAIRS
            + _YEAR_PAIRS
            + _CITY_PAIRS
            + _COMPANY_PAIRS
            + _QUANTITY_PAIRS
            + _UNRELATED_PAIRS
        )
    ]
)


@dataclass(frozen=True)
class BenchmarkStats:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float


def _fresh_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def evaluate_pairs(
    pairs: list[tuple[str, str, bool]],
    use_param_diff: bool,
    similarity_threshold: float = 0.80,
) -> BenchmarkStats:
    tp = fp = fn = tn = 0

    for query_a, query_b, should_match in pairs:
        # fresh engine per pair, no cross-pair contamination from earlier
        # entries in the loop affecting a later pair's nearest match
        engine = _fresh_engine()
        store(query_a, response=f"answer to: {query_a}", model="gpt-oss-120b", engine=engine)

        result = lookup(query_b, engine=engine, similarity_threshold=similarity_threshold, use_param_diff=use_param_diff)
        predicted_match = result.hit

        if predicted_match and should_match:
            tp += 1
        elif predicted_match and not should_match:
            fp += 1
        elif not predicted_match and should_match:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    return BenchmarkStats(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall)


def run_benchmark(pairs: list[tuple[str, str, bool]] = BENCHMARK_PAIRS) -> dict[str, BenchmarkStats]:
    return {
        "naive_cosine": evaluate_pairs(pairs, use_param_diff=False),
        "with_param_diff": evaluate_pairs(pairs, use_param_diff=True),
    }


if __name__ == "__main__":
    results = run_benchmark()
    print(f"benchmark set: {len(BENCHMARK_PAIRS)} pairs\n")
    for name, stats in results.items():
        print(f"{name}:")
        print(f"  tp={stats.tp} fp={stats.fp} fn={stats.fn} tn={stats.tn}")
        print(f"  precision={stats.precision:.3f} recall={stats.recall:.3f}\n")
