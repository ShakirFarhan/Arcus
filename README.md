# Arcus

[![CI](https://github.com/ShakirFarhan/Arcus/actions/workflows/ci.yml/badge.svg)](https://github.com/ShakirFarhan/Arcus/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/arcus-cli.svg)](https://pypi.org/project/arcus-cli/)
[![Python](https://img.shields.io/pypi/pyversions/arcus-cli.svg)](https://pypi.org/project/arcus-cli/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

A CLI that sits on top of Virginia Tech ARC's LLM API and makes it
smarter: it picks which of ARC's four open-weight models to route a
request to, checks the response before handing it back to you, and
caches answers to questions it's already seen. Runs entirely on your own
machine with your own ARC key. Nothing goes through a shared server.

```bash
arcus "explain how binary search works"
```

## Why

ARC gives every VT user free access to four open-weight models
(gpt-oss-120b, GLM-5.3, Kimi-K3, DeepSeek-V4-Flash) through one
OpenAI-compatible endpoint ([ARC's own docs](https://www.docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html)
cover the service itself, including its rate limits and data-handling
approval, arcus is a client built on top of it, not affiliated with
ARC). Picking a model by hand every time is
tedious, and a plain HTTP 200 doesn't tell you whether the response
inside it was actually any good, a truncated answer or a flat refusal
comes back looking the same as a correct one unless something reads the
content. Arcus adds three things on top of the raw API:

- **Adaptive routing** — a multi-armed bandit learns, per kind of
  question, which model tends to give the best result for the least
  latency and cost.
- **A quality gate** — validates every response (truncation, empty
  output, repetition loops, refusal phrases, schema conformance) before
  it reaches you, and silently retries with a different model if the
  first one produced garbage.
- **A correctness-aware cache** — skips the API call entirely for
  questions it's answered before, but only when it's actually confident
  the new question means the same thing as the cached one.

## How it works

```
your question (CLI arg or piped stdin)
        |
        v
context classifier -- code / reasoning-math / writing / long-document / general
        |
        v
semantic cache check -- hit? return the cached answer, skip everything below
        | miss
        v
bandit router -- picks a model, one bandit instance per (task type, length) bucket
        |
        v
ARC API call (your own key, OpenAI-compatible endpoint)
        |
        v
quality gate -- validates the response, retries with a different model on failure
        | pass
        v
answer to you + write to cache + reward logged back to the bandit
```

Everything after "your question" runs locally. The only network call
this tool ever makes is to ARC, with your own key.

### Context classification

Regex/keyword rules catch the obvious cases fast (a traceback is
obviously a code question, "write me a poem" is obviously a writing
request). Anything that doesn't match falls back to comparing the
prompt's embedding against a small set of labeled anchor examples per
category, so phrasing the regex rules never thought of still lands in
the right bucket instead of defaulting to "general." See
`src/arcus/routing/context.py`.

### Adaptive routing

Three interchangeable bandit algorithms, picked via config
(`bandit_algorithm` in `~/.config/arcus/config.toml`, default
`thompson`):

- **Epsilon-greedy** — simplest baseline, explores randomly a fixed
  fraction of the time.
- **UCB1** — no tunable knob, explores under-tried arms automatically
  via a confidence bound.
- **Thompson sampling** — Bayesian, samples from each arm's learned
  `Beta` distribution, the default because it adapts fastest early on.

A random-selection baseline (`--random`) is also wired in as an A/B
comparison point, mostly useful for the offline evaluation report below.
The reward each arm is updated with is a weighted mix of quality (from
the gate below), normalized latency, and a simulated cost signal built
from real published hosting rates for these same open-weight models
(ARC itself is free, this exists to demonstrate cost-aware routing as a
practice). See `src/arcus/routing/bandit.py` and
`src/arcus/routing/reward.py`.

Since every `arcus` invocation is a fresh process, there's no daemon
holding the bandit's learned state in memory between runs. Instead,
`src/arcus/routing/warm_start.py` rebuilds it at the start of every call
by replaying the local request log, which works because a bandit's
`update()` is just an associative accumulation of pull counts and reward
sums.

The four model ids arcus routes to live in `ArcModel`, but ARC runs its
own model catalog independently and can rename or retire an entry at
any time. `src/arcus/routing/model_catalog.py` checks the configured
list against what ARC is actually serving (cached for a few hours so
this doesn't cost a network round trip on every call) and quietly drops
anything that's no longer live, rather than routing to a model
guaranteed to fail. Local history logged under a since-renamed model id
is skipped the same way when the bandit's state gets rebuilt.

### Quality gate

Five independent checks run over every response: truncation
(`finish_reason == "length"`), empty output, repetition (trigram
duplication ratio), refusal-phrase matching, and optional Pydantic
schema validation for structured-output requests. Any failure logs a
negative reward for that model in that context and retries with a
different one, up to once per available arm, before giving up and
returning the last attempt. See `src/arcus/quality/gate.py`.

ARC caps concurrent requests per account rather than per model, so a
429 doesn't mean the model that was just called is bad, switching to a
different arm wouldn't help either. A rate limit gets a few short
retries against the same model before it's treated as a real failure,
so one busy moment doesn't unfairly tank that model's learned reward.

Similarly, ARC's access restriction (see Install below) applies to the
whole account, not one model, so hitting it stops the request
immediately with a clear message instead of cycling through every arm
against the same wall, and doesn't count against any model's reward.

### Semantic cache

Local `sentence-transformers` embeddings (`all-MiniLM-L6-v2`), cosine
similarity lookup against everything stored so far. Two things keep it
from just being a naive "similar enough, ship it" cache:

- **Volatility classification** — a query containing words like
  "today," "current," or "latest" gets a TTL of zero (never actually
  served stale), stable conceptual questions get a week.
- **Parameter-diff check** — before trusting a high-similarity match,
  numbers and capitalized entities extracted from both queries are
  compared. "when is project 2 due" and "when is project 3 due" read as
  almost identical to a cosine similarity score, this check catches
  that they're different questions.

Measured against a 62-pair labeled benchmark of true paraphrases and
near-duplicate-but-different prompts (`src/arcus/cache/benchmark.py`):

| approach                  | precision | recall |
| -------------------------- | --------- | ------ |
| naive cosine similarity    | 0.306     | 0.688  |
| + parameter-diff check     | 1.000     | 0.625  |

The param-diff check trades some recall (it rejects a few pairs it
shouldn't, "World War 1" vs "the First World War" gets flagged as a
conflicting parameter, a known and documented limitation) for a real
jump in precision, going from roughly 1-in-3 cache hits being wrong to
zero false hits in this benchmark.

### Offline policy evaluation and regret benchmarking

Every request logs the propensity (the probability the routing policy
assigned to whichever model it picked), which makes it possible to
estimate how a *different* policy would have performed without ever
running it live, using only the log that already exists. `arcus` logs
propensity from the very first request, this can't be added
retroactively to old data.

`src/arcus/eval/offline.py` implements inverse propensity scoring (IPS)
and doubly robust (DR) estimators plus percentile bootstrap confidence
intervals, and `evaluate_policies()` produces a comparison table:
the logged policy's actual average reward next to estimated values for
any alternative policies you want to compare it against (e.g. "what if
we'd always used the cheapest model").

Regret benchmarking is a different technique: it needs a *known*
ground-truth reward per arm to measure regret against, which real
traffic can't provide (a real request only ever explores one model per
round, so there's no way to know what the other three would have
scored). `src/arcus/eval/regret.py` simulates each algorithm against a
labeled synthetic reward environment instead, this is the standard way
to study a bandit algorithm's exploration behavior on its own, separate
from real-world model quality. A sample run (2000 rounds, seed 42):

| algorithm      | final cumulative regret |
| -------------- | ------------------------ |
| epsilon-greedy  | 7.9                       |
| thompson        | 25.4                      |
| ucb1            | 58.6                      |
| random          | 76.9                      |

All three real algorithms land well below the random baseline, which is
the actual point: they're spending far less time on worse-than-best
arms than picking blindly would.

### Document Q&A and web search

Both build directly on capabilities ARC's own API already provides,
rather than reimplementing them:

- **`arcus --doc <path> "question"`** uploads the file to ARC's RAG
  endpoint, attaches it to the request, and deletes it from your ARC
  account again once you have an answer. Works across all four core
  models, confirmed live against the real API.
- **`arcus --web "question"`** routes to ARC's `server:websearch` tool
  through three of its "legacy-tool-calling" model variants
  (`gpt-oss-120b`, `Kimi-K3`, and the older `glm-52` variant) confirmed
  to actually perform a real search and cite sources. A fourth,
  DeepSeek's legacy variant, accepts the same request without erroring
  but doesn't reliably act on it, live testing caught it answering a
  time-sensitive question wrong with no citation, so it's left out.

Both skip the semantic cache: a cached answer keyed on question text
alone would risk answering about the wrong document, or serving a
web-search answer that's since gone stale. See `src/arcus/cli.py`
(`run_doc_ask`, `run_web_ask`) and `ArcAdapter.upload_file`/
`delete_file` in `src/arcus/adapters/arc_adapter.py`.

## Install

```bash
pip install arcus-cli
# or, with uv
uv tool install arcus-cli
```

Or run from source:

```bash
git clone https://github.com/ShakirFarhan/Arcus.git
cd Arcus
uv sync
uv run arcus "explain how binary search works"
```

First run walks you through a one-time setup: it asks for your ARC key
(get one from `llm.arc.vt.edu` under User profile > Settings > Account
> API keys), makes one live call to check it works, and saves it to
`~/.config/arcus/config.toml` with `chmod 600`. No separate setup
command to remember.

ARC restricts the API to VT's campus network, so this (and every
`arcus` call after it) needs either an on-campus connection or VT's
VPN. Arcus surfaces this as a clear message rather than the generic
"no usable response" error when it happens.

For tab completion on the `chat`/`stats`/`models`/`config`/`--random`/
`--image`/`--doc`/`--web` words, add one of these to your shell config:

```bash
# zsh, in ~/.zshrc
eval "$(arcus --completion zsh)"

# bash, in ~/.bashrc
eval "$(arcus --completion bash)"
```

## Usage

```bash
# ask something directly
arcus "explain how binary search works"

# pipe an error straight in
python broken.py 2>&1 | arcus

# or combine piped context with an explicit instruction
python broken.py 2>&1 | arcus "why is this failing"

# force the random-routing baseline instead of the learned bandit policy
arcus --random "explain how binary search works"

# see how it's doing
arcus stats

# see every model ARC is currently serving, and which ones arcus routes to
arcus models

# hold a multi-turn conversation instead of a single question
arcus chat

# save the conversation to a file when you leave
arcus chat --save transcript.md

# ask about an image (routes to Kimi-K3, the one ARC model documented
# as vision-capable)
arcus --image screenshot.png "what's wrong with this code?"

# ask a question about a document, ARC handles the retrieval
arcus --doc syllabus.pdf "when is the midterm?"

# ask something that needs current information
arcus --web "what's the latest release of Python?"

# view or change local settings
arcus config
arcus config set bandit_algorithm ucb1

# check which version is installed
arcus --version
```

`arcus chat` opens a REPL that remembers everything said earlier in that
session (resending the growing transcript each turn, since ARC's API has
no session concept of its own) and routes each turn through the same
bandit/quality-gate/logging pipeline as a one-shot `arcus "..."` call.
Type `exit` or press ctrl-d to leave. The conversation only lives for
that one run, closing the terminal loses it, unless you pass `--save
<path>`, which writes the full transcript (not just whatever's still in
the trimmed context window) to a markdown file when you exit.

`arcus --image <path> "question"` attaches an image to a one-shot
question. It always goes to Kimi-K3 rather than through the usual
bandit comparison, confirmed directly against the API to be the only
one of the four models that can actually see an image, GLM-5.3 and
DeepSeek-V4-Flash both reject image content outright and gpt-oss-120b
accepts the request but reports it can't see anything. Skips the
semantic cache entirely too, matching on the question text alone would
risk serving back an answer about a completely different image. Not
available in `arcus chat` yet, one-shot only.

`arcus --doc <path> "question"` and `arcus --web "question"` work the
same way as `--image`, one-shot only, cache skipped, see "Document Q&A
and web search" above for what each actually does. Only one of
`--image`, `--doc`, or `--web` can be used at a time.

`arcus config` shows your current settings (the API key masked) and the
path to the config file. `arcus config set bandit_algorithm <algo>`
changes which bandit algorithm arcus uses without hand-editing the TOML
file. Re-keying isn't supported here on purpose, delete the config file
and run `arcus` again to go through setup fresh.

`arcus stats` reads your local SQLite log and prints a `rich`-formatted
table: request count, average reward, average latency, and cost score
per model per mode, plus your cache hit rate and how many attempts the
quality gate has caught and retried. Entirely local, no network call.

## Status

Everything described above is implemented and working: the ARC adapter,
context classification, all three bandit algorithms with propensity
tracking, the reward function, the quality gate (including rate-limit
backoff and VPN-restriction handling), the semantic cache and its
benchmark, the CLI (ask command, chat mode with transcript export,
image input, document Q&A, web search, a config command, first-run
wizard, error piping, stats, models), and the offline evaluation +
regret benchmarking layer.

Verified live against a real ARC key: all four models respond correctly
(`tests/adapters/test_arc_adapter_live.py`), a full end-to-end
`arcus "..."` run exercises the whole pipeline (context classification,
cache miss, bandit routing, a real ARC call, the quality gate, logging,
and caching the result) against real traffic, and image input, document
Q&A, and web search have each been run against real responses too, not
just unit tested. Test suite: 238 passing with a key set (234 plus 4
live-only tests), 4 skipped without one.

Worth knowing: ARC's models are reasoning models under the hood, they
write to a hidden `reasoning` field before `content`, so a small
`max_tokens` budget can get entirely spent on reasoning before any real
answer comes out. The CLI itself never sets `max_tokens`, so normal
usage isn't affected, ARC's server-side default leaves plenty of room,
this only matters if you're calling the adapter directly with a tight
budget of your own.

What's still open:

- Real logged usage is still thin (a handful of manual runs). Once
  there's a real query history, `arcus/eval/offline.py`'s
  `evaluate_policies()` is what turns it into the comparison table
  described above.
- Image input, document Q&A, and web search are all one-shot only,
  `arcus chat` doesn't support attaching an image, a document, or
  enabling search mid-conversation.

## Security & privacy

- Each install uses its own user's ARC key. Keys are never shared,
  bundled, or sent anywhere but ARC's own endpoint.
- No data leaves your machine except to ARC itself, with your own key.
  Request logs, cache entries, and stats are all local SQLite, nothing
  is aggregated or reported anywhere else.
- `arcus --doc` uploads the whole file to your ARC account temporarily
  (deleted again once you have an answer), and `arcus --web` sends your
  question through ARC's own web search tool. Both stay within ARC,
  same as every other request, but a document leaving your machine
  entirely (even briefly, even to your own account) is worth knowing
  about explicitly.
- This tool hasn't been through ARC's security review for regulated
  data (FERPA records, health data, etc.) the way ARC's own web
  interface has. Don't route sensitive regulated data through it.
- MIT licensed, source is fully readable, that's the actual trust
  mechanism here rather than a policy document.

## License

MIT, see `LICENSE`.
