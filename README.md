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

**Contents:** [Why](#why) · [How it works](#how-it-works) · [Install](#install)
· [Usage](#usage) · [Status](#status) · [Security & privacy](#security--privacy)

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

Optionally, code, math, and long-document questions can route across
ARC's `-thinking-*` reasoning-effort model variants too
(`arcus config set enable_reasoning_variants true`, default off).
Everyday questions stay on the fast base four either way. This is unit
tested but hasn't run against a real ARC key from this environment.
ARC's docs list these as separate catalog ids rather than a parameter
on the base model, the same pattern already confirmed for web search's
legacy-tool-calling variants below, but that's unverified here. Ask a
code or math question after turning it on and confirm it actually
answers before trusting it.

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

For tab completion on the `chat`/`stats`/`eval`/`models`/`config`/
`--random`/`--model`/`--image`/`--doc`/`--web` words, add one of these
to your shell config:

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

# skip the bandit entirely and pin a specific model for this one call
arcus --model GLM-5.3 "explain how binary search works"

# see how it's doing
arcus stats

# compare the routing policy actually run against offline alternatives
arcus eval

# see every model ARC is currently serving, and which ones arcus routes to
arcus models

# hold a multi-turn conversation instead of a single question
arcus chat

# inside chat, --doc/--web/--image/--model all work inline, one
# attachment per turn: "you: --doc paper.pdf summarize this"

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

Quick reference, details for each are below:

| Command | What it does |
| --- | --- |
| `arcus "<question>"` | Ask something, routed through the bandit + quality gate. |
| `arcus --random "<question>"` | Same, but routes randomly instead of using the learned policy. |
| `arcus --model NAME "<question>"` | Skip routing, pin one specific model. |
| `arcus --image PATH "<question>"` | Ask about an image (vision-capable model only). |
| `arcus --doc PATH "<question>"` | Ask about an uploaded document (RAG). |
| `arcus --web "<question>"` | Ask something needing current information (web search). |
| `arcus chat [--save PATH]` | Multi-turn conversation; `--doc`/`--web`/`--image`/`--model` all work inline per turn. |
| `arcus stats` | Local routing performance so far. |
| `arcus eval` | Offline comparison of the routing policy against alternatives. |
| `arcus models` | ARC's live model catalog vs. what arcus routes to. |
| `arcus config [set ...]` | View or change local settings. |
| `arcus --version` | Installed version. |

`arcus chat` opens a REPL that remembers everything said earlier in that
session (resending the growing transcript each turn, since ARC's API has
no session concept of its own) and routes each turn through the same
bandit/quality-gate/logging pipeline as a one-shot `arcus "..."` call.
Type `exit` or press ctrl-d to leave. The conversation only lives for
that one run, closing the terminal loses it, unless you pass `--save
<path>`, which writes the full transcript (not just whatever's still in
the trimmed context window) to a markdown file when you exit.

`--doc PATH`, `--web`, `--image PATH`, and `--model NAME` all work
inline inside `arcus chat` too, typed as part of a turn (`you: --doc
paper.pdf summarize this`), one attachment per turn, the same rules as
below apply. The attachment only applies to that one turn, a later turn
that wants to keep asking about the same document attaches it again.

`arcus --image <path> "question"` attaches an image to a one-shot
question. It always goes to Kimi-K3 rather than through the usual
bandit comparison, confirmed directly against the API to be the only
one of the four models that can actually see an image, GLM-5.3 and
DeepSeek-V4-Flash both reject image content outright and gpt-oss-120b
accepts the request but reports it can't see anything. Skips the
semantic cache entirely too, matching on the question text alone would
risk serving back an answer about a completely different image.

`arcus --doc <path> "question"` and `arcus --web "question"` work the
same way as `--image`, cache skipped, see "Document Q&A and web search"
above for what each actually does. Only one of `--image`, `--doc`, or
`--web` can be used at a time.

`arcus --model NAME "question"` skips the bandit entirely and always
uses that model, checked against ARC's live catalog first. Since
there's only one arm, the quality gate's checks (empty, truncated,
repetitive, refusal) still run and still get reported, there's just no
other model left to fall back to if it fails, that's the point of an
explicit override. Combine with `--web` or `--image` and the name has
to be one of the models valid for that mode.

`arcus config` shows your current settings (the API key masked) and the
path to the config file. `arcus config set bandit_algorithm <algo>`
changes which bandit algorithm arcus uses without hand-editing the TOML
file. `arcus config set enable_reasoning_variants <true|false>` turns
the reasoning-effort routing described above on or off. Re-keying isn't
supported here on purpose, delete the config file and run `arcus` again
to go through setup fresh.

`arcus stats` reads your local SQLite log and prints a `rich`-formatted
table: request count, average reward, average latency, and cost score
per model per mode, plus your cache hit rate and how many attempts the
quality gate has caught and retried. Entirely local, no network call.

`arcus eval` runs the offline policy evaluation described above against
your own logged history and prints the comparison table (IPS and
doubly-robust estimates with 95% confidence intervals for the greedy
policy and each "always use model X" baseline, against what actually
ran). Below 30 logged bandit-mode requests it still prints the table but
flags the numbers as illustrative only, a bootstrap confidence interval
on a handful of rows isn't a reliable comparison yet.

## Status

Everything above is built and working, adapter, context classification,
all three bandit algorithms, the reward function, the quality gate, the
semantic cache, the offline eval / regret code. The CLI covers all of
it: asking directly (with an optional `--model` override), chat with
inline attachments and transcript export, image/doc/web modes, config,
stats, and eval.

Live-tested against a real ARC key: all four models answer correctly
(`tests/adapters/test_arc_adapter_live.py`), and a full `arcus "..."`
run has gone through the real pipeline end to end, classification,
cache miss, routing, an actual ARC call, the quality gate, logging,
caching. Image input, document Q&A, and web search have each gotten a
real run too. Test suite: 273 passing with a key set (269 + 4
live-only), 4 skipped without one.

Exception: reasoning-effort variant routing (`enable_reasoning_variants`)
has only run against a fake adapter so far, which is why it defaults
off. See "Adaptive routing" above.

ARC's models are reasoning models under the hood, they write to a
hidden `reasoning` field before `content`, so a tight `max_tokens`
budget can get eaten up before any real answer shows up. The CLI never
sets `max_tokens` itself, so this doesn't affect normal usage, it only
matters if you're calling the adapter directly with your own tight
budget.

Still open:

- Real logged usage is thin (a handful of manual runs). `arcus eval`
  runs today, it just doesn't have enough data yet, and says so
  instead of faking confidence.
- Reasoning-effort routing needs a live-key run before it's safe to
  default on.

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
