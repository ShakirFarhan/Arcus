# Arcus

[![CI](https://github.com/ShakirFarhan/Arcus/actions/workflows/ci.yml/badge.svg)](https://github.com/ShakirFarhan/Arcus/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/arcus-cli.svg)](https://pypi.org/project/arcus-cli/)
[![Python](https://img.shields.io/pypi/pyversions/arcus-cli.svg)](https://pypi.org/project/arcus-cli/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

VT gives every student, researcher and staff member free access to four
large language models through [ARC](https://llm.arc.vt.edu). Arcus is a
command line client for it.

```bash
arcus "explain how binary search works"
```

It picks a model for you, checks the answer before showing it to you,
and can run one instruction over a whole spreadsheet. Everything stays
on your machine and talks only to ARC, with your own key.

[Install](#install) · [Commands](#commands) · [Batch](#batch) ·
[How it picks a model](#how-it-picks-a-model) · [Privacy](#privacy) ·
[Status](#status)

## Install

```bash
pip install arcus-cli
```

First run asks for your ARC key. Get one at
[llm.arc.vt.edu](https://llm.arc.vt.edu) under **User profile →
Settings → Account → API keys**. It's saved to
`~/.config/arcus/config.toml` with `chmod 600`.

**You need to be on campus or on the VPN.** ARC restricts its API to
VT's network. Arcus says so when that's the problem rather than leaving
you guessing.

Optional, and it adds about 750MB because it pulls in torch:

```bash
pip install 'arcus-cli[cache]'
```

That turns on the semantic cache, which skips the API call for questions
you've already asked. Without it everything works the same, you just pay
for repeat questions.

Tab completion:

```bash
eval "$(arcus --completion zsh)"    # or bash
```

## Commands

```bash
# ask something
arcus "why does my Django migration keep failing"

# pipe an error straight in
python broken.py 2>&1 | arcus

# or pipe it with an instruction on top
python broken.py 2>&1 | arcus "explain this like I'm new to async"

# a conversation instead of one question
arcus chat
arcus chat --save notes.md

# ask about a file, an image, or something current
arcus --doc syllabus.pdf "when is the midterm"
arcus --image screenshot.png "what's wrong with this code"
arcus --web "what's the newest stable Python release"

# pin a model instead of letting it choose
arcus --model GLM-5.3 "explain covariance"

# label an entire spreadsheet
arcus batch survey.csv "classify the sentiment as positive, negative, or neutral"
```

Inside `arcus chat` the flags work per turn:

```
you: --doc paper.pdf what's the main claim here
you: --web who won the game last night
```

Housekeeping:

| | |
|---|---|
| `arcus stats` | what each model has been doing for you |
| `arcus models` | what ARC is serving right now |
| `arcus config` | show or change settings |
| `arcus eval` | compare the routing against just picking one model |
| `arcus judge` | grade any answers still waiting to be scored |

## Batch

This is the part a browser can't do. Point it at a file, give it one
instruction, get a column back.

```bash
arcus batch survey.csv "classify the sentiment as positive, negative, or neutral"
```

```
  survey.csv — 812 rows, 3 columns
  reading column: response          (change with --column)
  answers must be one of: positive, negative, neutral

  trying 5 rows first…

    row 1     The course was well organized but the pace was brutal
               → positive

    row 3     Honestly I struggled the entire semester
               → negative

  model: gpt-oss-120b
  estimate: 2m to 7m for the remaining 807 rows

  continue? [Y/n]
```

You get `survey.labeled.csv`, everything preserved with a `result`
column added. The same shape works for extraction ("pull out the sample
size and the method"), triage ("bug, feature request, or question"), or
anything else that's one instruction repeated a lot.

CSV and JSONL both work. Excel exports with their hidden byte-order mark
work. Cells containing commas, quotes and line breaks come back intact.

Things worth knowing:

**It shows you five rows before doing the other 807.** The expensive
mistake is a prompt that was subtly wrong, and five requests is a
cheaper way to find that out than eight hundred.

**You can kill it.** Rows are written as they finish. Ctrl-C at row
4,000 of 10,000, run the same command again, and it carries on from
4,000.

**One model does the whole file.** The router that runs everywhere else
is switched off here, because a dataset labelled half by one model and
half by another has a problem baked into it that no analysis afterwards
will fix. A row the model can't answer goes to
`survey.labeled.failed.csv` rather than quietly to a different model.

**Answers get checked against the labels you named.** Models reply
`Positive.` and `Sentiment: positive` and `**positive**` however you ask
them not to, and at five thousand rows nobody notices. Anything that
can't be matched is flagged instead of guessed at.

**Afterwards a second model looks at a sample:**

```
  cross-checked 160 rows against GLM-5.3: 94% matched
  9 disagreed → survey.labeled.review.csv
```

Read that as "here are the ambiguous rows", not as an accuracy score.
Two models agreeing doesn't mean much, they've read a lot of the same
text and tend to be wrong about the same things. Two models disagreeing
does mean something, and those nine rows are worth your own eyes.
`--no-cross-check` turns it off.

You also get `survey.labeled.manifest.json` recording which model ran,
under what instruction, over which column. If you publish anything based
on machine-labelled data, someone will ask.

Overrides, all optional: `--column`, `--out`, `--choices`, `--model`,
`--concurrency`, `--limit`, `--yes`, `--no-cross-check`.

## How it picks a model

ARC serves four models — gpt-oss-120b, GLM-5.3, Kimi-K3 and
DeepSeek-V4-Flash — and choosing between them by hand every time gets
old. Arcus sorts your question into a category, then uses a multi-armed
bandit to learn which model does best in that category, scoring answers
on quality and speed.

Two things make that more useful than it sounds.

**It reads the answer before you do.** A 200 from the API says nothing
about whether the thing inside it is any good. Arcus checks for
truncation, empty responses, repetition loops and refusals, and quietly
retries on another model when it finds one. A sample of the answers that
pass then goes to a second model for a 0–10 grade, so what the router
learns from is how good the answer was, not just that it arrived. That
grading happens in the background on your next command, never while
you're waiting.

**It knows what fits.** Three of the four models hold 128k tokens and
DeepSeek holds 512k. Pipe a big log file in and it goes straight to the
one that can take it, rather than failing three times first.

Smaller things it deals with: ARC allows ten requests at a time per
account and signals that in a way most clients misread, so Arcus backs
off and retries instead of blaming the model. ARC renames models
occasionally, so the model list is checked against what's actually being
served. And a question you've asked before comes back from the local
cache with no network call, but only when it's confident it really is
the same question — "when is project 2 due" and "when is project 3 due"
look nearly identical to a similarity score and are not the same
question.

For images, Kimi-K3 and DeepSeek-V4-Flash can read one, GLM-5.3 refuses
cleanly, and gpt-oss-120b will confidently describe an image it cannot
see. `--image` only goes to the two that work.

If you want the details: `src/arcus/routing/` for the bandit and model
catalog, `src/arcus/quality/` for the checks and the grader,
`src/arcus/batch/` for batch.

## Privacy

- Your key, your machine. Requests go to ARC and nowhere else.
- Logs, cache and batch state are local SQLite. Nothing is uploaded,
  aggregated or shared.
- `--doc` uploads your file to your own ARC account and deletes it once
  you have an answer. `--web` sends your question through ARC's search
  tool. Both stay inside ARC, but a document leaving your laptop at all
  is worth knowing about.
- Batch keeps your rows locally while a job runs so it can resume.
  They're yours to delete.
- Arcus has not been through ARC's review for regulated data. ARC itself
  has, this client hasn't. Don't put FERPA records or export-controlled
  material through it.

## Status

Working and in use: the ARC adapter, the quality checks, the grader, the
semantic cache, batch, and the CLI around all of it. 457 tests, plus a
few that only run when a real key is present.

Tested against the real API rather than only mocked: all four models
answer, a full question runs end to end, and batch has been run over a
file seeded with the kinds of rows that usually break CSV handling.

What isn't proven:

- **The routing works, but nobody has shown it beats just picking one
  model.** The machinery to find out is built — `arcus eval` runs
  inverse-propensity and doubly-robust estimates with confidence
  intervals — it needs more logged history than exists yet. Until then,
  treat adaptive routing as a reasonable idea being measured, not a
  proven win.
- **Reasoning-effort routing is off by default.** ARC's docs list
  `reasoning_effort` for all four models; in practice gpt-oss-120b and
  GLM-5.3 respond to it and the other two ignore it. Turn it on with
  `arcus config set enable_reasoning_variants true` if you want to
  experiment.
- ARC's models write hidden reasoning before their visible answer, so a
  small `max_tokens` can be spent entirely on thinking. Arcus never sets
  it, so this only bites if you use the adapter directly.

`src/arcus/eval/regret.py` simulates the bandit algorithms against
invented reward distributions. That's a standard way to study how an
algorithm explores, and it says nothing about ARC or these models, so
its numbers aren't reproduced here.

## License

MIT. Not affiliated with or endorsed by Virginia Tech ARC — it's a
client built on a service they run. Their own documentation for the
service is
[here](https://www.docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html).
