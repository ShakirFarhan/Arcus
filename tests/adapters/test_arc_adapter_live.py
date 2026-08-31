import os

import pytest

from arcus.adapters.arc_adapter import ArcAdapter, ArcModel

# These hit the real ARC endpoint, so they only run if you've actually got
# a key set. No key, no test, rather than failing CI or anyone else's
# machine that doesn't have ARC access.
pytestmark = pytest.mark.skipif(
    not os.environ.get("ARC_API_KEY"),
    reason="set ARC_API_KEY to run live checks against ARC",
)


@pytest.fixture(scope="module")
def adapter():
    return ArcAdapter()


@pytest.mark.parametrize("model", list(ArcModel))
def test_model_responds(adapter, model):
    # all four ARC models are reasoning models under the hood, they spend
    # completion tokens on a hidden "reasoning" field before ever writing
    # to "content". a small max_tokens budget can get fully eaten by that
    # reasoning step, leaving content empty even though the model is
    # working fine. GLM-5.2 in particular used over 700 reasoning tokens
    # on this exact one-word prompt in testing, so this needs real
    # headroom, not just a little more than the old value of 10.
    completion = adapter.chat(
        model,
        [{"role": "user", "content": "Reply with exactly one word: pong"}],
        max_tokens=1024,
    )

    choice = completion.choices[0]
    content = choice.message.content or ""

    assert content.strip(), f"{model} returned an empty response"
    assert choice.finish_reason in ("stop", "length"), (
        f"{model} finished with {choice.finish_reason!r}, likely truncated or errored mid-response"
    )
