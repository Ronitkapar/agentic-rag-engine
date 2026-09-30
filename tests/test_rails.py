"""Guardrail rail-detection invariants.

NeMo's "the rail fired" signal is *string matching*, not a structured event:
app/guardrails/rails.py does

    fired = any(indicator in content for indicator in RAIL_INDICATORS)

So the whole safety mechanism silently stops working if someone edits a
`define bot ...` message in colang_rules.py and forgets RAIL_INDICATORS. There
is no exception, no log line, no failed test — the rail just stops firing and
every blocked query sails through. These tests lock that coupling in place.
"""

import re

from app.guardrails.colang_rules import COLANG_CONTENT, RAIL_INDICATORS


def _bot_messages(colang: str) -> list[str]:
    """Every utterance message defined by a `define bot <name>` block."""
    messages = []
    in_bot_block = False
    for raw_line in colang.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("define "):
            in_bot_block = line.startswith("define bot ")
            continue
        if in_bot_block and line.startswith('"'):
            messages.append(line.strip('"'))
    return messages


def _indicator_matches(message: str) -> list[str]:
    """Indicators that would be found in this message by the rails.py predicate."""
    return [i for i in RAIL_INDICATORS if i in message]


def test_colang_defines_at_least_one_bot_message():
    """Guard against a parser change that silently matches nothing."""
    assert _bot_messages(COLANG_CONTENT), "no `define bot` messages parsed"


def test_every_bot_message_has_a_matching_rail_indicator():
    """The core invariant.

    A bot message with no indicator is a rail that will never be detected as
    fired. If you just changed a bot response, add its indicator here.
    """
    unmatched = [
        m for m in _bot_messages(COLANG_CONTENT) if not _indicator_matches(m)
    ]
    assert not unmatched, (
        "These `define bot` messages have no matching RAIL_INDICATORS entry, so "
        "the rail will never be reported as fired:\n"
        + "\n".join(f"  - {m}" for m in unmatched)
    )


def test_rail_indicators_are_not_orphaned():
    """An indicator that matches no bot message is dead config."""
    messages = _bot_messages(COLANG_CONTENT)
    orphans = [i for i in RAIL_INDICATORS if not any(i in m for m in messages)]
    assert not orphans, f"RAIL_INDICATORS entries matching no bot message: {orphans}"


def test_indicators_are_substrings_not_whole_messages():
    """Indicators must stay shorter than the message.

    A full-message indicator still works today but breaks the moment someone
    appends a sentence to the bot response — the exact edit this file exists
    to catch. Substring indicators absorb that.
    """
    messages = _bot_messages(COLANG_CONTENT)
    exact = [i for i in RAIL_INDICATORS if i in messages]
    assert not exact, (
        "These indicators are identical to a full bot message, so editing that "
        f"message will break detection: {exact}"
    )


def test_refusal_and_greeting_rails_are_all_covered():
    """Guard the specific rails that block vs. the ones that merely greet.

    Off-topic and jailbreak refusals are the security-relevant ones; if either
    loses its indicator the assistant starts answering out-of-scope questions
    and honouring "ignore all previous instructions".
    """
    content = COLANG_CONTENT
    for name, expected in (
        ("refuse off topic", "I can't help with that"),
        ("refuse jailbreak", "I maintain consistent guidelines"),
    ):
        block = re.search(
            rf"define bot {re.escape(name)}\n((?:\s+\"[^\"]+\"\n?)+)", content
        )
        assert block, f"no `define bot {name}` block found in COLANG_CONTENT"
        message = block.group(1)
        assert expected in message, f"{name} message lost its anchor text {expected!r}"
        assert _indicator_matches(message), (
            f"the `{name}` rail has no RAIL_INDICATORS entry — it will never fire"
        )
