"""
The dialogue in original mode reproduces the original implementation.

``dialogue_reference.json`` holds the answers to a 39-turn script for every
persona and both channels.
The logic of the original dialogue was verified turn by turn against
the original code;
the texts include the corrections that are always used
(wording, whole-number ranges, "no more explanations").
"""
import random

import pytest

from glassbox.dialogue import (DialogueFixes, DialogueManager,
                               resolve_dialogue_fixes)
from glassbox.dialogue.templates import FEATURE_TALK, Templates
from glassbox.model import GlassBoxModel

PERSONAS = ['{:02d}'.format(i) for i in range(10)]


@pytest.fixture(scope='module')
def original_model():
    return GlassBoxModel(implementation='original',
                         fixes={'deterministic_order': True})


def _run(manager, script, pid, channel='chat'):
    names = PERSONAS
    other = names[(names.index(pid) + 1) % len(names)]
    random.seed(2018)
    answers = []
    for intent, args in script:
        a = dict(args)
        if a.get('persona') == 'P':
            a['persona'] = pid
        if a.get('persona') == 'OTHER':
            a['persona'] = other
        answers.append(manager.process_intent(intent, a, channel=channel))
    return answers


@pytest.mark.parametrize('channel', ['chat', 'speech'])
@pytest.mark.parametrize('pid', PERSONAS)
def test_original_dialogue(pid, channel, original_model, dialogue_reference):
    manager = DialogueManager(original_model, implementation='original')
    assert _run(manager, dialogue_reference['script'], pid, channel) == \
        dialogue_reference['runs'][channel][pid]


@pytest.mark.parametrize('pid', PERSONAS)
def test_fixed_dialogue_runs(pid, model, dialogue_reference):
    manager = DialogueManager(model, implementation='fixed')
    for channel in ('chat', 'speech'):
        answers = _run(manager, dialogue_reference['script'], pid, channel)
        assert all(isinstance(a, str) and a for a in answers[:-1])
        assert not any('  ' in a for a in answers)
        assert not any('process you application' in a for a in answers)


def test_fixes():
    assert resolve_dialogue_fixes('fixed') == DialogueFixes.fixed()
    assert resolve_dialogue_fixes('original') == DialogueFixes.original()
    custom = resolve_dialogue_fixes('original', {'why_reset': True})
    assert custom.why_reset and not custom.feature_keywords
    with pytest.raises(ValueError):
        resolve_dialogue_fixes('fixed', {'no_such_fix': True})
    with pytest.raises(ValueError):
        resolve_dialogue_fixes('fixed', {'wording': False})  # not a switch


def test_corrected_wording():
    t = Templates()
    assert 'process your application' in t.APPLICATION_RECEIVED
    assert t.NO_ACTION.endswith('{}.')
    assert FEATURE_TALK['credit_history']['part1'] == 'had {value}'
    assert 'instalment' in FEATURE_TALK['installment_rate']['name']


def _applied(model, implementation='fixed', fixes=None, pid='03'):
    manager = DialogueManager(model, implementation=implementation,
                              fixes=fixes)
    manager.process_intent('hello')
    manager.process_intent('apply', {'persona': pid})
    return manager


@pytest.mark.parametrize('implementation', ['original', 'fixed'])
def test_myself_default(model, implementation):
    manager = _applied(model, implementation=implementation)
    chat = manager.process_intent('myself', channel='chat')
    speech = manager.process_intent('myself', channel='speech')
    assert 'pounds' in chat and 'savings account' in chat and 'age' not in \
        speech
    assert speech == ('Dear Ruth, at the time of application: you have had '
                      'overdraft on your current account.')
    assert manager.process_intent('myself', {'features': ['A13']},
                                  channel='speech') == \
        'Dear Ruth, at the time of application: you have been 29 years old.'
    with pytest.raises(ValueError):
        manager.process_intent('myself', channel='radio')


@pytest.mark.parametrize('channel', ['chat', 'speech'])
def test_exhausted_queue(model, channel):
    for implementation in ('original', 'fixed'):
        manager = _applied(model, implementation=implementation)
        answers = [manager.process_intent('action', channel=channel)
                   for _ in range(30)]
        exhausted = [i for i, a in enumerate(answers)
                     if 'no more suggestions' in a]
        assert exhausted
        # the next request starts from the beginning (same suggestion; the
        # sentence template is chosen at random)
        assert 'at most 883' in answers[0]
        assert 'at most 883' in answers[exhausted[0] + 1]


def test_integer_ranges(model):
    for implementation in ('original', 'fixed'):
        manager = _applied(model, implementation=implementation)
        answer = manager.process_intent('action', channel='speech')
        assert '954.5' not in answer and '955' in answer


def test_events(model):
    events = []
    manager = DialogueManager(model, on_event=lambda e, p: events.append(e))
    manager.process_intent('hello')
    manager.process_intent('apply', {'persona': '04'})
    manager.process_intent('outcome')
    manager.process_intent('why')
    manager.process_intent('bye')
    assert events == ['processing', 'processed', 'outcome', 'why', 'goodbye']


def test_crash_fixes(model):
    manager = _applied(model, implementation='original')
    # applying again as an unknown person crashed the original (NameError)
    answer = manager.process_intent('apply', {'persona': '99'})
    assert 'we do not have this person on record' in answer


def test_action_restrictions(model):
    manager = _applied(model)
    plain = [manager.process_intent('action') for _ in range(3)]
    assert any('asked for' in a for a in plain)  # loan amount suggestions
    dropped = _applied(model)
    answers = [dropped.process_intent('action', {'features_drop': ['A5']})
               for _ in range(6)]
    assert all('asked for' not in a for a in answers)
    assert 'When avoiding using: loan amount' in answers[0]
    required = _applied(model)
    answer = required.process_intent('action',
                                     {'features_require': ['A2']})
    assert 'months' in answer and 'When considering using: loan duration' \
        in answer
    # the original ignored the restrictions
    original = _applied(model, implementation='original')
    assert 'asked for' in original.process_intent('action',
                                                  {'features_drop': ['A5']})
