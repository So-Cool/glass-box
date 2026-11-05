"""The dialogue reproduces the original implementation (checked based on answers
to a 39-turn script for every persona stored in ``dialogue_reference.json``)."""
import random

import pytest

from glassbox.dialogue import (DialogueFixes, DialogueManager,
                               resolve_dialogue_fixes)
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


@pytest.mark.parametrize('pid', PERSONAS)
def test_original_dialogue(pid, original_model, dialogue_reference):
    manager = DialogueManager(original_model, implementation='original')
    assert _run(manager, dialogue_reference['script'], pid) == \
        dialogue_reference['runs'][pid]


@pytest.mark.parametrize('pid', PERSONAS)
def test_fixed_dialogue_runs(pid, model, dialogue_reference):
    manager = DialogueManager(model, implementation='fixed')
    for channel in ('chat', 'speech'):
        answers = _run(manager, dialogue_reference['script'], pid, channel)
        assert all(isinstance(a, str) and a for a in answers[:-1])
        assert not any('  ' in a for a in answers)
        assert not any('process you application' in a for a in answers)


def test_channel_dependent_fixes():
    chat = resolve_dialogue_fixes('fixed', channel='chat')
    speech = resolve_dialogue_fixes('fixed', channel='speech')
    assert chat == DialogueFixes.fixed()
    assert not speech.myself_all_features and not speech.integer_ranges
    assert speech.wording and speech.feature_keywords
    forced = resolve_dialogue_fixes('fixed', {'integer_ranges': True},
                                    channel='speech')
    assert forced.integer_ranges
    assert resolve_dialogue_fixes('original') == DialogueFixes.original()
    with pytest.raises(ValueError):
        resolve_dialogue_fixes('fixed', {'no_such_fix': True})


def _applied(model, implementation='fixed', fixes=None, pid='03'):
    manager = DialogueManager(model, implementation=implementation,
                              fixes=fixes)
    manager.process_intent('hello')
    manager.process_intent('apply', {'persona': pid})
    return manager


def test_myself_default(model):
    manager = _applied(model)
    chat = manager.process_intent('myself', channel='chat')
    speech = manager.process_intent('myself', channel='speech')
    assert 'pounds' in chat and 'savings account' in chat
    assert speech == ('Dear Ruth, at the time of application: you have had '
                      'overdraft on your current account.')


def test_exhausted_queue(model):
    manager = _applied(model, fixes={'stop_when_exhausted': True})
    answers = [manager.process_intent('action') for _ in range(30)]
    assert any('no more suggestions' in a for a in answers)
    original = _applied(model, implementation='original')
    answers = [original.process_intent('action') for _ in range(30)]
    assert not any('no more' in a for a in answers)


def test_integer_ranges(model):
    manager = _applied(model, fixes={'integer_ranges': True})
    answer = manager.process_intent('action')
    assert '954.5' not in answer and '955' in answer
    manager = _applied(model, fixes={'integer_ranges': False})
    assert '954.5' in manager.process_intent('action')


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
