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

from glassbox.dialogue import DialogueManager
from glassbox.dialogue.templates import Templates
from glassbox.model import GlassBoxModel

PERSONAS = ['{:02d}'.format(i) for i in range(10)]


@pytest.fixture(scope='module')
def legacy_model():
    return GlassBoxModel(implementation='legacy',
                         fixes={'deterministic_order': True})


def _run(manager, script, pid, channel='chat'):
    names = PERSONAS
    other = names[(names.index(pid) + 1) % len(names)]
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
def test_legacy_dialogue(pid, channel, legacy_model, dialogue_reference):
    manager = DialogueManager(legacy_model, seed=2018)
    assert _run(manager, dialogue_reference['script'], pid, channel) == \
        dialogue_reference['runs'][channel][pid]


@pytest.mark.parametrize('pid', PERSONAS)
def test_fixed_dialogue_runs(pid, model, dialogue_reference):
    for channel in ('chat', 'speech'):
        manager = DialogueManager(model, seed=2018)
        answers = _run(manager, dialogue_reference['script'], pid, channel)
        assert all(isinstance(a, str) and a for a in answers[:-1])
        assert not any('  ' in a for a in answers)
        assert not any('process you application' in a for a in answers)


def test_seed(model):
    def answers(manager):
        manager.process_intent('hello')
        manager.process_intent('apply', {'persona': '03'})
        return [manager.process_intent('why') for _ in range(5)]
    random.seed(1)
    first = answers(DialogueManager(model, seed=7))
    random.seed(2)  # the global random state does not matter
    assert answers(DialogueManager(model, seed=7)) == first
    assert any(answers(DialogueManager(model, seed=s)) != first
               for s in range(8, 12))
    with pytest.raises(ValueError):
        GlassBoxModel(implementation='original')  # now called 'legacy'


def test_corrected_wording(model):
    t = Templates()
    assert 'process your application' in t.APPLICATION_RECEIVED
    assert t.NO_ACTION.endswith('{}.')
    assert model.phrasing['A3']['part1'] == 'had {value}'
    assert 'instalment' in model.phrasing['A8']['name']


def test_single_sources(model):
    from glassbox.data import FEATURE_SUBSET, load_personas, persona_names
    assert set(model.phrasing) == set(model.feature_id) == set(FEATURE_SUBSET)
    assert set(model.actionable) | set(model.protected) <= set(
        model.feature_id)
    assert persona_names() == {i: p.name for i, p in load_personas().items()}
    assert model.persona_names == persona_names()


def _applied(model, pid='03'):
    manager = DialogueManager(model)
    manager.process_intent('hello')
    manager.process_intent('apply', {'persona': pid})
    return manager


@pytest.mark.parametrize('implementation', ['legacy', 'fixed'])
def test_myself_default(implementation):
    manager = _applied(GlassBoxModel(implementation=implementation))
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
    for explainer_model in (model, GlassBoxModel(implementation='legacy')):
        manager = _applied(explainer_model)
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
    for explainer_model in (model, GlassBoxModel(implementation='legacy')):
        manager = _applied(explainer_model)
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
    manager = _applied(GlassBoxModel(implementation='legacy'))
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
    # also with the original explainer (the original dialogue ignored them)
    original = _applied(GlassBoxModel(implementation='legacy'))
    assert 'asked for' not in original.process_intent(
        'action', {'features_drop': ['A5']})
