import pytest

from glassbox_testing import load_json
from glassbox.nlu.command import parse_command
from glassbox.nlu.intentioner import (Intentioner, decode_feature_fixed,
                                      decode_feature_original,
                                      decode_persona)
from glassbox.resources import load_json as load_resource


def test_original_intentioner(dialogue_reference):
    intentioner = Intentioner(implementation='original')
    for case in dialogue_reference['intentioner']:
        intent, args = intentioner.parse_to_intent(case['parse'])
        assert [intent, args] == [case['intent'], case['args']], case['text']


def _feature_mentions():
    examples = load_resource('nlu', 'training_data.json')['rasa_nlu_data'][
        'common_examples']
    return [(e['text'][en['start']:en['end']], en['value'])
            for e in examples for en in e.get('entities', [])
            if en['entity'] == 'feature']


def test_feature_keywords():
    mentions = _feature_mentions()
    assert all(decode_feature_fixed(s) == v for s, v in mentions)
    wrong = [s for s, v in mentions if decode_feature_original(s) != v]
    assert len(wrong) == 11 and 'separated' in wrong
    assert decode_feature_fixed('my employment type') == 'A17'
    assert decode_feature_fixed('current employment') == 'A7'


def test_why_reset():
    parse = {'intent': {'name': 'why'},
             'entities': [{'entity': 'why_reset', 'value': 'reset',
                           'start': 0, 'end': 5}]}
    assert Intentioner(implementation='fixed').parse_to_intent(parse) == \
        ('why', {'restart': True})
    assert Intentioner(implementation='original').parse_to_intent(parse) == \
        ('why', {})


def test_decode_persona():
    assert decode_persona('4') == '04'
    assert decode_persona('four') == '04'
    assert decode_persona('marge') == '09'
    assert decode_persona('Bart') == 'Bart'


def test_command():
    assert parse_command('cmd: why->features_require:[A9];'
                         'features_drop:[A13, A2]') == \
        ('why', {'features_require': ['A9'], 'features_drop': ['A13', 'A2']})
    assert parse_command('cmd: apply->persona:04') == \
        ('apply', {'persona': '04'})
    assert parse_command('cmd: why->restart:true') == \
        ('why', {'restart': True})
    assert parse_command('cmd: hello') == ('hello', {})
    with pytest.raises(ValueError):
        parse_command('cmd: why->nonsense')


@pytest.mark.slow
def test_spacy_backend():
    pytest.importorskip('spacy')
    from glassbox.nlu.spacy_backend import SpacyNLU
    try:
        nlu = SpacyNLU.load()
    except OSError:
        pytest.skip('The spaCy model is not installed.')
    intentioner = Intentioner(nlu)
    assert intentioner.sentence_to_intent('Hello')[0] == 'hello'
    assert intentioner.sentence_to_intent('I am 04') == \
        ('apply', {'persona': '04'})
    assert intentioner.sentence_to_intent('Is the decision fair?')[0] == \
        'fairness'


@pytest.mark.slow
def test_held_out_accuracy():
    """The shipped model on sentences not used for training."""
    pytest.importorskip('spacy')
    from glassbox.nlu.spacy_backend import SpacyNLU
    try:
        nlu = SpacyNLU.load()
    except OSError:
        pytest.skip('The spaCy model is not installed.')
    from glassbox.nlu import evaluation as evaluate
    test = load_json('nlu_test_set.json')['rasa_nlu_data']['common_examples']
    scores = evaluate.evaluate(nlu, test)
    assert scores['intent accuracy'] >= 0.75
    assert scores['(intent, arguments) accuracy'] >= 0.65


def test_action_restrictions_parsing():
    parse = {'intent': {'name': 'action'}, 'entities': [
        {'entity': 'why_despite', 'value': 'disregarding', 'start': 0,
         'end': 12},
        {'entity': 'feature', 'value': 'age', 'start': 16, 'end': 19}]}
    assert Intentioner(implementation='fixed').parse_to_intent(parse) == \
        ('action', {'features_drop': ['A13']})
    assert Intentioner(implementation='original').parse_to_intent(parse) == \
        ('action', {})
    with_feature = {'intent': {'name': 'action'}, 'entities': [
        {'entity': 'feature', 'value': 'installment rate', 'start': 20,
         'end': 36}]}
    assert Intentioner().parse_to_intent(with_feature) == \
        ('action', {'features_require': ['A8']})


def test_extra_training_data_mentions():
    from glassbox.resources import load_json as load_resource
    examples = load_resource('nlu', 'training_data_extra.json')[
        'rasa_nlu_data']['common_examples']
    for e in examples:
        for en in e['entities']:
            if en['entity'] == 'feature':
                span = e['text'][en['start']:en['end']]
                assert decode_feature_fixed(span) == en['value'], span


def test_data_generation_reproduces_the_data(tmp_path):
    from glassbox.nlu import data_generation
    from glassbox.resources import load_json as load_resource
    n_train, n_test = data_generation.write_data(
        training_output=str(tmp_path / 'train.json'),
        test_output=str(tmp_path / 'test.json'))
    with open(tmp_path / 'train.json') as f:
        import json
        train = json.load(f)
    assert train == load_resource('nlu', 'training_data_extra.json')
    with open(tmp_path / 'test.json') as f:
        assert json.load(f) == load_json('nlu_test_set.json')
    assert (n_train, n_test) == (241, 49)
    other = data_generation.generate(data_generation.TRAIN, 'train',
                                     repeats=1, seed=1)
    assert 0 < len(other) < n_train
