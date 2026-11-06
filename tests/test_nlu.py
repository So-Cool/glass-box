import pytest

from glassbox_testing import load_json
from glassbox.nlu.command import parse_command
from glassbox.nlu.intentioner import (Intentioner, decode_feature,
                                      decode_persona)
from glassbox.resources import load_json as load_resource


def test_intentioner(dialogue_reference):
    """The original intent mapping with the corrections that are always used
    (15 of the 169 mappings differ from the original code)."""
    intentioner = Intentioner()
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
    assert all(decode_feature(s) == v for s, v in mentions)
    # the original substring rules got these wrong
    assert decode_feature('separated') == 'A9'
    assert decode_feature('married') == 'A9'
    assert decode_feature('my employment type') == 'A17'
    assert decode_feature('current employment') == 'A7'


def test_why_reset():
    parse = {'intent': {'name': 'why'},
             'entities': [{'entity': 'why_reset', 'value': 'reset',
                           'start': 0, 'end': 5}]}
    assert Intentioner().parse_to_intent(parse) == ('why', {'restart': True})


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
    assert Intentioner().parse_to_intent(parse) == \
        ('action', {'features_drop': ['A13']})
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
                assert decode_feature(span) == en['value'], span


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


@pytest.mark.slow
def test_retrained_model_is_cached(tmp_path, monkeypatch):
    """An unusable shipped model is retrained once and then cached."""
    pytest.importorskip('spacy')
    from glassbox.nlu import spacy_backend
    try:
        spacy_backend._load_spacy(spacy_backend.DEFAULT_SPACY_MODEL)
    except OSError:
        pytest.skip('The spaCy model is not installed.')
    monkeypatch.setenv('GLASSBOX_CACHE', str(tmp_path / 'cache'))
    # pretend that the shipped model cannot be loaded
    monkeypatch.setattr(spacy_backend, 'DEFAULT_MODEL_DIR',
                        str(tmp_path / 'missing'))
    calls = []
    train = spacy_backend.SpacyNLU.train.__func__

    def counting_train(cls, *args, **kwargs):
        calls.append(1)
        return train(cls, *args, **kwargs)
    monkeypatch.setattr(spacy_backend.SpacyNLU, 'train',
                        classmethod(counting_train))
    first = spacy_backend.SpacyNLU.load()
    assert calls == [1]
    assert first.training_data_sha256 == \
        spacy_backend.training_data_checksum()
    second = spacy_backend.SpacyNLU.load()  # from the cache
    assert calls == [1]
    assert second.parse('hello')['intent']['name'] == 'greet'
    # a cache trained on other data is not used
    first.training_data_sha256 = 'other'
    first.save(spacy_backend.cache_dir())
    spacy_backend.SpacyNLU.load()
    assert calls == [1, 1]
    with pytest.raises((OSError, ValueError)):
        spacy_backend.SpacyNLU.load(cache=False, retrain_if_needed=False)
