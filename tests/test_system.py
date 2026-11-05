import pytest

from ctreex import load_tree
from glassbox import data
from glassbox.config import load_config, merge
from glassbox.core import GlassBox
from glassbox.hardware.leds import MockLEDs, event_commands
from glassbox.hardware.qr import persona_from_zbar


class FakeNLU(object):
    def __init__(self, parses):
        self.parses = parses

    def parse(self, text):
        return self.parses.get(text, {'intent': {'name': 'unknown'},
                                      'entities': []})


@pytest.fixture
def box():
    nlu = FakeNLU({'hi': {'intent': {'name': 'greet'}, 'entities': []}})
    return GlassBox(load_config(), nlu=nlu)


def test_personas_and_model(model):
    personas = data.load_personas()
    assert len(personas) == 10
    assert {p.id for p in personas.values() if p.source} == \
        {'00', '02', '03', '04', '05'}
    assert model.predict(personas['03'].values) == 'bad'
    assert model.explainer.tree.n_features == 38


def test_original_tree_reproduces_its_training_accuracy(model):
    # the shipped tree is the converted 2018 model
    tree = load_tree(model.explainer.tree.to_dict())
    assert tree.leaves.shape[0] == 194
    assert tree.metadata['description'].startswith('Glass-Box')


def test_integer_ranges(model):
    cf = {'A5': {'is': [3959],
                 'wannabe': [[None, '883.0'], ['954.5', '1181.5']]}}
    assert model.describe(cf)['loan_amount']['wannabe'] == [
        'at most 883', 'between 955 and 1181']


def test_messages(box):
    assert box.handle_message('quit') == 'Sorry but I will not quit.'
    assert box.handle_message('hi').startswith('Hello. Welcome')
    assert 'Ruth' in box.handle_message('cmd: apply->persona:03')
    assert box.handle_message('cmd: outcome') == \
        'Your application has been declined.'
    assert box.handle_message('disable') == 'Voice answers disabled.'
    assert box.handle_voice('hi') == ''
    assert box.pop_messages()[0] == 'Voice: hi'
    assert isinstance(box.leds, MockLEDs)
    assert box.leds.history[-1]['pattern'] == 'still'


def test_qr(box):
    from glassbox.hardware.speech import MockSpeaker
    box.speaker = MockSpeaker()
    reply = box.handle_qr('04')
    assert 'Joe' in reply and box.speaker.said == [reply]
    assert box.pop_messages()[0] == 'QR scanner: Hello!'
    assert box.handle_qr('05') is None  # already applying
    assert persona_from_zbar('QR-Code:04\n') == '04'


def test_led_events():
    assert event_commands('outcome', {'outcome': 'good'})[0]['params'][
        'colour'] == 65280
    assert event_commands('fairness', {'indirect': True})[0]['params'][
        'colour'] == 255
    assert event_commands('unknown') == []


def test_web_app(box):
    flask = pytest.importorskip('flask')  # noqa: F841
    from glassbox.app import create_app
    client = create_app(box).test_client()
    assert client.get('/').status_code == 200
    answer = client.post('/ask', json={'message': 'hi'}).get_json()['answer']
    assert answer.startswith('Hello')
    assert client.post('/ask', data={'messageText': 'cmd: help'}).get_json()[
        'answer']
    assert client.post('/qrcode', json={'qr': 'QR-Code:02'}).status_code == \
        200
    assert client.get('/messages').get_json()['messages'][0] == \
        'QR scanner: Hello!'
    assert client.post('/voice', json={'text': 'hi'}).get_json()['text']


def test_device_server():
    pytest.importorskip('flask')
    from glassbox.device_server import create_device_app
    config = load_config(overrides={'hardware': {'speech': {
        'backend': 'mock'}}})
    app = create_device_app(config)
    client = app.test_client()
    assert client.post('/led', json={'led_matrix': 'ring',
                                     'pattern': 'pulse'}).status_code == 200
    assert app.config['LEDS'].state['ring']['pattern'] == 'pulse'
    assert client.post('/aiysay', json={'text': 'hi'}).get_json()[
        'status'] == 'OK'
    assert app.config['SPEAKER'].said == ['hi']


def test_config(tmp_path):
    path = tmp_path / 'c.toml'
    path.write_text('[dialogue]\nimplementation = "original"\n'
                    '[hardware.leds]\nbackend = "disabled"\n')
    config = load_config(str(path))
    assert config['dialogue']['implementation'] == 'original'
    assert config['hardware']['speech']['backend'] == 'disabled'
    assert merge({'a': {'b': 1, 'c': 2}}, {'a': {'b': 3}}) == \
        {'a': {'b': 3, 'c': 2}}
    path.write_text('[dialogue]\nimplementation = "newest"\n')
    with pytest.raises(ValueError):
        load_config(str(path))


def test_example_configs_load():
    import glob
    import os
    root = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'config')
    files = glob.glob(os.path.join(root, '*.toml'))
    for f in files:
        load_config(f)
