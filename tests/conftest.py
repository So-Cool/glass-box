import pytest

from glassbox_testing import load_json


@pytest.fixture(scope='session')
def dialogue_reference():
    return load_json('dialogue_reference.json')


@pytest.fixture(scope='session')
def model():
    from glassbox.model import GlassBoxModel
    return GlassBoxModel()
