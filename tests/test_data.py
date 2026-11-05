"""
Data checks

Requires the ``data`` extra (MLaaS) and the German Credit data.
The data are downloaded from UCI when ``GLASSBOX_TEST_DOWNLOAD=1``;
otherwise the tests run only if the data are already cached (``glassbox data``).
"""
import os

import numpy as np
import pytest

mlaas = pytest.importorskip('mlaas.data.tools')

from glassbox import data  # noqa: E402

CACHED = os.path.isfile(os.path.join(mlaas.get_data_home(), 'german.data'))
pytestmark = pytest.mark.skipif(
    not (CACHED or os.environ.get('GLASSBOX_TEST_DOWNLOAD') == '1'),
    reason='German Credit data not available (run "glassbox data").')


@pytest.fixture(scope='module')
def german():
    return data.load_german_credit(verify=False)


def test_processed_data_checksum(german):
    assert data.processed_data_checksum(german.data) == \
        data.PROCESSED_DATA_SHA256


def test_personas_match_data_rows(german):
    """Flags changes of the data set: 5 personas are copies of data rows."""
    matches = data.personas_matching_data(german)
    assert matches == {'00': True, '02': True, '03': True, '04': True,
                       '05': True}


def test_shipped_tree_on_data(german, model):
    columns = model.metadata['columns']
    assert list(german.feature_id_onehot) == columns
    predictions = model.explainer.tree.predict(german.data_onehot)
    assert np.mean(predictions == german.target) == pytest.approx(0.991)
