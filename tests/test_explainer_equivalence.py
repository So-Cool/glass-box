"""CtreeX's legacy implementation reproduces the 2018 Glass-Box explainer."""
import gzip
import json
import os

import numpy as np
import pytest

from ctreex import Feature, FeatureSpec, TreeCounterfactualExplainer
from ctreex.counterfactuals import support_set_metric, volatility_index
from glassbox.model import DEFAULT_TREE

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                    'ctreex')
PERSONAS = ['{:02d}'.format(i) for i in range(10)]


def load_json(name):
    opener = gzip.open if name.endswith('.gz') else open
    with opener(os.path.join(DATA, name), 'rt') as f:
        return json.load(f)


def keys_order(d):
    """The (order-preserving) format of the reference fixtures."""
    if isinstance(d, dict):
        return [[str(k), keys_order(v)] for k, v in d.items()]
    if isinstance(d, (list, tuple)):
        return [keys_order(v) for v in d]
    if isinstance(d, np.integer):
        return int(d)
    return d


def sort_strings(o):
    if isinstance(o, list):
        o = [sort_strings(v) for v in o]
        if o and all(isinstance(v, str) for v in o):
            return sorted(o)
    return o


@pytest.fixture(scope='module')
def personas_reference():
    return load_json('glassbox_personas_reference.json.gz')


@pytest.fixture(scope='module')
def original_explainer(personas_reference):
    fm = personas_reference['feature_map']
    features = [Feature(f, 'numerical' if fm[f]['map'] == 'numerical'
                        else 'categorical', display_name=fm[f]['name'],
                        value_names=fm[f]['map']
                        if isinstance(fm[f]['map'], dict) else {})
                for f in fm if f != 'class']
    spec = FeatureSpec.from_column_names(personas_reference['columns'],
                                         features=features)
    return TreeCounterfactualExplainer(
        DEFAULT_TREE, spec, implementation='legacy',
        class_names=personas_reference['class_names'])


def test_meta_features(original_explainer, personas_reference):
    names = [m.name for m in original_explainer.meta_features]
    assert names == personas_reference['meta_features']


def test_leaf_meta_features(original_explainer, personas_reference):
    ref = personas_reference['leafs_meta_features']
    lmf = original_explainer.leaf_meta_features
    leaves = [i for i in ref['index'] if i >= 0]
    assert lmf.leaf_ids.tolist() == leaves
    for idx, row in zip(ref['index'], ref['values']):
        if idx < 0:
            continue
        assert lmf.row(idx).tolist() == row[:-1]
        assert lmf.leaf_classes[lmf.leaf_index(idx)] == row[-1]


@pytest.mark.parametrize('persona', PERSONAS)
def test_persona_counterfactuals(persona, original_explainer,
                                 personas_reference):
    case = personas_reference['personas'][persona]
    ref = case['reference']
    e = original_explainer.explain_instance(np.array(case['x']),
                                            instance_values=case['values'])
    assert e.leaf == ref['leaf']
    assert e.prediction == ref['class']
    assert e.meta_features.tolist() == ref['meta_instance']
    leaf_ids = original_explainer.leaf_meta_features.leaf_ids.tolist()
    assert {str(l): float(d) for l, d in zip(leaf_ids, e.leaf_distances)} \
        == ref['distances']
    assert keys_order(e.support) == ref['support']
    assert len(e.counterfactuals) == len(ref['by_distance'])
    for mine, theirs in zip(e.counterfactuals, ref['by_distance']):
        assert mine.distance == theirs['distance']
        assert keys_order(mine.per_leaf) == theirs['per_leaf']
        assert keys_order(mine.per_feature) == theirs['per_feature']
        # the original orders categories with list(set(...)) -- random order
        assert sort_strings(keys_order(mine.ranked)) == \
            sort_strings(theirs['transformed'])
        assert sort_strings(keys_order(original_explainer.describe(
            mine.ranked))) == sort_strings(theirs['translated'])


def test_values_decoded_from_one_hot(original_explainer, personas_reference):
    case = personas_reference['personas']['03']
    assert original_explainer.spec.decode(case['x']) == case['values']


def test_leaf_distance_matrix(original_explainer):
    ref = load_json('glassbox_metric_reference.json.gz')
    matrix, ids, classes = original_explainer.leaf_distance_matrix()
    assert ids.tolist() == ref['index']
    assert classes.tolist() == ref['class']
    assert np.array_equal(matrix, np.array(ref['values']))
    for leaf, value in ref['volatility'].items():
        assert volatility_index(int(leaf), matrix, ids, classes) == \
            pytest.approx(value, abs=1e-12)
    for leaf, value in ref['support_metric'].items():
        assert support_set_metric(int(leaf), matrix, ids, classes) == value


def test_toy_example_volatility():
    toy = load_json('small_reference.json')['toy']
    matrix = np.array(toy['values'])
    ids, classes = np.array(toy['index']), np.array(toy['class'])
    for leaf, value in toy['volatility'].items():
        assert volatility_index(int(leaf), matrix, ids, classes) == \
            pytest.approx(value, abs=1e-12)


def test_balloons():
    ref = load_json('small_reference.json')['balloon']
    spec = FeatureSpec.from_column_names(ref['tree']['feature_names'])
    explainer = TreeCounterfactualExplainer(
        ref['tree'], spec, implementation='legacy',
        class_names=['false', 'true'])
    assert [m.name for m in explainer.meta_features] == ref['meta_features']
    lmf = explainer.leaf_meta_features
    assert lmf.leaf_ids.tolist() == ref['leaves']
    for leaf, row in ref['leaf_matrix'].items():
        assert lmf.row(int(leaf)).tolist() == row
    matrix, _, classes = explainer.leaf_distance_matrix()
    assert np.array_equal(matrix, np.array(ref['metric']))
    assert classes.tolist() == ref['class']
