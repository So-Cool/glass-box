"""
The Glass-Box model: the decision tree, its features and the explainer.

The shipped model is the decision tree used in the IJCAI 2018 demo, converted
from its scikit-learn 0.19 pickle to JSON (see ``tools/convert_original_model``
in the repository). A tree can also be loaded from any JSON file written by
:meth:`ctreex.TreeStructure.to_json` or a (modern) pickle of a scikit-learn
decision tree trained on the same one-hot encoded features.
"""
import numpy as np

from ctreex import Feature, FeatureSpec, TreeCounterfactualExplainer
from glassbox.data import load_feature_metadata, load_personas
from glassbox.resources import resource_path

__all__ = ['GlassBoxModel', 'DEFAULT_TREE', 'EXPLAINER_IMPLEMENTATION',
           'train_tree']

DEFAULT_TREE = resource_path('glassbox_german_credit_tree.json')
#: glassbox implementation -> ctreex implementation
EXPLAINER_IMPLEMENTATION = {'original': 'legacy', 'fixed': 'fixed'}


def _feature_spec(metadata):
    fm = metadata['feature_map']
    features = []
    for f in metadata['feature_id']:
        numerical = fm[f]['map'] == 'numerical'
        features.append(Feature(
            f, 'numerical' if numerical else 'categorical',
            display_name=fm[f]['name'],
            value_names={} if numerical else dict(fm[f]['map']),
            integer=numerical and fm[f].get('dtype') == 'integer'))
    return FeatureSpec.from_column_names(metadata['columns'],
                                         features=features)


class GlassBoxModel(object):
    """
    The decision tree with everything needed to explain its predictions.

    Parameters
    ----------
    tree : str or object, optional
        A tree accepted by :func:`ctreex.load_tree` (default: the original
        Glass-Box tree).
    implementation : {'original', 'fixed'}
        ``'original'`` uses the legacy implementation of
        :class:`ctreex.TreeCounterfactualExplainer` (the 2018 explainer);
        ``'fixed'`` enables the explainer's fixes.
    fixes : ctreex.Fixes or dict, optional
        Per-fix overrides for the explainer (see :mod:`ctreex.fixes`).
    """

    def __init__(self, tree=None, implementation='fixed', fixes=None):
        self.metadata = load_feature_metadata()
        self.feature_map = self.metadata['feature_map']
        self.feature_id = self.metadata['feature_id']
        self.class_id = self.metadata['class_id']
        self.spec = _feature_spec(self.metadata)
        self.personas = load_personas()
        self.explainer = TreeCounterfactualExplainer(
            DEFAULT_TREE if tree is None else tree, self.spec,
            implementation=EXPLAINER_IMPLEMENTATION[implementation],
            fixes=fixes,
            class_names=self.feature_map[self.class_id]['map'])

    def feature_name(self, feature_id):
        """The (snake_case) name of a feature, e.g. ``loan_amount``."""
        return self.feature_map[feature_id]['name']

    def is_integer(self, feature_id):
        return self.feature_map[feature_id].get('dtype') == 'integer'

    def encode(self, values):
        """One-hot encodes base feature values (like MLaaS)."""
        return self.spec.encode(values)

    def explain(self, values):
        """Counterfactual explanation of an applicant (feature codes)."""
        return self.explainer.explain_instance(self.encode(values),
                                               instance_values=dict(values))

    def predict(self, values):
        return self.explain(values).prediction

    def translate_values(self, values):
        """
        ``(feature name, human-readable value)`` pairs of an applicant in
        the order of ``values``.
        """
        out = []
        for feature, value in values.items():
            m = self.feature_map[feature]['map']
            out.append((self.feature_name(feature),
                        value if m == 'numerical' else m[str(value)]))
        return out

    def describe(self, counterfactual, integer_ranges=False):
        """
        Human-readable counterfactual (``feature name -> {'is', 'wannabe'}``).

        Parameters
        ----------
        counterfactual : dict
            ``{feature id: {'is': [...], 'wannabe': [...]}}``.
        integer_ranges : bool
            Describe intervals of integer features with whole numbers (e.g.,
            ``(954.5, 1181.5]`` -> ``between 955 and 1181``) instead of the
            split thresholds
            (see :func:`ctreex.counterfactuals.describe_integer_range`).
        """
        return self.explainer.describe({0: counterfactual},
                                       integer_ranges=integer_ranges)[0]


def train_tree(data_holder, random_state=None, **params):
    """
    Trains a new decision tree on the German Credit data.

    The original model was ``DecisionTreeClassifier(min_samples_split=3)``
    trained on all the data.

    Returns
    -------
    ctreex.TreeStructure
    """
    from sklearn.tree import DecisionTreeClassifier
    from ctreex import TreeStructure
    params.setdefault('min_samples_split', 3)
    clf = DecisionTreeClassifier(random_state=random_state, **params)
    clf.fit(data_holder.data_onehot, np.asarray(data_holder.target))
    return TreeStructure.from_sklearn(
        clf, feature_names=list(data_holder.feature_id_onehot),
        metadata={'description': 'Retrained Glass-Box tree.'})
