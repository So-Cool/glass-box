"""
Evaluates the default NLU backend on the held-out test set.

``glassbox nlu test`` compares a model trained on the 162 original examples
("2018 data") with one trained on the original and the added examples ("2018 +
2026 data"), both with the default (spaCy) backend, on sentences written with
templates that are not used for training (``tests/data/nlu_test_set.json``,
see :mod:`glassbox.nlu.data_generation`).

Reported:
intent accuracy, entity recall and precision (exact span and type), and
the accuracy of the (intent, arguments) pairs given to the dialogue (fixed
implementation).
"""
import json

from glassbox.nlu.intentioner import Intentioner
from glassbox.nlu.spacy_backend import (EXTRA_TRAINING_DATA, TRAINING_DATA,
                                        SpacyNLU, load_training_data)

__all__ = ['evaluate', 'compare', 'format_table', 'load_test_set']


def load_test_set(path):
    """The examples of a test set (Rasa NLU format)."""
    with open(path) as f:
        return json.load(f)['rasa_nlu_data']['common_examples']


def evaluate(nlu, examples):
    """
    Scores an NLU backend on annotated examples.

    Returns
    -------
    dict
        ``intent accuracy``, ``entity recall``, ``entity precision`` and
        ``(intent, arguments) accuracy``.
    """
    intentioner = Intentioner(nlu)
    gold_mapper = Intentioner()
    # the gold entities carry the annotated ids (e.g., A13, 09) as values
    gold_mapper.decode_feature = lambda value: value
    intents = mapped = found = predicted = gold = 0
    for e in examples:
        parse = nlu.parse(e['text'])
        intents += parse['intent']['name'] == e['intent']
        g = {(x['start'], x['end'], x['entity']) for x in e['entities']}
        p = {(x['start'], x['end'], x['entity']) for x in parse['entities']}
        found += len(g & p)
        gold += len(g)
        predicted += len(p)
        gold_parse = {'intent': {'name': e['intent']},
                      'entities': e['entities']}
        mapped += (intentioner.parse_to_intent(parse)
                   == gold_mapper.parse_to_intent(gold_parse))
    n = len(examples)
    return {'intent accuracy': intents / n,
            'entity recall': found / gold if gold else 1.,
            'entity precision': found / predicted if predicted else 1.,
            '(intent, arguments) accuracy': mapped / n}


def compare(examples, seed=42):
    """
    Trains on the 2018 data and on the 2018 + 2026 data and scores both.

    Parameters
    ----------
    examples : list of dict
        The test set.
    seed : int
        Random state of the training (cross-validation of the SVM).

    Returns
    -------
    dict
        ``{label: scores}``.
    """
    rows = {}
    for label, paths in (('2018 data', (TRAINING_DATA, )),
                         ('2018 + 2026 data', (TRAINING_DATA,
                                               EXTRA_TRAINING_DATA))):
        nlu = SpacyNLU.train(load_training_data(paths), random_state=seed)
        rows[label] = evaluate(nlu, examples)
    return rows


def format_table(rows):
    """Formats the output of :func:`compare` as a text table."""
    lines = ['{:32s}'.format('') + ''.join('{:>18s}'.format(k)
                                           for k in rows)]
    for metric in next(iter(rows.values())):
        lines.append('{:32s}'.format(metric) + ''.join(
            '{:>18.1%}'.format(rows[k][metric]) for k in rows))
    return '\n'.join(lines)
