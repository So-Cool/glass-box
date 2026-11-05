"""
Evaluates the default NLU backend on the held-out test set.

    uv run python tools/nlu/evaluate.py

Compares a model trained on the 162 original examples ("2018 data") with one
trained on the original and the added examples ("2018 + 2026 data"), both with
the default (spaCy) backend, on ``tests/data/nlu_test_set.json``,
which contains sentences written with templates that are not used for training.

Reported: intent accuracy, entity recall and precision (exact span and type),
and the end-to-end accuracy of the (intent, arguments) pairs given to the
dialogue (fixed implementation).
"""
import json
import os
import warnings

from glassbox.nlu import Intentioner
from glassbox.nlu.spacy_backend import (EXTRA_TRAINING_DATA, TRAINING_DATA,
                                        SpacyNLU, load_training_data)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
TEST_SET = os.path.join(ROOT, 'tests', 'data', 'nlu_test_set.json')


def evaluate(nlu, examples):
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


def main():
    warnings.filterwarnings('ignore')
    with open(TEST_SET) as f:
        test = json.load(f)['rasa_nlu_data']['common_examples']
    rows = {}
    for label, paths in (('2018 data', (TRAINING_DATA, )),
                         ('2018 + 2026 data', (TRAINING_DATA,
                                               EXTRA_TRAINING_DATA))):
        nlu = SpacyNLU.train(load_training_data(paths))
        rows[label] = evaluate(nlu, test)
    print('{} test sentences'.format(len(test)))
    print('{:32s}'.format('') + ''.join('{:>18s}'.format(k) for k in rows))
    for metric in next(iter(rows.values())):
        print('{:32s}'.format(metric) + ''.join(
            '{:>18.1%}'.format(rows[k][metric]) for k in rows))
    return rows


if __name__ == '__main__':
    main()
