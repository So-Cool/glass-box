"""The ``glassbox`` command line tool."""
import argparse
import json
import logging
import sys

from glassbox.config import load_config
from glassbox.nlu.data_generation import (DEFAULT_TEST_OUTPUT,
                                          DEFAULT_TRAINING_OUTPUT)

__all__ = ['main']


def _box(args):
    from glassbox.core import GlassBox
    overrides = {}
    if getattr(args, 'implementation', None):
        overrides['dialogue'] = {'implementation': args.implementation}
    return GlassBox(load_config(args.config, overrides))


def cmd_serve(args):
    from glassbox.app import create_app
    box = _box(args)
    app = create_app(box)
    cfg = box.config['app']
    app.run(host=args.host or cfg['host'], port=args.port or cfg['port'],
            debug=False, use_reloader=False, threaded=True)


def cmd_device(args):
    from glassbox.device_server import create_device_app
    config = load_config(args.config)
    app = create_device_app(config)
    cfg = config['device']
    app.run(host=args.host or cfg['host'], port=args.port or cfg['port'],
            debug=False, use_reloader=False, threaded=True)


def cmd_chat(args):
    box = _box(args)
    print('Glass-Box chat -- type "hello" to start, Ctrl-D to leave.')
    for line in sys.stdin if not sys.stdin.isatty() else iter(
            lambda: input('> '), None):
        line = line.strip()
        if line:
            print(box.handle_message(line.strip('.?!')))


def cmd_data(args):
    from glassbox import data
    dh = data.load_german_credit(data_home=args.data_home, verify=False)
    checksum = data.processed_data_checksum(dh.data)
    print('Pre-processed data: {} rows, checksum {}'.format(
        len(dh.data), checksum))
    ok = checksum == data.PROCESSED_DATA_SHA256
    print('Matches the Glass-Box data: {}'.format(ok))
    personas = data.personas_matching_data(dh)
    print('Personas copied from the data set: {}'.format(personas))
    return 0 if ok and all(personas.values()) else 1


def cmd_nlu(args):
    if args.nlu_command == 'train':
        from glassbox.nlu.spacy_backend import SpacyNLU
        nlu = SpacyNLU.train(spacy_model=args.spacy_model,
                             random_state=args.seed)
        print('Saved to', nlu.save(args.output) if args.output
              else nlu.save())
    elif args.nlu_command == 'parse':
        from glassbox.nlu import Intentioner
        from glassbox.nlu.spacy_backend import SpacyNLU
        nlu = SpacyNLU.load()
        parsed = nlu.parse(args.text)
        print(json.dumps(parsed, indent=2))
        print(Intentioner(nlu).parse_to_intent(parsed))
    elif args.nlu_command == 'generate':
        from glassbox.nlu import data_generation
        n_train, n_test = data_generation.write_data(
            training_output=args.output, test_output=args.test_output,
            repeats=args.repeats, seed=args.seed,
            test_repeats=args.test_repeats, test_seed=args.test_seed)
        print('{} training examples -> {}'.format(n_train, args.output))
        print('{} test examples -> {}'.format(n_test, args.test_output))
    else:  # test
        from glassbox.nlu import evaluation
        examples = evaluation.load_test_set(args.test_set)
        rows = evaluation.compare(examples, seed=args.seed)
        print('{} test sentences'.format(len(examples)))
        print(evaluation.format_table(rows))


def cmd_model(args):
    from ctreex import load_tree
    if args.action == 'export':
        tree = load_tree(args.input)
    else:
        from glassbox import data
        from glassbox.model import train_tree
        tree = train_tree(data.load_german_credit(), args.random_state)
    tree.to_json(args.output, indent=None)
    print('Saved', tree, 'to', args.output)


def cmd_config(args):
    print(json.dumps(load_config(args.config), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='glassbox', description='Glass-Box: Explaining loan decisions '
        'with counterfactual statements.')
    parser.add_argument('-c', '--config', help='TOML configuration file')
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('serve', help='run the chat web application')
    p.add_argument('--host')
    p.add_argument('--port', type=int)
    p.add_argument('--implementation', choices=['original', 'fixed'])
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser('device', help='run the Raspberry Pi device server')
    p.add_argument('--host')
    p.add_argument('--port', type=int)
    p.set_defaults(func=cmd_device)

    p = sub.add_parser('chat', help='chat in the terminal')
    p.add_argument('--implementation', choices=['original', 'fixed'])
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser('data', help='download and verify the German Credit '
                       'data (needs the "data" extra)')
    p.add_argument('--data-home')
    p.set_defaults(func=cmd_data)

    p = sub.add_parser('nlu', help='the default NLU model: train, try, '
                       'generate training data and evaluate')
    nlu = p.add_subparsers(dest='nlu_command', required=True)
    q = nlu.add_parser('train', help='train the default NLU model on the '
                       'original and the added training data')
    q.add_argument('--spacy-model', default='en_core_web_sm')
    q.add_argument('--output', help='model directory (default: the model '
                   'shipped with the package)')
    q.add_argument('--seed', type=int, default=42,
                   help='random state of the training (default: 42)')
    q.set_defaults(func=cmd_nlu)
    q = nlu.add_parser('parse', help='show how a sentence is understood')
    q.add_argument('text')
    q.set_defaults(func=cmd_nlu)
    q = nlu.add_parser('generate', help='generate the added training data '
                       'and the held-out test set from templates')
    q.add_argument('--repeats', type=int, default=3,
                   help='fills of every training template (default: 3)')
    q.add_argument('--seed', type=int, default=2026,
                   help='seed of the training data (default: 2026)')
    q.add_argument('--test-repeats', type=int, default=3,
                   help='fills of every test template (default: 3)')
    q.add_argument('--test-seed', type=int, default=7,
                   help='seed of the test set (default: 7)')
    q.add_argument('--output', default=DEFAULT_TRAINING_OUTPUT,
                   help='training data file (default: the file shipped with '
                   'the package)')
    q.add_argument('--test-output', default=DEFAULT_TEST_OUTPUT,
                   help='test set file (default: %(default)s)')
    q.set_defaults(func=cmd_nlu)
    q = nlu.add_parser('test', help='evaluate models trained on the original '
                       'and on all the training data on the test set')
    q.add_argument('--test-set', default=DEFAULT_TEST_OUTPUT,
                   help='test set file (default: %(default)s)')
    q.add_argument('--seed', type=int, default=42,
                   help='random state of the training (default: 42)')
    q.set_defaults(func=cmd_nlu)

    p = sub.add_parser('model', help='convert a pickled tree to JSON or '
                       'retrain the tree')
    p.add_argument('action', choices=['export', 'retrain'])
    p.add_argument('--input', help='pickled tree (export)')
    p.add_argument('--output', required=True, help='JSON file')
    p.add_argument('--random-state', type=int)
    p.set_defaults(func=cmd_model)

    p = sub.add_parser('config', help='print the effective configuration')
    p.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    config_level = 'DEBUG' if args.verbose else 'INFO'
    logging.basicConfig(level=config_level,
                        format='%(asctime)s %(name)s %(levelname)s: '
                        '%(message)s')
    try:
        return args.func(args) or 0
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
