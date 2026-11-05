"""The ``glassbox`` command line tool."""
import argparse
import json
import logging
import sys

from glassbox.config import load_config

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
    print('Pre-processed data: {} rows, checksum {}'.format(len(dh.data),
                                                           checksum))
    ok = checksum == data.PROCESSED_DATA_SHA256
    print('Matches the Glass-Box data: {}'.format(ok))
    personas = data.personas_matching_data(dh)
    print('Personas copied from the data set: {}'.format(personas))
    return 0 if ok and all(personas.values()) else 1


def cmd_nlu(args):
    from glassbox.nlu.spacy_backend import SpacyNLU
    if args.action == 'train':
        nlu = SpacyNLU.train(spacy_model=args.spacy_model)
        print('Saved to', nlu.save(args.output) if args.output
              else nlu.save())
    else:
        from glassbox.nlu import Intentioner
        nlu = SpacyNLU.load()
        parsed = nlu.parse(args.text)
        print(json.dumps(parsed, indent=2))
        print(Intentioner(nlu).parse_to_intent(parsed))


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

    p = sub.add_parser('nlu', help='train or try the default NLU model')
    p.add_argument('action', choices=['train', 'parse'])
    p.add_argument('text', nargs='?', default='')
    p.add_argument('--spacy-model', default='en_core_web_sm')
    p.add_argument('--output')
    p.set_defaults(func=cmd_nlu)

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
