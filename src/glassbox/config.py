"""
Configuration (TOML files merged over :data:`DEFAULTS`).

See ``config/*.toml`` in the repository for documented examples:
a laptop (chat only, mock hardware), the Raspberry Pi device server and
a standalone Raspberry Pi.
"""
import copy
import os
import tomllib

__all__ = ['DEFAULTS', 'load_config', 'merge']

DEFAULTS = {
    'app': {'host': '127.0.0.1', 'port': 5000, 'poll_interval': 2.0},
    'model': {'tree': None},
    'dialogue': {'implementation': 'fixed',
                 'explainer_fixes': {}},
    'nlu': {'backend': 'spacy', 'thinking_delay': 0.0,
            'spacy': {'model_dir': None, 'spacy_model': None},
            'rasa_legacy': {'url': 'http://127.0.0.1:5005',
                            'project': 'default', 'model': 'chat'}},
    'hardware': {
        'leds': {'backend': 'mock'},
        'speech': {'backend': 'disabled'},
        'voice': {'enabled': False, 'app_url': 'http://127.0.0.1:5000'},
        'qr': {'enabled': False, 'app_url': 'http://127.0.0.1:5000',
               'device': '/dev/video0'}},
    'device': {'host': '0.0.0.0', 'port': 5001},
    'logging': {'level': 'INFO'},
}


def merge(base, override):
    """Recursively merges dictionaries (``override`` wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path=None, overrides=None):
    """
    Loads a configuration file.

    Parameters
    ----------
    path : str, optional
        A TOML file (default: ``$GLASSBOX_CONFIG`` if set, else the
        defaults only).
    overrides : dict, optional
        Values applied on top of the file.
    """
    path = path or os.environ.get('GLASSBOX_CONFIG')
    config = DEFAULTS
    if path:
        with open(path, 'rb') as f:
            config = merge(config, tomllib.load(f))
    if overrides:
        config = merge(config, overrides)
    if 'fixes' in config['dialogue']:
        raise ValueError('The dialogue fixes are no longer switchable -- '
                         'remove [dialogue.fixes] from the configuration '
                         '(the explainer fixes are in '
                         '[dialogue.explainer_fixes]).')
    if config['dialogue']['implementation'] not in ('original', 'fixed'):
        raise ValueError("dialogue.implementation has to be 'original' or "
                         "'fixed'.")
    return config
