"""The ``cmd:`` syntax for issuing intents directly (no NLU).

Example: ``cmd: why->features_require:[A9];features_drop:[A13]``.
"""

__all__ = ['parse_command', 'is_command']

PREFIX = 'cmd:'


def is_command(message):
    return message.startswith(PREFIX)


def parse_command(message):
    """
    Parses ``cmd: <intent>->key:value;key:[v1,v2]``.

    Returns
    -------
    tuple
        ``(intent, args)``; list values are given in square brackets.
    """
    if not is_command(message):
        raise ValueError('Commands start with {!r}.'.format(PREFIX))
    cmd = message[len(PREFIX):].strip().split("->")
    intent = cmd[0].strip()
    args = {}
    if len(cmd) > 1 and cmd[1].strip():
        for item in cmd[1].split(";"):
            if not item.strip():
                continue
            if ":" not in item:
                raise ValueError('Arguments have the form key:value; got '
                                 '{!r}.'.format(item))
            key, value = item.split(":", 1)
            if "[" in value:
                value = [j.strip().strip("[]") for j in value.split(",")
                         if j.strip().strip("[]")]
            else:
                value = value.strip()
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
            args[key.strip()] = value
    return intent, args
