"""
Switches between the original dialogue and fixed behaviour.

Each behaviour-changing fix of the dialogue and of the mapping from parsed
sentences to intents can be toggled individually.
The original implementation deliberately kept spoken answers short, so with
``implementation='fixed'`` the default behaviour depends on the output channel:
the original behaviour is kept for speech (``channel='speech'``) and the fix is
used for chat (``channel='chat'``).
"""
import dataclasses

__all__ = ['DialogueFixes', 'resolve_dialogue_fixes', 'CHANNELS',
           'CHANNEL_DEPENDENT']

CHANNELS = ('chat', 'speech')
#: Fixes whose default (with ``implementation='fixed'``) depends on the
#: channel: original behaviour when speaking, the fix in chat.
CHANNEL_DEPENDENT = ('myself_all_features', 'stop_when_exhausted',
                     'integer_ranges')


@dataclasses.dataclass(frozen=True)
class DialogueFixes(object):
    """
    Behaviour-changing fixes of the dialogue.

    Attributes
    ----------
    wording : bool
        Grammar and typography of the answers: missing verbs ("had you between
        100 and 500 pounds on your savings account" -> "had you had ..."),
        double and leading spaces, sentences starting with a lower-case
        letter, a missing full stop, a missing comma before "your loan would
        be ...", "process you application", the article
        of the housing type chosen from the wrong value and a typo in the help
        message.
    myself_all_features : bool
        Asking about oneself without naming a feature lists *all* the
        features; the original reported only the current account status
        (A1) to keep spoken answers short. *Channel-dependent.*
    stop_when_exhausted : bool
        When all the counterfactuals of a question (why/fairness/action) have
        been given, say so (and start from the beginning on the next
        request); the original silently started over. *Channel-dependent.*
    integer_ranges : bool
        Describe ranges of integer features with whole numbers (``between
        955 and 1181 pounds`` instead of ``between 954.5 and 1181.5
        pounds``). *Channel-dependent.*
    feature_keywords : bool
        Recognise features in sentences with keywords derived from the NLU
        training data; the original keyword rules sometimes missed annotated
        feature mentions (e.g. "married", "rent", "employed") and mapped
        "separated" to the installment rate.
    action_restrictions : bool
        "What can I do" requests accept the same restrictions as "why"
        ("Disregarding my age, what can I do ...?", "What can I do with my
        installment rate ...?"): suggestions that change a dropped feature
        are skipped and suggestions must change all the required ones. The
        original ignored them, although its help message suggested such
        questions.
    why_reset : bool
        Honour the ``why_reset`` entity ("restart why") recognised by the NLU
        model; the original ignored it (restarting was only possible with
        the ``cmd:`` syntax).
    """
    wording: bool = True
    myself_all_features: bool = True
    stop_when_exhausted: bool = True
    integer_ranges: bool = True
    feature_keywords: bool = True
    why_reset: bool = True
    action_restrictions: bool = True

    @classmethod
    def original(cls):
        return cls(**{f.name: False for f in dataclasses.fields(cls)})

    @classmethod
    def fixed(cls, channel='chat'):
        if channel not in CHANNELS:
            raise ValueError('channel has to be one of {}.'.format(CHANNELS))
        values = {f.name: True for f in dataclasses.fields(cls)}
        if channel == 'speech':
            values.update({name: False for name in CHANNEL_DEPENDENT})
        return cls(**values)


def resolve_dialogue_fixes(implementation='fixed', overrides=None,
                           channel='chat'):
    """
    Resolves the dialogue switches for a channel.

    Parameters
    ----------
    implementation : {'original', 'fixed'}
    overrides : dict, optional
        Explicit values of selected switches (applied to every channel).
    channel : {'chat', 'speech'}
    """
    if implementation == 'original':
        base = DialogueFixes.original()
    elif implementation == 'fixed':
        base = DialogueFixes.fixed(channel)
    else:
        raise ValueError("implementation has to be 'original' or 'fixed'.")
    if overrides:
        unknown = set(overrides) - {f.name for f in
                                    dataclasses.fields(DialogueFixes)}
        if unknown:
            raise ValueError('Unknown dialogue fixes: {}.'.format(
                sorted(unknown)))
        base = dataclasses.replace(base, **overrides)
    return base
