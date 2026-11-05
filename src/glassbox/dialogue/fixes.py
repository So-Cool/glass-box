"""
Switches between the original dialogue and fixed behaviour.

Each behaviour-changing fix of the dialogue and of the mapping from parsed
sentences to intents can be toggled individually.
Corrections of the wording, whole numbers in ranges and telling the applicant
when all the explanations have been given are not switchable -- they are
always used (see the changelog).
"""
import dataclasses

__all__ = ['DialogueFixes', 'resolve_dialogue_fixes']


@dataclasses.dataclass(frozen=True)
class DialogueFixes(object):
    """
    Behaviour-changing fixes of the dialogue.

    Attributes
    ----------
    feature_keywords : bool
        Recognise features in sentences with keywords derived from the NLU
        training data; the original keyword rules sometimes missed annotated
        feature mentions (e.g., "married", "rent", "employed") and mapped
        "separated" to the instalment rate.
    action_restrictions : bool
        "What can I do" requests accept the same restrictions as "why"
        ("Disregarding my age, what can I do ...?", "What can I do with my
        instalment rate ...?"): suggestions that change a dropped feature
        are skipped and suggestions must change all the required ones. The
        original ignored them, although its help message suggested such
        questions.
    why_reset : bool
        Honour the ``why_reset`` entity ("restart why") recognised by the NLU
        model; the original ignored it (restarting was only possible with
        the ``cmd:`` syntax).
    """
    feature_keywords: bool = True
    why_reset: bool = True
    action_restrictions: bool = True

    @classmethod
    def original(cls):
        return cls(**{f.name: False for f in dataclasses.fields(cls)})

    @classmethod
    def fixed(cls):
        return cls(**{f.name: True for f in dataclasses.fields(cls)})


def resolve_dialogue_fixes(implementation='fixed', overrides=None):
    """
    Resolves the dialogue switches.

    Parameters
    ----------
    implementation : {'original', 'fixed'}
    overrides : dict, optional
        Explicit values of selected switches.
    """
    if implementation == 'original':
        base = DialogueFixes.original()
    elif implementation == 'fixed':
        base = DialogueFixes.fixed()
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
