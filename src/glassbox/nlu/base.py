"""The interface of NLU backends."""
import abc

__all__ = ['NLUBackend', 'get_backend', 'BACKENDS']


class NLUBackend(abc.ABC):
    """
    Parses a sentence into an intent and entities (Rasa NLU format).

    :meth:`parse` returns::

        {"text": ..., "intent": {"name": ..., "confidence": ...},
         "entities": [{"entity": ..., "value": ..., "start": ...,
                       "end": ...}, ...]}

    where ``value`` is the text span of an entity.
    """

    @abc.abstractmethod
    def parse(self, text):
        """Parses a sentence."""


BACKENDS = ('spacy', 'rasa-legacy')


def get_backend(name='spacy', **options):
    """
    Creates an NLU backend.

    Parameters
    ----------
    name : {'spacy', 'rasa-legacy'}
    **options
        Passed to the backend class.
    """
    if name == 'spacy':
        from glassbox.nlu.spacy_backend import SpacyNLU
        return SpacyNLU.load(**options)
    if name == 'rasa-legacy':
        from glassbox.nlu.rasa_legacy import RasaLegacyNLU
        return RasaLegacyNLU(**options)
    raise ValueError('Unknown NLU backend {!r}; use one of {}.'.format(
        name, BACKENDS))
