"""
Natural language understanding: sentences -> intents and entities.

Backends (``nlu.backend`` in the configuration):

``spacy`` (default)
    :class:`glassbox.nlu.spacy_backend.SpacyNLU` -- a reimplementation of the
    original Rasa NLU 0.12 pipeline (spaCy features, a CRF entity extractor
    and a linear SVM intent classifier) trained on the original training
    data (``training_data.json``) extended with additional examples
    (``training_data_extra.json``).
``rasa-legacy``
    :class:`glassbox.nlu.rasa_legacy.RasaLegacyNLU` -- the *original* trained
    Rasa NLU 0.12.3 model served by a Docker container
    (see ``docker/rasa-legacy``).

:class:`glassbox.nlu.intentioner.Intentioner` turns a parse into a dialogue
intent and its arguments; :func:`glassbox.nlu.command.parse_command` handles
the ``cmd: intent->arg:value;...`` syntax.
"""
from glassbox.nlu.base import NLUBackend, get_backend
from glassbox.nlu.intentioner import Intentioner

__all__ = ['NLUBackend', 'get_backend', 'Intentioner']
