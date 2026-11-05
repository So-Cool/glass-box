"""Speech output (text-to-speech of the AIY Voice Kit)."""
import abc
import logging

from glassbox.hardware._http import post_json

__all__ = ['Speaker', 'AIYSpeaker', 'RemoteSpeaker', 'MockSpeaker',
           'DisabledSpeaker', 'create_speaker']

logger = logging.getLogger(__name__)


class Speaker(abc.ABC):
    enabled = True

    @abc.abstractmethod
    def say(self, text):
        """Speaks a text."""


class DisabledSpeaker(Speaker):
    enabled = False

    def say(self, text):
        pass


class MockSpeaker(Speaker):
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)
        logger.info('SAY: %s', text)


class RemoteSpeaker(Speaker):
    """Asks the device server to speak (``POST <url>/say``)."""

    def __init__(self, url, timeout=1.0):
        self.url = url.rstrip('/') + '/say'
        self.timeout = timeout

    def say(self, text):
        post_json(self.url, {'text': text}, timeout=self.timeout)


class AIYSpeaker(Speaker):  # pragma: no cover (hardware)
    """
    Text-to-speech with the AIY library (Raspberry Pi with an AIY Voice Kit).

    Supports the 2018 library (``aiy.audio.say``) and the current one
    (``aiy.voice.tts.say``).
    """

    def __init__(self, lang='en-GB', volume=60):
        try:
            import aiy.audio
            self._say = lambda text: aiy.audio.say(text)
        except ImportError:
            try:
                from aiy.voice import tts
            except ImportError as e:
                raise ImportError('The aiy speech backend requires the AIY '
                                  'Projects library (see the hardware '
                                  'guide).') from e
            self._say = lambda text: tts.say(text, lang=lang, volume=volume)

    def say(self, text):
        self._say(text)


def create_speaker(config):
    """Creates the speech backend from ``[hardware.speech]``."""
    backend = config.get('backend', 'disabled')
    if backend == 'disabled':
        return DisabledSpeaker()
    if backend == 'mock':
        return MockSpeaker()
    if backend == 'remote':
        return RemoteSpeaker(config['url'], timeout=config.get('timeout', 1))
    if backend == 'aiy':
        return AIYSpeaker(**{k: v for k, v in config.items()
                             if k in ('lang', 'volume')})
    raise ValueError('Unknown speech backend: {!r}.'.format(backend))
