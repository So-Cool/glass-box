"""
The assembled Glass-Box system.

:class:`GlassBox` wires the model, the dialogue manager, the NLU and the
hardware together and implements the message handling of the chat server:

* ``quit``/``exit`` are refused, ``disable`` toggles spoken answers;
* ``cmd: <intent>->...`` issues an intent directly;
* any other text is parsed by the NLU;
* voice input and QR scans produce spoken answers and are logged in the chat.
"""
import collections
import logging
import threading
import time

from glassbox.config import load_config
from glassbox.dialogue import DialogueManager
from glassbox.hardware import create_leds, create_speaker
from glassbox.model import GlassBoxModel
from glassbox.nlu import Intentioner, get_backend
from glassbox.nlu.command import is_command, parse_command

__all__ = ['GlassBox']

logger = logging.getLogger(__name__)
CRASH = ("Sorry but I have just crashed. Unfortunately, you have to restart "
         "your application.")


class GlassBox(object):
    """
    The Glass-Box system.

    Parameters
    ----------
    config : dict, optional
        Configuration (see :func:`glassbox.config.load_config`).
    nlu : glassbox.nlu.NLUBackend, optional
        An NLU backend (created from the configuration -- lazily -- when
        omitted).
    """

    def __init__(self, config=None, nlu=None):
        self.config = load_config() if config is None else config
        dialogue = self.config['dialogue']
        self.leds = create_leds(self.config['hardware']['leds'])
        self.speaker = create_speaker(self.config['hardware']['speech'])
        self.model = GlassBoxModel(
            tree=self.config['model'].get('tree'),
            implementation=dialogue['implementation'],
            fixes=dialogue.get('explainer_fixes') or None)
        self.dialogue = DialogueManager(
            self.model, implementation=dialogue['implementation'],
            on_event=self.leds.show)
        self._nlu = nlu
        self.intentioner = Intentioner(nlu)
        self.messages = collections.deque()
        self.voice_disabled = False
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ NLU
    @property
    def nlu(self):
        if self._nlu is None:
            cfg = self.config['nlu']
            if cfg['backend'] == 'spacy':
                options = {k: v for k, v in cfg.get('spacy', {}).items()
                           if v is not None}
            else:
                options = dict(cfg.get('rasa_legacy', {}))
            self._nlu = get_backend(cfg['backend'], **options)
            self.intentioner.nlu = self._nlu
        return self._nlu

    def ask(self, sentence, channel='chat'):
        """Answers a natural language request."""
        start = time.time()
        self.leds.show('thinking')
        try:
            self.nlu  # noqa: B018 (load lazily)
            intent, args = self.intentioner.sentence_to_intent(sentence)
        finally:
            delay = self.config['nlu'].get('thinking_delay', 0)
            if delay and self.leds.visible and time.time() - start < 5:
                time.sleep(delay)  # let the "thinking" LEDs be seen
            self.leds.show('idle')
        logger.debug('%r -> %s %s', sentence, intent, args)
        return self.dialogue.process_intent(intent, args, channel=channel)

    # ------------------------------------------------------------- messages
    def handle_message(self, message, channel='chat'):
        """Answers a chat (or voice) message."""
        message = message.strip()
        with self.lock:
            if message in ('quit', 'exit'):
                return "Sorry but I will not quit."
            if message == 'disable':
                self.voice_disabled = not self.voice_disabled
                return ("Voice answers disabled." if self.voice_disabled
                        else "Voice answers enabled.")
            if is_command(message):
                self.leds.show('command')
                try:
                    intent, args = parse_command(message)
                except ValueError as e:
                    return 'Invalid command: {}'.format(e)
                return self.dialogue.process_intent(intent, args,
                                                    channel=channel)
            try:
                return self.ask(message, channel=channel)
            except Exception:
                logger.exception('Failed to answer %r.', message)
                return CRASH

    def handle_voice(self, text):
        """Answers voice input; returns the text to speak ('' if muted)."""
        self.messages.append('Voice: ' + text)
        response = self.handle_message(text, channel='speech')
        self.messages.append('Voice: ' + response)
        return '' if self.voice_disabled else response

    def handle_qr(self, persona):
        """Starts an application for a scanned persona card."""
        with self.lock:
            if self.dialogue.persona is not None:
                return None
            for text, (intent, args) in (
                    ('QR scanner: Hello!', ('hello', {})),
                    ('QR scanner: My personal identification number is: '
                     '{}.'.format(persona), ('apply', {'persona': persona}))):
                self.messages.append(text)
                reply = self.dialogue.process_intent(intent, args,
                                                     channel='speech')
                self.messages.append(reply)
        if not self.voice_disabled:
            self.speaker.say(reply)
        return reply

    def page_loaded(self):
        self.leds.show('page_loaded')

    def pop_messages(self):
        """Returns (and forgets) the queued asynchronous messages."""
        out = []
        while self.messages:
            out.append(self.messages.popleft())
        return out
