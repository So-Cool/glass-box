"""
Voice input on the AIY Voice Kit (Raspberry Pi only).

Behaviour: wait for the button, recognise speech, send the text to
the Glass-Box application and speak the reply.
"""
import logging

from glassbox.hardware._http import post_json

__all__ = ['AIYVoiceInput']

logger = logging.getLogger(__name__)


class AIYVoiceInput(object):  # pragma: no cover (hardware)
    """
    Push-to-talk speech recognition with the AIY library.

    The 2018 Voice Kit (v1) used the Google Assistant gRPC API
    (``aiy.assistant.grpc``), which Google has deprecated;
    the current AIY library offers Google Cloud Speech (``aiy.cloudspeech``),
    which is used when the former is not available.

    Parameters
    ----------
    handle_text : callable
        Called with the recognised text; returns the reply to speak (or an
        empty string).
    speaker : glassbox.hardware.speech.Speaker
    """

    def __init__(self, handle_text, speaker, language='en-GB'):
        self.handle_text = handle_text
        self.speaker = speaker
        self.language = language

    @classmethod
    def posting_to(cls, app_url, speaker, timeout=30.):
        """Voice input that sends the text to ``<app_url>/voice``."""
        url = app_url.rstrip('/') + '/voice'

        def handle(text):
            reply = post_json(url, {'text': text}, timeout=timeout)
            if reply is None:
                logger.warning('Host unreachable: %s', url)
                return ''
            return reply.get('text', '')
        return cls(handle, speaker)

    def run(self):
        try:
            self._run_v1()
        except ImportError:
            self._run_cloudspeech()

    def _respond(self, text):
        if not text:
            return
        logger.info('You said "%s"', text)
        reply = self.handle_text(text)
        if reply:
            self.speaker.say(reply)

    def _run_v1(self):
        import aiy.assistant.grpc
        import aiy.audio
        import aiy.voicehat
        status_ui = aiy.voicehat.get_status_ui()
        status_ui.status('starting')
        assistant = aiy.assistant.grpc.get_assistant()
        button = aiy.voicehat.get_button()
        with aiy.audio.get_recorder():
            while True:
                status_ui.status('ready')
                logger.info('Press the button and speak')
                button.wait_for_press()
                status_ui.status('listening')
                text, _ = assistant.recognize()
                self._respond(text)

    def _run_cloudspeech(self):
        from aiy.board import Board
        from aiy.cloudspeech import CloudSpeechClient
        client = CloudSpeechClient()
        with Board() as board:
            while True:
                logger.info('Press the button and speak')
                board.button.wait_for_press()
                text = client.recognize(language_code=self.language)
                self._respond(text)
