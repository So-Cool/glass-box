"""
The device server running on the Raspberry Pi.

Port of the original Glass-Box server -- which handled LEDs and
speech over HTTP -- together with the voice input and the QR scanner modules.
These components run in background threads and send their input to
the Glass-Box application (``hardware.voice.app_url``/``hardware.qr.app_url``).

LEDs and speech are configured by ``[device.leds]``/``[device.speech]``
(falling back to ``[hardware.leds]``/``[hardware.speech]``).

Endpoints: ``POST /led`` (LED command), ``POST /say`` (alias ``/aiysay``,
``{"text": ...}``), ``GET /health``.
"""
import logging
import threading

import flask

from glassbox.hardware import create_leds, create_speaker
from glassbox.hardware.qr import ZbarQRScanner
from glassbox.hardware.voice import AIYVoiceInput

__all__ = ['create_device_app', 'start_agents']

logger = logging.getLogger(__name__)


def start_agents(config, speaker):
    """Starts the voice input and QR scanner threads (if enabled)."""
    threads = []
    hardware = config['hardware']
    if hardware['voice'].get('enabled'):
        voice = AIYVoiceInput.posting_to(hardware['voice']['app_url'],
                                         speaker)
        threads.append(threading.Thread(target=voice.run, name='voice',
                                        daemon=True))
    if hardware['qr'].get('enabled'):
        scanner = ZbarQRScanner.posting_to(
            hardware['qr']['app_url'], device=hardware['qr']['device'])
        threads.append(threading.Thread(target=scanner.run, name='qr',
                                        daemon=True))
    for t in threads:
        t.start()
    return threads


def create_device_app(config, start=True):
    """Creates the device server (and starts the input agents)."""
    # [device.leds]/[device.speech] take precedence over [hardware.*] so that
    # one file can configure both processes on a standalone Raspberry Pi
    leds = create_leds(config['device'].get('leds')
                       or config['hardware']['leds'])
    speaker = create_speaker(config['device'].get('speech')
                             or config['hardware']['speech'])
    app = flask.Flask(__name__)
    app.config['LEDS'], app.config['SPEAKER'] = leds, speaker

    @app.route('/')
    def index():
        return 'Glass-Box device server'

    @app.route('/led', methods=['POST'])
    def led():
        command = flask.request.get_json(silent=True)
        if not isinstance(command, dict):
            return flask.jsonify({'status': 'NOT-OK'}), 400
        leds.send(command)
        return flask.jsonify({'status': 'OK'})

    @app.route('/say', methods=['POST'])
    @app.route('/aiysay', methods=['POST'])
    def say():
        data = flask.request.get_json(silent=True) or {}
        text = data.get('text', '')
        if not text:
            return flask.jsonify({'status': 'NOT-OK'})
        speaker.say(text)
        return flask.jsonify({'status': 'OK'})

    @app.route('/health')
    def health():
        return flask.jsonify({'status': 'OK'})

    if start:
        app.config['AGENTS'] = start_agents(config, speaker)
    return app
