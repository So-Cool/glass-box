"""
The Glass-Box web application (chat interface and device endpoints).

Endpoints (aliases of the original names in brackets):

``GET /``                                                          the chat page
``POST /ask``                       ``{"message": ...}`` or form ``messageText``
``POST /voice`` (``/voicehat``)           ``{"text": ...}`` from the voice input
``POST /qrcode``                ``{"persona": ...}`` or ``{"qr": "QR-Code:04"}``
``GET /messages`` (``/getmessage``)            asynchronous messages (voice, QR)
``GET /health``
"""
import flask

from glassbox.core import GlassBox
from glassbox.hardware.qr import persona_from_zbar

__all__ = ['create_app']


def create_app(glassbox=None, config=None):
    """Creates the Flask application."""
    box = glassbox or GlassBox(config)
    app = flask.Flask(__name__)
    app.config['GLASSBOX'] = box

    @app.route('/')
    def index():
        box.page_loaded()
        return flask.render_template(
            'chat.html',
            poll_interval=int(1000 * box.config['app']['poll_interval']))

    @app.route('/ask', methods=['POST'])
    def ask():
        data = flask.request.get_json(silent=True) or {}
        message = data.get('message') or flask.request.form.get(
            'messageText', '')
        message = message.strip().strip('.?!')
        if not message:
            return flask.jsonify({'status': 'NOT-OK', 'answer': ''}), 400
        return flask.jsonify({'status': 'OK',
                              'answer': box.handle_message(message)})

    @app.route('/voice', methods=['POST'])
    @app.route('/voicehat', methods=['POST'])
    def voice():
        data = flask.request.get_json(silent=True) or {}
        text = data.get('text', '')
        if not text:
            return flask.jsonify({'status': 'NOT-OK', 'text': ''}), 400
        return flask.jsonify({'status': 'OK', 'text': box.handle_voice(text)})

    @app.route('/qrcode', methods=['POST'])
    def qrcode():
        data = flask.request.get_json(silent=True) or {}
        persona = data.get('persona') or persona_from_zbar(data.get('qr', ''))
        if not persona:
            return flask.jsonify({'status': 'NOT-OK'}), 400
        box.handle_qr(persona)
        return flask.jsonify({'status': 'OK'})

    @app.route('/messages')
    def messages():
        return flask.jsonify({'status': 'OK', 'messages': box.pop_messages()})

    @app.route('/getmessage')
    def get_message():  # the original polling endpoint (one message)
        answer = box.messages.popleft() if box.messages else ''
        return flask.jsonify({'status': 'OK', 'answer': answer})

    @app.route('/health')
    def health():
        return flask.jsonify({'status': 'OK'})

    return app
