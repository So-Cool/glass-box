"""
The LED ring and the 8x8 LED matrix ("square") of the device.

The dialogue reports *events*; :data:`EVENT_PATTERNS` maps them to the LED
commands. A command is a dictionary understood by the device server's ``/led``
endpoint::

    {"led_matrix": "ring" | "square", "pattern": ..., "params": {...},
     "brightness": 0-255, "pause": ..., "unpause": ..., "rotate": ...}
"""
import abc
import logging

from glassbox.hardware._http import post_json

__all__ = ['LEDController', 'NeoPixelLEDs', 'RemoteLEDs', 'MockLEDs',
           'DisabledLEDs', 'create_leds', 'EVENT_PATTERNS', 'event_commands']

logger = logging.getLogger(__name__)
GREEN, RED, YELLOW, BLUE = 65280, 16711680, 16776960, 255


def _ascii(picture, colour):
    return {'led_matrix': 'square', 'pattern': 'ascii',
            'params': {'ascii_pic': picture, 'colour': colour,
                       'scroll': False}}


#: LED commands of the original implementation for every event.
EVENT_PATTERNS = {
    'page_loaded': [{'led_matrix': 'ring', 'pattern': 'pulse'},
                    {'led_matrix': 'square', 'pattern': 'pulse'}],
    'thinking': [{'led_matrix': 'ring', 'pattern': 'rainbow',
                  'params': {'clockwise': True}},
                 {'led_matrix': 'square', 'pattern': 'rainbow'}],
    'idle': [{'led_matrix': 'ring', 'pattern': 'still'},
             {'led_matrix': 'square', 'pattern': 'still'}],
    'hello_again': [{'led_matrix': 'ring', 'pattern': 'still'},
                    {'led_matrix': 'square', 'pattern': 'still'}],
    'goodbye': [{'led_matrix': 'ring', 'pattern': 'pulse'},
                {'led_matrix': 'square', 'pattern': 'pulse'}],
    'processing': [{'led_matrix': 'ring', 'pattern': 'rainbow',
                    'params': {'clockwise': True}}],
    'processed': [{'led_matrix': 'ring', 'pattern': 'still'}],
    'action': [{'led_matrix': 'square', 'pattern': 'bucket'}],
    'why': [_ascii('why', GREEN)],
    'myself': [_ascii('me', YELLOW)],
    'command': [{'led_matrix': 'square', 'pattern': 'drop'}],
}


def event_commands(event, params=None):
    """LED commands for an event (handles the outcome-dependent ones)."""
    params = params or {}
    if event == 'outcome':
        good = params.get('outcome') == 'good'
        return [_ascii('yes' if good else 'no', GREEN if good else RED)]
    if event == 'fairness':
        colour = BLUE if params.get('indirect') else YELLOW
        good = params.get('outcome') == 'good'
        return [_ascii('yes' if good else 'no', colour)]
    return EVENT_PATTERNS.get(event, [])


class LEDController(abc.ABC):
    """Shows dialogue events on the LEDs."""

    #: Whether the LEDs can actually be seen (used for the "thinking" pause).
    visible = True

    def show(self, event, params=None):
        for command in event_commands(event, params):
            self.send(command)

    @abc.abstractmethod
    def send(self, command):
        """Executes an LED command."""

    def close(self):
        pass


class DisabledLEDs(LEDController):
    visible = False

    def send(self, command):
        pass


class MockLEDs(LEDController):
    """Logs LED commands (and keeps the last command of every matrix)."""
    visible = False

    def __init__(self):
        self.state = {}
        self.history = []

    def send(self, command):
        self.history.append(command)
        self.state[command.get('led_matrix')] = command
        logger.info('LED %s: %s %s', command.get('led_matrix'),
                    command.get('pattern'), command.get('params', ''))


class RemoteLEDs(LEDController):
    """Sends LED commands to the device server (``POST <url>/led``)."""

    def __init__(self, url, timeout=0.1):
        self.url = url.rstrip('/') + '/led'
        self.timeout = timeout

    def send(self, command):
        post_json(self.url, command, timeout=self.timeout)


class NeoPixelLEDs(LEDController):
    """
    Drives the WS281x LEDs with ``neopixel_python`` (Raspberry Pi only).

    Parameters
    ----------
    ring, square : dict
        Keyword arguments of ``neopixel_python.ring.RingLED`` and
        ``neopixel_python.square.SquareLED`` (e.g., ``led_dma``,
        ``brightness``).
    """

    def __init__(self, ring=None, square=None):
        try:
            from neopixel_python.ring import RingLED
            from neopixel_python.square import SquareLED
        except ImportError as e:  # pragma: no cover (hardware)
            raise ImportError('The neopixel LED backend requires '
                              'neopixel_python: pip install '
                              '"glassbox[pi]".') from e
        defaults = {'pattern': 'none', 'led_dma': 11, 'brightness': 165}
        self.matrices = {
            'ring': RingLED(**dict(defaults, **(ring or {}))),
            'square': SquareLED(**dict(defaults, **(square or {})))}
        for m in self.matrices.values():
            m.start()

    def send(self, command):  # pragma: no cover (hardware)
        apply_command(self.matrices, command)

    def close(self):  # pragma: no cover (hardware)
        for m in self.matrices.values():
            m.stop()


def apply_command(matrices, command):
    """Executes an LED command on ``{'ring': ..., 'square': ...}`` drivers."""
    matrix = matrices.get(command.get('led_matrix'))
    if matrix is None:
        return False
    if command.get('pattern') is not None:
        matrix.switch_pattern(command['pattern'],
                              **(command.get('params') or {}))
    try:
        brightness = int(command.get('brightness'))
        if 0 <= brightness <= 255:
            matrix.set_brightness(brightness)
    except (TypeError, ValueError):
        pass
    if command.get('pause') is not None:
        matrix.pause()
    if command.get('unpause') is not None:
        matrix.unpause()
    try:
        matrix.rotate(int(command.get('rotate')))
    except (TypeError, ValueError):
        pass
    return True


def create_leds(config):
    """Creates the LED backend from the ``[hardware.leds]`` configuration."""
    backend = config.get('backend', 'disabled')
    if backend == 'disabled':
        return DisabledLEDs()
    if backend == 'mock':
        return MockLEDs()
    if backend == 'remote':
        return RemoteLEDs(config['url'], timeout=config.get('timeout', 0.1))
    if backend == 'neopixel':
        return NeoPixelLEDs(config.get('ring'), config.get('square'))
    raise ValueError('Unknown LED backend: {!r}.'.format(backend))
