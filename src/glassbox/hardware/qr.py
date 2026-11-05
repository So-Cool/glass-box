"""
QR code scanner (Raspberry Pi camera + ``zbarcam``).

Behaviour: the application cards carry the persona id (e.g., ``04``);
``zbarcam`` prints ``QR-Code:04`` for every scan.
"""
import logging
import subprocess

from glassbox.hardware._http import post_json

__all__ = ['ZbarQRScanner', 'persona_from_zbar']

logger = logging.getLogger(__name__)


def persona_from_zbar(line):
    """Extracts the persona id from a ``zbarcam`` output line."""
    line = line.strip()
    if line.startswith('QR-Code:'):
        line = line[len('QR-Code:'):]
    return line.strip(':').strip()


class ZbarQRScanner(object):  # pragma: no cover (hardware)
    """
    Reads QR codes with ``zbarcam`` and reports the persona ids.

    Parameters
    ----------
    handle_persona : callable
        Called with every scanned persona id.
    device : str
        The camera device (``/dev/video0``).
    """

    def __init__(self, handle_persona, device='/dev/video0',
                 prescale='320x240'):
        self.handle_persona = handle_persona
        self.command = ['zbarcam', device, '--prescale=' + prescale,
                        '--nodisplay']

    @classmethod
    def posting_to(cls, app_url, **kwargs):
        """A scanner sending the ids to ``<app_url>/qrcode``."""
        url = app_url.rstrip('/') + '/qrcode'

        def handle(persona):
            if post_json(url, {'persona': persona}, timeout=5) is None:
                logger.warning('Host unreachable: %s', url)
        return cls(handle, **kwargs)

    def run(self):
        process = subprocess.Popen(self.command, stdout=subprocess.PIPE,
                                   universal_newlines=True)
        try:
            for line in process.stdout:
                persona = persona_from_zbar(line)
                if persona:
                    logger.info('Scanned persona %s', persona)
                    self.handle_persona(persona)
        finally:
            process.terminate()
