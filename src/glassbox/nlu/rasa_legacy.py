"""The original Rasa NLU 0.12.3 model served over HTTP (optional backend)."""
import json
import urllib.request

from glassbox.nlu.base import NLUBackend

__all__ = ['RasaLegacyNLU']


class RasaLegacyNLU(NLUBackend):
    """
    Client of the Rasa NLU 0.12 HTTP server (``POST /parse``).

    Parameters
    ----------
    url : str
        Address of the server.
    project, model : str
        Rasa NLU project and model names (``default``/``chat`` in the image).
    timeout : float
        Request timeout in seconds.
    """

    def __init__(self, url='http://localhost:5005', project='default',
                 model='chat', timeout=10.):
        self.url = url.rstrip('/')
        self.project = project
        self.model = model
        self.timeout = timeout

    def parse(self, text):
        payload = json.dumps({'q': text, 'project': self.project,
                              'model': self.model}).encode('utf-8')
        request = urllib.request.Request(
            self.url + '/parse', data=payload,
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request,
                                        timeout=self.timeout) as response:
                parsed = json.loads(response.read().decode('utf-8'))
        except OSError as e:
            raise ConnectionError('The legacy Rasa NLU server at {} is not '
                                  'reachable -- is the Docker container '
                                  'running?'.format(self.url)) from e
        parsed.setdefault('text', text)
        return parsed
