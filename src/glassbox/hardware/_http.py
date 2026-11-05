"""Minimal JSON-over-HTTP helper (standard library only)."""
import json
import logging
import urllib.request

logger = logging.getLogger(__name__)


def post_json(url, payload, timeout=1.0):
    """POSTs JSON and returns the decoded JSON reply (``None`` on failure)."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode('utf-8')
        return json.loads(body) if body else {}
    except Exception as e:  # hardware is best effort
        logger.debug('POST %s failed: %s', url, e)
        return None
