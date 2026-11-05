"""Access to the data files shipped with the package."""
import json
import os

__all__ = ['resource_path', 'load_json']

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'resources')


def resource_path(*parts):
    """Absolute path of a packaged resource."""
    return os.path.join(RESOURCES, *parts)


def load_json(*parts):
    with open(resource_path(*parts)) as f:
        return json.load(f)
