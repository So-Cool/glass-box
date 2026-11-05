"""Helpers shared by the glassbox tests."""
import json
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def load_json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)
