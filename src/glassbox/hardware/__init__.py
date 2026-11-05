"""
Glass-Box hardware: voice input, speech output, QR code scanner and LEDs.

Every component has interchangeable backends selected in the configuration:

=========  =====================================================  ==========
component  backends                                               runs on
=========  =====================================================  ==========
LEDs       ``neopixel`` (local), ``remote`` (HTTP), ``mock``,     Pi / any
           ``disabled``
speech     ``aiy`` (local), ``remote`` (HTTP), ``mock``,          Pi / any
           ``disabled``
voice      ``aiy`` (button + Google Assistant speech recognition)  Pi
QR codes   ``zbar`` (``zbarcam`` and a camera)                     Pi
=========  =====================================================  ==========

``remote`` backends talk to the device server (``glassbox device``) running on
the Raspberry Pi; ``mock`` backends log what the hardware would do, which lets
the system run on a laptop without any hardware.
"""
from glassbox.hardware.leds import LEDController, create_leds
from glassbox.hardware.speech import Speaker, create_speaker

__all__ = ['LEDController', 'create_leds', 'Speaker', 'create_speaker']
