"""Maps NLU parses to dialogue intents and arguments."""
import json
import re

from glassbox.dialogue.templates import APPLICANT_NAMES, NUMBERS
from glassbox.resources import resource_path

__all__ = ['Intentioner', 'decode_feature',
           'decode_persona', 'feature_synonyms']

INTENT_MAP = {"greet": "hello", "apply": "apply", "myself": "myself",
              "outcome": "outcome", "fairness": "fairness",
              "action": "action", "why": "why", "help": "help",
              "repeat": "repeat", "farewell": "bye"}


def decode_persona(persona):
    """Turns a number, a number word or a name into a persona id."""
    try:
        return "{0:02d}".format(int(persona))
    except ValueError:
        pass
    if persona in NUMBERS:
        return NUMBERS[persona]
    for i, name in APPLICANT_NAMES.items():
        if name.lower() == persona.lower():
            return i
    return persona


_SYNONYMS = None


def feature_synonyms():
    """Feature mentions annotated in the NLU training data (lower case)."""
    global _SYNONYMS
    if _SYNONYMS is None:
        examples = []
        for name in ('training_data.json', 'training_data_extra.json'):
            with open(resource_path('nlu', name)) as f:
                examples += json.load(f)['rasa_nlu_data']['common_examples']
        _SYNONYMS = {}
        for e in examples:
            for en in e.get('entities', []):
                if en['entity'] == 'feature':
                    span = e['text'][en['start']:en['end']].lower().strip()
                    _SYNONYMS.setdefault(span, en['value'])
    return _SYNONYMS


# (feature id, word-boundary patterns) checked in order -- the more specific
# patterns first
_KEYWORDS = [
    ("A17", [r"\bjob", r"\bemploy\w* type", r"\boccupation", r"\bprofession",
             r"\bskill"]),
    ("A7", [r"\bemploy", r"\bunemploy", r"\bwork(ed|ing)? for"]),
    ("A1", [r"\bcurrent account", r"\bchecking", r"\bcurrent\b"]),
    ("A6", [r"\bsaving"]),
    ("A8", [r"\binstal+ment", r"\brate\b", r"\binterest"]),
    ("A3", [r"\bhistory", r"\bpaid back", r"\brepa(y|id)"]),
    ("A2", [r"\bduration", r"\bhow long", r"\bmonths?\b", r"\bterm\b"]),
    ("A5", [r"\bamount", r"\bhow much", r"\bmoney", r"\bborrow",
            r"\bpounds?\b", r"\bsum\b"]),
    ("A9", [r"\bmarital", r"\bmarried", r"\bmarriage", r"\bsingle",
            r"\bdivorced", r"\bseparated", r"\bwidow", r"\bgender",
            r"\bsex\b", r"\bmale", r"\bfemale", r"\bman\b", r"\bwoman\b"]),
    ("A13", [r"\bage\b", r"\bold\b", r"\bolder", r"\byounger", r"\byears?\b",
             r"\bborn"]),
    ("A14", [r"\bother loans?", r"\bother credits?", r"\bloans? elsewhere",
             r"\bother"]),
    ("A15", [r"\bhous", r"\bhome", r"\brent", r"\bown\b", r"\bowner",
             r"\bcouncil", r"\blive", r"\bliving"]),
    ("A20", [r"\bforeign", r"\bmigrant", r"\bimmigra", r"\bworker",
             r"\bnationality", r"\bcitizen"]),
]


def decode_feature(feature_name):
    """
    Recognises a feature mentioned in a sentence: the mentions annotated in
    the NLU training data, then word-boundary keywords.
    """
    text = feature_name.lower().strip()
    synonyms = feature_synonyms()
    if text in synonyms:
        return synonyms[text]
    for feature_id, patterns in _KEYWORDS:
        if any(re.search(p, text) for p in patterns):
            return feature_id
    return None


class Intentioner(object):
    """
    Turns NLU parses into ``(intent, args)`` pairs.

    Parameters
    ----------
    nlu : glassbox.nlu.NLUBackend, optional
        Needed by :meth:`sentence_to_intent`.

    Compared with the original implementation, features are recognised with
    the mentions annotated in the training data and word-boundary keywords
    (the original substring rules missed, e.g., "married", "rent" and
    "employed" and mapped "separated" to the instalment rate), the
    ``why_reset`` entity ("restart why") restarts the explanations and "what
    can I do" requests carry restrictions like "why" requests.
    """

    def __init__(self, nlu=None):
        self.nlu = nlu

    def decode_feature(self, feature_name):
        return decode_feature(feature_name)

    def sentence_to_intent(self, sentence):
        """Parses a sentence with the NLU backend and maps it."""
        if self.nlu is None:
            raise RuntimeError('No NLU backend was given.')
        return self.parse_to_intent(self.nlu.parse(sentence))

    def parse_to_intent(self, parsed):
        """Maps a parse (see :class:`glassbox.nlu.NLUBackend`)."""
        args = {}
        intent = parsed.get("intent", {}) or {}
        intent = intent.get("name", None)
        entities = parsed.get("entities", [])
        entities_names = list(set([i.get("entity") for i in entities
                                   if i.get("entity") is not None]))

        if intent == "apply":
            if "persona" in entities_names:
                for i in entities:
                    if i.get("entity", "") == "persona":
                        persona = i.get("value", "")
                        args["persona"] = (decode_persona(persona) if persona
                                           else "")
                        break
        elif intent == "myself":
            features = []
            for i in entities:
                if i.get("entity", "") == "feature" and i.get("value", ""):
                    value_id = self.decode_feature(i["value"])
                    if value_id is not None:
                        features.append(value_id)
            if features:
                args["features"] = features
        elif intent == "why":
            args = self._why_arguments(entities, entities_names)
        elif intent == "action":
            args = self._why_arguments(entities, entities_names)
            args.pop("restart", None)
            if not args:  # "what can I do with my X" -> change X
                features = [self.decode_feature(i["value"]) for i in entities
                            if i.get("entity", "") == "feature"
                            and i.get("value", "")]
                features = [f for f in features if f is not None]
                if features:
                    args["features_require"] = features

        return INTENT_MAP.get(intent, "unknown"), args

    def _why_arguments(self, entities, entities_names):
        args = {}
        # collect all features and their location
        features = []
        for i in entities:
            if i.get("entity", "") == "feature" and i.get("value", ""):
                value_id = self.decode_feature(i["value"])
                if value_id is not None:
                    features.append((i.get("start", None), value_id))

        def location(name):
            for i in entities:
                if i.get("entity", "") == name:
                    return i.get("start", None)
            return None
        why_despite_location = location("why_despite")
        why_given_location = location("why_given")

        features_drop, features_require = [], []
        if why_despite_location is not None and why_given_location is None:
            features_drop = [f for _, f in features]
        elif why_despite_location is None and why_given_location is not None:
            features_require = [f for _, f in features]
        elif why_despite_location is not None:
            given_first = why_given_location - why_despite_location < 0
            for f_loc, f in features:
                if given_first:
                    if why_given_location < f_loc < why_despite_location:
                        features_require.append(f)
                    else:
                        features_drop.append(f)
                else:
                    if why_despite_location < f_loc < why_given_location:
                        features_drop.append(f)
                    else:
                        features_require.append(f)

        if features_require:
            args["features_require"] = features_require
        if features_drop:
            args["features_drop"] = features_drop
        if "why_reset" in entities_names:
            args["restart"] = True
        return args
