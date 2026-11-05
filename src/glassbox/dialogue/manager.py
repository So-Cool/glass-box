"""
The dialogue manager: a state machine answering the applicant's requests.

 Requests are *intents* with arguments (produced by
 :class:`glassbox.nlu.Intentioner` or the ``cmd:`` syntax):

``hello``, ``bye``, ``apply`` (``persona``), ``outcome``, ``help``,
``fairness``, ``action``, ``why`` (``features_require``, ``features_drop``,
``restart``), ``myself`` (``features``), ``repeat``.

With ``implementation='original'`` (and the explainer's ``deterministic_order``
fix) the answers are identical to the original implementation.

Hardware feedback (the LEDs of the original device) is reported to an optional
``on_event`` callback -- see :mod:`glassbox.hardware.leds` for the events.
"""
import logging
import random
import re

from glassbox.dialogue.fixes import resolve_dialogue_fixes
from glassbox.dialogue.templates import (APPLICANT_NAMES, FEATURE_ACTIONABLE,
                                         FEATURE_UNFAIR, OUTCOME_MAP,
                                         Templates)
from glassbox.model import GlassBoxModel

__all__ = ['DialogueManager']

logger = logging.getLogger(__name__)
INTENTS = ('hello', 'bye', 'apply', 'outcome', 'help', 'fairness', 'action',
           'why', 'myself', 'repeat')


def _tidy(text):
    """Collapses spaces, capitalises and ends a sentence (``wording``)."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r' ([,.])', r'\1', text)
    text = re.sub(r'([.?!]) ([a-z])',
                  lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.?!)':
        text += '.'
    return text


class DialogueManager(object):
    """
    Answers loan applicants' requests with counterfactual explanations.

    Parameters
    ----------
    model : GlassBoxModel, optional
        The model (default: the original Glass-Box tree, with the explainer
        configured by ``implementation`` and ``explainer_fixes``).
    implementation : {'original', 'fixed'}
        Default behaviour of all the dialogue switches (see
        :mod:`glassbox.dialogue.fixes`).
    fixes : dict, optional
        Overrides of individual dialogue switches.
    explainer_fixes : dict, optional
        Overrides of the counterfactual explainer switches (see
        :class:`ctreex.Fixes`); used when ``model`` is not given.
    on_event : callable, optional
        Called with ``(event, params)`` for hardware feedback.
    rng : random.Random, optional
        Source of randomness for the choice of sentence templates (default:
        the global ``random`` module).
    """

    def __init__(self, model=None, implementation='fixed', fixes=None,
                 explainer_fixes=None, on_event=None, rng=None):
        self.implementation = implementation
        self.fix_overrides = dict(fixes or {})
        resolve_dialogue_fixes(implementation, self.fix_overrides)  # validate
        if model is None:
            model = GlassBoxModel(implementation=implementation,
                                  fixes=explainer_fixes)
        self.model = model
        self.on_event = on_event
        self.rng = random if rng is None else rng
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self):
        """Forgets the conversation."""
        self.conversation_started = False
        self.last_said = None
        self.applied = False
        self.persona = None
        self.outcome = None
        self.contrastive_per_leaf = None
        self.contrastive_per_leaf_transformed = None
        self.contrastive_per_feature = None
        self.unfairness = []
        self.actionability = []
        self.why = []
        self.features_drop = []
        self.features_require = []
        self.action_drop = []
        self.action_require = []
        self._exhausted = {'why': False, 'fairness': False, 'action': False}

    def fixes(self, channel='chat'):
        """The dialogue switches used for a channel."""
        return resolve_dialogue_fixes(self.implementation, self.fix_overrides,
                                      channel)

    def _event(self, name, **params):
        if self.on_event is not None:
            try:
                self.on_event(name, params)
            except Exception:  # hardware feedback must never break dialogue
                logger.exception('Event handler failed for %s.', name)

    def _say(self, text, fixes):
        if fixes.wording and text:
            text = _tidy(text)
        self.last_said = text
        return text

    # ------------------------------------------------------------- language
    def _range_text(self, ranges):
        return ' or '.join(ranges)

    def get_statement(self, feature_name, feature_value, templates):
        """Describes the applicant's value of a feature."""
        text_map = templates.feature_talk[feature_name]
        if "property" in text_map:  # numerical
            count = "singular" if feature_value == 1 else "plural"
            return text_map["part1"].format(
                value=feature_value, value_type=text_map["property"][count])
        if "value" in text_map:  # categorical binary
            return text_map["part1"].format(
                value=text_map["value"][feature_value])
        if "value_hypothetical1" in text_map:  # different tense
            a = "" if feature_value == "council" else "a"
            return text_map["part1"].format(
                value=text_map["value_hypothetical1"][feature_value], a=a)
        if "map" in text_map:  # categorical with a simple map
            return text_map["part1"].format(value=feature_value)
        return "COULD NOT GENERATE A STATEMENT FOR **{}**".format(
            feature_name)

    def get_counterfactual(self, cf_dictionary, templates, fixes):
        """Phrases a counterfactual (``{feature: {'is', 'wannabe'}}``)."""
        t = templates
        translated = self.model.describe(cf_dictionary,
                                         integer_ranges=fixes.integer_ranges)
        comp = [t.COUNTERFACTUAL_TEMPLATE_COMP_1,
                t.COUNTERFACTUAL_TEMPLATE_COMP_2,
                t.COUNTERFACTUAL_TEMPLATE_COMP_3]
        plain = [t.COUNTERFACTUAL_TEMPLATE_1, t.COUNTERFACTUAL_TEMPLATE_2,
                 t.COUNTERFACTUAL_TEMPLATE_3]
        sentences = []
        for feature, cf in translated.items():
            talk = t.feature_talk[feature]
            if "property" in talk:  # numerical
                choice = self.rng.choice(comp)
                current = cf["is"][0]
                count = "singular" if current == 1 else "plural"
                current = talk["part2"].format(
                    value=current, value_type=talk["property"][count])
                hypothetical = talk["part1"].format(
                    value=self._range_text(cf["wannabe"]),
                    value_type=talk["property"]["plural"])
                sentences.append(choice.format(
                    hypothetical_state=hypothetical, current_state=current))
            elif "value" in talk:  # categorical binary
                choice = self.rng.choice(plain)
                hypothetical = talk["part1"].format(
                    value=talk["value"][cf["wannabe"][0]])
                sentences.append(choice.format(
                    hypothetical_state=hypothetical))
            elif "value_hypothetical1" in talk:  # different tense
                choice = self.rng.choice(comp)
                value = cf["is"][0]
                a = "" if value == "council" else "a"
                current = talk["part2"].format(
                    value=talk["value_hypothetical2"][value], a=a)
                options = [talk["value_hypothetical1"][i]
                           for i in cf["wannabe"]]
                if fixes.wording:  # the article of every option
                    hypothetical = ' or '.join(
                        talk["part1"].format(
                            value=v, a="" if i == "council" else "a")
                        for v, i in zip(options, cf["wannabe"]))
                else:  # article chosen from the last option only
                    a = "" if cf["wannabe"][-1] == "council" else "a"
                    hypothetical = talk["part1"].format(
                        value=' or '.join(options), a=a)
                sentences.append(choice.format(
                    hypothetical_state=hypothetical, current_state=current))
            elif "map" in talk:  # categorical with a simple map
                choice = self.rng.choice(comp)
                current = talk["part2"].format(value=cf["is"][0])
                hypothetical = talk["part1"].format(
                    value=' or '.join(cf["wannabe"]))
                sentences.append(choice.format(
                    hypothetical_state=hypothetical, current_state=current))
            else:
                sentences.append("COULD NOT GENERATE A COUNTERFACTUAL FOR "
                                 "**{}**".format(feature))
        return ' and '.join(sentences)

    def outcome_complement(self, current_outcome):
        for i in OUTCOME_MAP:
            if i != current_outcome:
                return i
        return "**unknown**"

    # ------------------------------------------------------------- helpers
    def _next(self, name, build, fixes):
        """
        Pops the next item of the ``name`` queue, rebuilding it when empty.

        Returns ``(item, rebuilt)``; ``item`` is ``None`` when nothing is
        available or -- with ``stop_when_exhausted`` -- when the queue has
        just run out (then ``'exhausted'`` is returned).
        """
        attribute = {'fairness': 'unfairness', 'action': 'actionability',
                     'why': 'why'}[name]
        queue = getattr(self, attribute)
        if queue:
            item = queue.pop(0)
            return item, False
        if fixes.stop_when_exhausted and self._exhausted[name]:
            self._exhausted[name] = False
            return 'exhausted', False
        queue = build()
        setattr(self, attribute, queue)
        self._exhausted[name] = True
        if queue:
            return queue.pop(0), True
        self._exhausted[name] = False
        return None, True

    def _build_unfairness(self):
        unfairness = []
        for dist in sorted(self.contrastive_per_feature.keys()):
            current_keys = self.contrastive_per_feature[dist].keys()
            to_copy = []
            for i in FEATURE_UNFAIR:
                if i in current_keys:
                    to_copy += self.contrastive_per_feature[dist][i]
            to_copy = list(set(to_copy))
            for i in to_copy:
                unfairness.append((dist, self.contrastive_per_leaf[dist][i]))
        return unfairness

    def _build_actionability(self):
        actionability = []
        # all the features of a counterfactual have to be actionable
        for dist in sorted(self.contrastive_per_leaf_transformed.keys()):
            for point in self.contrastive_per_leaf_transformed[dist].values():
                if not all(f in FEATURE_ACTIONABLE for f in point):
                    continue
                # restrictions (action_restrictions fix; empty otherwise)
                if any(f not in point for f in self.action_require):
                    continue
                if any(f in point for f in self.action_drop):
                    continue
                actionability.append((dist, point))
        return actionability

    def _restrictions_note(self, require, drop, point, t):
        """The "(When considering ... and avoiding ...)" note."""
        name = self.model.feature_name
        regard = " and ".join(t.feature_talk[name(i)]["name"]
                              for i in require if i in point)
        disregard = " and ".join(t.feature_talk[name(i)]["name"]
                                 for i in drop)
        if regard and disregard:
            return " (When considering: {} and avoiding: {}.)".format(
                regard, disregard)
        if regard:
            return " (When considering using: {}.)".format(regard)
        if disregard:
            return " (When avoiding using: {}.)".format(disregard)
        return ""

    def _build_why(self):
        why = []
        for dist in sorted(self.contrastive_per_leaf_transformed.keys()):
            for point in self.contrastive_per_leaf_transformed[dist].values():
                if any(f not in point for f in self.features_require):
                    continue
                if any(f in point for f in self.features_drop):
                    continue
                why.append((dist, point))
        return why

    # ---------------------------------------------------------------- intents
    def process_intent(self, intent, args=None, channel='chat'):
        """
        Answers a request.

        Parameters
        ----------
        intent : str
            One of the intents listed in the module documentation (any other
            value is answered as an unknown request).
        args : dict, optional
            Intent arguments.
        channel : {'chat', 'speech'}
            The output channel (selects the channel-dependent fixes).

        Returns
        -------
        str
            The answer.
        """
        args = {} if args is None else args
        fixes = self.fixes(channel)
        t = Templates(fixed=fixes.wording)
        handler = getattr(self, '_intent_' + intent, None) \
            if intent in INTENTS else None
        if handler is None:
            return self._intent_unknown(args, fixes, t)
        return handler(args, fixes, t)

    def _intent_hello(self, args, fixes, t):
        if self.conversation_started and not self.applied:
            self._event('hello_again')
            return self._say(t.GREET + " " + t.HOW_TO_APPLY, fixes)
        if self.conversation_started and self.applied:
            return self._say(t.ALREADY_APPLIED_HELLO, fixes)
        self.conversation_started = True
        return self._say(t.GREET + " " + t.HOW_TO_APPLY, fixes)

    def _intent_bye(self, args, fixes, t):
        if self.conversation_started:
            self._event('goodbye')
            self.reset()
            return t.BYE
        return t.BYE_NOT_APPLIED

    def _intent_apply(self, args, fixes, t):
        if self.conversation_started and not self.applied:
            self._event('processing')
            persona = args.get("persona", None)
            if persona is None:
                return self._say(t.NOT_ON_RECORD.format(
                    list(APPLICANT_NAMES.keys())), fixes)
            if persona not in APPLICANT_NAMES:
                return self._say(t.UNKNOWN_PERSON.format(
                    persona, list(APPLICANT_NAMES.keys())), fixes)
            name = APPLICANT_NAMES[persona]
            self.persona = persona
            response = t.APPLICATION_RECEIVED.format(name=name)
            response += " " + t.APPLICATION_PROCESSED
            self.applied = True
            self._explain()
            self._event('processed')
            return self._say(response, fixes)
        if self.conversation_started and self.applied:
            persona = args.get("persona", None)
            if persona is None or persona == self.persona:
                return self._say(t.ALREADY_APPLIED, fixes)
            if persona in APPLICANT_NAMES:
                return self._say(t.DIFFERENT_PERSON.format(
                    APPLICANT_NAMES[self.persona], APPLICANT_NAMES[persona]),
                    fixes)
            return self._say(t.DIFFERENT_UNKNOWN_PERSON.format(
                APPLICANT_NAMES[self.persona], persona), fixes)
        return self._say(t.SAY_HELLO, fixes)

    def _explain(self):
        """Computes the counterfactuals of the applicant."""
        values = self.model.personas[self.persona].values
        explanation = self.model.explain(values)
        self.outcome = explanation.prediction
        self.contrastive_per_leaf = {}
        self.contrastive_per_leaf_transformed = {}
        self.contrastive_per_feature = {}
        for found in explanation.counterfactuals:
            self.contrastive_per_leaf[found.distance] = found.per_leaf
            self.contrastive_per_leaf_transformed[found.distance] = \
                found.ranked
            self.contrastive_per_feature[found.distance] = found.per_feature

    def _intent_outcome(self, args, fixes, t):
        if self.conversation_started and self.applied:
            self._event('outcome', outcome=self.outcome)
            return self._say(t.APPLICATION_OUTCOME.format(
                outcome=OUTCOME_MAP[self.outcome]), fixes)
        if self.conversation_started:
            return self._say(t.HOW_TO_APPLY, fixes)
        return self._say(t.SAY_HELLO, fixes)

    def _intent_help(self, args, fixes, t):
        if self.conversation_started and not self.applied:
            return self._say(t.HOW_TO_APPLY, fixes)
        if self.conversation_started:
            return self._say(t.HELP_APPLIED, fixes)
        return self._say(t.SAY_HELLO, fixes)

    def _intent_fairness(self, args, fixes, t):
        if self.conversation_started and not self.applied:
            return self._say(t.HOW_TO_APPLY, fixes)
        if not self.conversation_started:
            return self._say(t.SAY_HELLO, fixes)
        item, _ = self._next('fairness', self._build_unfairness, fixes)
        if item == 'exhausted':
            return self._say(t.NO_MORE['fairness'], fixes)
        if item is None:
            return self._say(t.FAIR, fixes)
        unfair_dist, unfair_first = item
        cf = self.get_counterfactual(unfair_first, t, fixes)
        template = t.FAIRNESS_INDIRECT if unfair_dist > 1 else t.FAIRNESS
        self._event('fairness', outcome=self.outcome,
                    indirect=unfair_dist > 1)
        return self._say(template.format(fairness="unfair",
                                         counterfactual=cf), fixes)

    def _intent_action(self, args, fixes, t):
        if self.conversation_started and not self.applied:
            return self._say(t.HOW_TO_APPLY, fixes)
        if not self.conversation_started:
            return self._say(t.SAY_HELLO, fixes)
        if fixes.action_restrictions:
            require = args.get("features_require", [])
            drop = args.get("features_drop", [])
            if (set(require) != set(self.action_require)
                    or set(drop) != set(self.action_drop)):
                # new restrictions: start a new list of suggestions
                self.action_require, self.action_drop = require, drop
                self.actionability = []
                self._exhausted['action'] = False
        item, _ = self._next('action', self._build_actionability, fixes)
        if item == 'exhausted':
            return self._say(t.NO_MORE['action'], fixes)
        if item is None:
            return self._say(t.NO_ACTION.format(OUTCOME_MAP[self.outcome]),
                             fixes)
        self._event('action')
        _, action_first = item
        cf = self.get_counterfactual(action_first, t, fixes)
        answer = t.ACTIONABILITY.format(
            current_outcome=OUTCOME_MAP[self.outcome],
            wannabe_outcome=OUTCOME_MAP[self.outcome_complement(self.outcome)],
            counterfactual=cf)
        if fixes.action_restrictions:
            answer += self._restrictions_note(
                self.action_require, self.action_drop, action_first, t)
        return self._say(answer, fixes)

    def _intent_why(self, args, fixes, t):
        if self.conversation_started and not self.applied:
            return self._say(t.HOW_TO_APPLY, fixes)
        if not self.conversation_started:
            return self._say(t.SAY_HELLO, fixes)
        restart = args.get("restart", False)
        features_drop = args.get("features_drop", [])
        features_require = args.get("features_require", [])
        if not features_require and not features_drop:
            cont = True
        else:
            cont = (set(features_require) == set(self.features_require)
                    and set(features_drop) == set(self.features_drop))

        note = ""
        if restart or not self.why or not cont:  # rebuild the list
            if (fixes.stop_when_exhausted and not restart and cont
                    and not self.why and self._exhausted['why']):
                self._exhausted['why'] = False
                self._event('why')
                return self._say(t.NO_MORE['why'], fixes)
            self.features_drop = features_drop
            self.features_require = features_require
            self.why = self._build_why()
            self._exhausted['why'] = True
            if self.why:
                why_dist, why_first = self.why.pop(0)
                note = t.NO_ENQUIRY
            else:
                why_first = None
                self._exhausted['why'] = False
        else:
            why_dist, why_first = self.why.pop(0)

        if why_first is None:
            answer = t.NO_COUNTERFACTUALS
        else:
            template = self.rng.choice([t.COUNTERFACTUAL_STATEMENT_1,
                                        t.COUNTERFACTUAL_STATEMENT_2])
            intro = self.rng.choice(["", t.APPLICATION_OUTCOME.format(
                outcome=OUTCOME_MAP[self.outcome])])
            outro = t.COUNTERFACTUAL_OUTCOME.format(
                outcome=OUTCOME_MAP[self.outcome_complement(self.outcome)])
            cf = self.get_counterfactual(why_first, t, fixes)

            # mention the restrictions imposed
            name = self.model.feature_name
            regard = " and ".join(t.feature_talk[name(i)]["name"]
                                  for i in self.features_require
                                  if i in why_first)
            disregard = " and ".join(t.feature_talk[name(i)]["name"]
                                     for i in self.features_drop)
            if regard and disregard:
                outro += ". (When considering: {} and avoiding: {}.)".format(
                    regard, disregard)
            elif regard:
                outro += ". (When considering using: {}.)".format(regard)
            elif disregard:
                outro += ". (When avoiding using: {}.)".format(disregard)
            answer = note + template.format(intro=intro,
                                            counterfactual_statement=cf,
                                            counterfactual_outcome=outro)
        self._event('why')
        return self._say(answer, fixes)

    def _intent_myself(self, args, fixes, t):
        if self.conversation_started and not self.applied:
            return self._say(t.HOW_TO_APPLY, fixes)
        if not self.conversation_started:
            return self._say(t.SAY_HELLO, fixes)
        default = [] if fixes.myself_all_features else ["A1"]
        features_of_interest = args.get("features", default)
        if features_of_interest:
            features_of_interest = [self.model.feature_name(i)
                                    for i in features_of_interest]
        else:
            features_of_interest = [self.model.feature_name(i)
                                    for i in self.model.feature_id]
        values = self.model.personas[self.persona].values
        about = ""
        for feature_name, feature_value in self.model.translate_values(values):
            if feature_name in features_of_interest:
                about += " you have " + self.get_statement(
                    feature_name, feature_value, t) + " and "
        self._event('myself')
        return self._say(t.ABOUT.format(name=APPLICANT_NAMES[self.persona])
                         + about[:-5] + ".", fixes)

    def _intent_repeat(self, args, fixes, t):
        if self.conversation_started:
            return self.last_said
        return t.NOTHING_SAID

    def _intent_unknown(self, args, fixes, t):
        if self.conversation_started and not self.applied:
            return self._say(t.HOW_TO_APPLY, fixes)
        if self.conversation_started:
            return self._say(t.DUNNO, fixes)
        return self._say(t.SAY_HELLO, fixes)
