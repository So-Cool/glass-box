"""
Generates the additional NLU training data and the held-out NLU test set.

The examples are generated from sentence templates with slots for features,
personas and marker words (``given``, ``despite``, ``reset``) -- see
:data:`TRAIN` and :data:`TEST`;
the test set uses different templates (and partly different feature phrases)
than the training data.
Use ``glassbox nlu generate`` to write the files.

Examples use the markup ``[text](entity=value)``.
"""
import json
import os
import random
import re

from glassbox.resources import resource_path

__all__ = ['generate', 'write_data', 'TRAIN', 'TEST',
           'DEFAULT_TRAINING_OUTPUT', 'DEFAULT_TEST_OUTPUT']

#: the training data added in 2026 (shipped with the package)
DEFAULT_TRAINING_OUTPUT = resource_path('nlu', 'training_data_extra.json')
#: the held-out test set (part of the repository's tests)
DEFAULT_TEST_OUTPUT = os.path.join('tests', 'data', 'nlu_test_set.json')

PERSONAS = {"00": "Stu", "01": "Jonathan", "02": "Lurleen", "03": "Ruth",
            "04": "Joe", "05": "Kirk", "06": "Mona", "07": "Selma",
            "08": "Montgomery", "09": "Marge"}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine"]

# feature phrases: (training, test) -- the test set reuses some phrases but
# never the sentence templates
FEATURES = {
    "A1": (["current account", "checking account", "current account status",
            "bank account"], ["current account balance", "checking account"]),
    "A2": (["loan duration", "duration of the loan", "loan term",
            "repayment period"], ["loan length", "loan duration"]),
    "A3": (["credit history", "repayment history",
            "history of paying back loans"], ["credit record",
                                              "credit history"]),
    "A5": (["loan amount", "amount I asked for", "amount of money",
            "requested amount"], ["amount borrowed", "loan amount"]),
    "A6": (["savings", "savings account", "savings account status"],
           ["savings balance", "savings"]),
    "A7": (["employment duration", "time in my current job",
            "length of employment"], ["years of employment",
                                      "employment duration"]),
    "A8": (["installment rate", "monthly installment", "instalment rate"],
           ["installment rate", "repayment rate"]),
    "A9": (["marital status", "gender", "sex", "marital status and gender"],
           ["marital status", "being married"]),
    "A13": (["age", "date of birth"], ["age", "being too young"]),
    "A14": (["other loans", "other installment plans", "loans elsewhere"],
            ["other credits", "other loans"]),
    "A15": (["housing", "housing situation", "home ownership"],
            ["housing", "accommodation"]),
    "A17": (["job", "job type", "occupation", "employment type"],
            ["profession", "job"]),
    "A20": (["foreign worker status", "migrant status", "nationality"],
            ["foreign worker status", "citizenship"]),
}

TRAIN = {
    "greet": ["hello there", "good morning", "good afternoon",
              "good evening", "hey there", "hi there", "greetings",
              "hello robot", "hello Glass-Box", "hi Glass-Box",
              "hey, anyone there?", "morning",
              "hello, how are you today",
              "hi, is this the bank",
              "hello, can you hear me"],
    "farewell": ["see you", "goodbye then", "that is all, thanks",
                 "I am done", "thanks, bye", "cheers, bye",
                 "that will be all", "good bye", "see you later",
                 "I have no more questions",
                 "ok, that is everything, bye"],
    "help": ["help", "help me", "how does this work",
             "what should I ask", "I need help", "what can you do for me",
             "what are my options", "how do I use this"],
    "repeat": ["say that again", "pardon", "come again",
               "what did you say", "can you say it again",
               "one more time please", "repeat that",
               "I did not catch that",
               "what was that",
               "sorry, say it again please"],
    "outcome": ["did I get the loan", "was my loan approved",
                "what was the decision", "has my application been approved",
                "tell me the outcome", "did you accept my application",
                "am I getting the loan", "what is the decision",
                "was my application successful", "was my loan rejected",
                "have I been approved",
                "am I accepted",
                "was I successful",
                "did my loan get approved"],
    "fairness": ["is this decision fair", "was I treated fairly",
                 "is the decision biased", "was I discriminated against",
                 "is this outcome unfair", "did you discriminate against me",
                 "is the model fair", "was the decision just"],
    "apply_name": ["I am {name}", "my name is {name}", "this is {name}",
                   "{name} here", "I would like to apply, I am {name}",
                   "I want to apply as {name}", "call me {name}",
                   "it's {name}",
                   "hi, I am {name}",
                   "hello, my name is {name}",
                   "the name is {name}",
                   "{name} is my name",
                   "I am {name} and I want a loan"],
    "apply_number": ["my application number is {num}", "my number is {num}",
                     "application {num}", "applicant number {num}",
                     "I have application number {num}"],
    "myself_feature": ["what is my {f}", "tell me about my {f}",
                       "what do you know about my {f}",
                       "what do you have on record about my {f}",
                       "what did I declare as my {f}"],
    "myself_two": ["what are my {f} and my {f2}",
                   "tell me about my {f} and {f2}"],
    "myself": ["tell me about myself", "what information do you have about me",
               "what do you know about me", "read me my application",
               "what is on my record"],
    "why": ["why was my application declined", "why was my loan rejected",
            "why did I not get the loan", "explain the decision",
            "why was my application accepted", "explain why",
            "why did you decline me", "can you explain the outcome",
            "what's the reason my application was declined",
            "give me the reasons",
            "tell me why",
            "what are the reasons behind the decision",
            "what led to this decision"],
    "why_given": ["why, {given} my {f}", "explain the decision {given} my {f}",
                  "why was I declined {given} my {f} and {f2}",
                  "tell me why, {given} my {f}",
                  "what are the reasons, {given} my {f}",
                  "explain this {given} my {f}"],
    "why_despite": ["why, {despite} my {f}",
                    "explain the decision {despite} my {f}",
                    "{despite} my {f}, why was I declined",
                    "tell me why, {despite} my {f}",
                    "what are the reasons, {despite} my {f}",
                    "give me the reasons {despite} my {f}"],
    "why_both": ["why {given} my {f} {despite} my {f2}",
                 "explain the decision {given} my {f} and {despite} my {f2}"],
    "why_reset": ["{reset} the explanations", "{reset} why",
                  "{reset} the reasons from the beginning",
                  "please {reset} the reasons",
                  "can you {reset} the why"],
    "action": ["what can I do to get the loan",
               "how can I get my loan approved",
               "what should I change to get the loan", "what can I change",
               "how could I improve my chances",
               "what would I have to change to be accepted",
               "what else could I change",
               "what would help me get the loan",
               "how can I get accepted",
               "what steps can I take"],
    "action_despite": [
        "{despite} my {f}, what can I do to get my loan accepted",
        "what can I do {despite} my {f}",
        "how can I get the loan {despite} my {f}",
        "what else could I change, {despite} my {f}",
        "what steps can I take {despite} my {f}"],
    "action_given": ["what can I do with my {f} to get my loan accepted",
                     "how should I change my {f} to get the loan",
                     "what can I change about my {f}"],
}
TEST = {
    "greet": ["hi, good to meet you", "hello, is anybody there"],
    "farewell": ["ok, thank you, goodbye", "nothing else, bye"],
    "help": ["I am not sure what to ask", "can you tell me what to do"],
    "repeat": ["sorry, could you repeat that", "I missed that"],
    "outcome": ["have I been accepted", "so what is the verdict",
                "did my application go through"],
    "fairness": ["was the decision unbiased", "is that a fair outcome"],
    "apply_name": ["hello, it is {name}", "{name} applying",
                   "the applicant is {name}"],
    "apply_number": ["number {num}", "my application is {num}"],
    "myself_feature": ["what have you got on my {f}", "remind me of my {f}"],
    "myself": ["what does my application say about me"],
    "why": ["how come I was declined", "what is the reason for the decision"],
    "why_given": ["explain it {given} my {f}"],
    "why_despite": ["what is the reason, {despite} my {f}"],
    "why_reset": ["please {reset} the explanation"],
    "action": ["what could I do differently",
               "how do I get the loan approved next time"],
    "action_despite": ["what else can I do, {despite} my {f}"],
    "action_given": ["can I change my {f} to get the loan"],
}
GIVEN = (["given", "considering", "taking into account", "based on"],
         ["given", "in view of"])
DESPITE = (["despite", "disregarding", "ignoring", "regardless of",
            "without considering"], ["despite", "leaving aside"])
RESET = (["restart", "reset", "start over"], ["restart", "reset"])


def annotate(intent, template, rng, split):
    """Fills a template; returns markup and the intent name."""
    i = 0 if split == 'train' else 1
    text = template
    features = rng.sample(sorted(FEATURES), 2)
    for slot, f in (('{f}', features[0]), ('{f2}', features[1])):
        if slot in text:
            phrase = rng.choice(FEATURES[f][i])
            text = text.replace(slot, '[{}](feature={})'.format(phrase, f))
    if '{name}' in text:
        pid = rng.choice(sorted(PERSONAS))
        text = text.replace('{name}', '[{}](persona={})'.format(
            PERSONAS[pid], pid))
    if '{num}' in text:
        pid = rng.choice(sorted(PERSONAS))
        shown = rng.choice([str(int(pid)), pid, WORDS[int(pid)]])
        text = text.replace('{num}', '[{}](persona={})'.format(shown, pid))
    for slot, words, entity in (('{given}', GIVEN, 'why_given'),
                                ('{despite}', DESPITE, 'why_despite'),
                                ('{reset}', RESET, 'why_reset')):
        if slot in text:
            text = text.replace(slot, '[{}]({}=true)'.format(
                rng.choice(words[i]), entity))
    name = intent.split('_')[0]
    return {'myself': 'myself', 'apply': 'apply', 'action': 'action',
            'why': 'why'}.get(name, intent), text


MARKUP = re.compile(
    r'\[(?P<text>[^\]]+)\]\((?P<entity>[a-z_]+)=(?P<value>[^)]+)\)')


def to_example(intent, markup):
    text, entities, offset = '', [], 0
    for m in MARKUP.finditer(markup):
        text += markup[offset:m.start()]
        start = len(text)
        text += m.group('text')
        entities.append({'start': start, 'end': len(text),
                         'value': m.group('value'),
                         'entity': m.group('entity')})
        offset = m.end()
    text += markup[offset:]
    return {'text': text, 'intent': intent, 'entities': entities}


def generate(templates, split, repeats, seed):
    """
    Fills the templates of a split.

    Parameters
    ----------
    templates : dict
        ``{intent: [templates]}`` (:data:`TRAIN` or :data:`TEST`).
    split : {'train', 'test'}
        Selects the feature phrases and marker words of the split.
    repeats : int
        How many times every template with slots is filled.
    seed : int
        Seed of the random choice of the slot fillers.

    Returns
    -------
    list of dict
        Rasa NLU ``common_examples`` (duplicates removed).
    """
    rng = random.Random(seed)
    examples = []
    for intent, group in templates.items():
        slotted = any('{' in t for t in group)
        for template in group:
            for _ in range(repeats if slotted else 1):
                examples.append(to_example(*annotate(intent, template, rng,
                                                     split)))
    unique = {json.dumps(e, sort_keys=True): e for e in examples}
    return list(unique.values())


def _write(path, examples):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'rasa_nlu_data': {'common_examples': examples}}, f,
                  indent=1)


def write_data(training_output=DEFAULT_TRAINING_OUTPUT,
               test_output=DEFAULT_TEST_OUTPUT, repeats=3, seed=2026,
               test_repeats=3, test_seed=7):
    """
    Generates and writes the training data and the test set.

    The defaults reproduce the shipped training data and the test set of the
    repository.

    Returns
    -------
    tuple
        The numbers of training and test examples.
    """
    train = generate(TRAIN, 'train', repeats=repeats, seed=seed)
    test = generate(TEST, 'test', repeats=test_repeats, seed=test_seed)
    if training_output:
        _write(training_output, train)
    if test_output:
        _write(test_output, test)
    return len(train), len(test)
