"""
Answer templates of the dialogue.

The templates of the original implementation with corrected grammar and
typography (missing verbs, articles and punctuation, typos);
the original wording is not kept.
How every feature is put into words (``phrasing``), the persona names and the
actionable and protected features are part of the data
(``resources/german_credit_features.json`` and ``resources/personas.json``).
"""

NUMBERS = {"zero": "00", "one": "01", "two": "02", "three": "03",
           "four": "04", "five": "05", "six": "06", "seven": "07",
           "eight": "08", "nine": "09"}
OUTCOME_MAP = {"good": "accepted", "bad": "declined"}


class Templates(object):
    """Answer templates."""

    DUNNO = "Sorry but I do not know what to say."
    GREET = "Hello. Welcome to Robo Bank."
    HOW_TO_APPLY = ("If you would like to apply for a loan please provide "
                    "your name or application number. Alternatively please "
                    "scan your application card.")
    BYE = "Thank you for your application. Goodbye."
    APPLICATION_RECEIVED = ("Thank you for your application, {name}. Please "
                            "bear with me a moment while I process your "
                            "application.")
    APPLICATION_PROCESSED = ("Your loan application has been processed. "
                             "Would you like to know the outcome?")
    APPLICATION_OUTCOME = "Your application has been {outcome}."
    COUNTERFACTUAL_OUTCOME = "your loan would be {outcome}"
    ABOUT = "Dear {name}, at the time of application:"
    FAIRNESS = ("I believe that this decision is {fairness}. The reason I am "
                "saying this is because the outcome of your application "
                "would have changed {counterfactual}.")
    FAIRNESS_INDIRECT = ("I believe that this decision is indirectly "
                         "{fairness}. The reason I am saying this is because "
                         "some protected features when combined with other "
                         "details of your application cause the outcome of "
                         "your application to change, that is "
                         "{counterfactual}.")
    ACTIONABILITY = ("In order to have your loan application "
                     "{wannabe_outcome} instead of {current_outcome} please "
                     "consider the following options. The outcome of your "
                     "application would change {counterfactual}.")
    COUNTERFACTUAL_STATEMENT_1 = ("{intro} {counterfactual_statement}, "
                                  "{counterfactual_outcome}")
    COUNTERFACTUAL_STATEMENT_2 = ("{intro} {counterfactual_outcome} "
                                  "{counterfactual_statement}")
    COUNTERFACTUAL_TEMPLATE_COMP_1 = ("had you {hypothetical_state}, instead "
                                      "of {current_state}")
    COUNTERFACTUAL_TEMPLATE_COMP_2 = ("if you had {hypothetical_state}, "
                                      "rather than {current_state}")
    COUNTERFACTUAL_TEMPLATE_COMP_3 = ("assuming that you had "
                                      "{hypothetical_state}, rather than "
                                      "{current_state}")
    COUNTERFACTUAL_TEMPLATE_1 = "had you {hypothetical_state}"
    COUNTERFACTUAL_TEMPLATE_2 = "if you had {hypothetical_state}"
    COUNTERFACTUAL_TEMPLATE_3 = "assuming that you had {hypothetical_state}"

    SAY_HELLO = "Say 'hello' to start a conversation."
    ALREADY_APPLIED = ("You have already applied for your loan. Would you "
                       "like to know the outcome?")
    ALREADY_APPLIED_HELLO = (
        "You have already applied for your loan. Would you like to know the "
        "outcome? Alternatively say 'goodbye' to finish this application "
        "process.")
    BYE_NOT_APPLIED = ("Goodbye. You have not applied yet for a loan. Do "
                       "you want to apply?")
    NOT_ON_RECORD = ("I am sorry but we do not have you on our record. Please "
                     "reapply for a loan. Available personas (identification "
                     "numbers) are: {}.")
    UNKNOWN_PERSON = ("I am sorry but I do not know a person with "
                      "identification number: {}. Available personas "
                      "(identification numbers) are: {}.")
    DIFFERENT_PERSON = ("You have already applied for a loan as a different "
                        "person: {}. If you want to reapply as {}, please say "
                        "'goodbye' to finish this application first.")
    DIFFERENT_UNKNOWN_PERSON = (
        "You have already applied for a loan as a different person: {}. You "
        "are now trying to apply as {}, but we do not have this person on "
        "record. Continue your application or say 'goodbye' to end the "
        "process.")
    HELP_APPLIED = (
        "Now that you have applied for a loan you can ask why. Try asking one "
        "of the following questions: 'Why has my loan application been "
        "denied?' or 'Is the decision fair?' or 'What can I do to get my "
        "loan application accepted?' or 'Disregarding my age what can I do "
        "to get my loan application accepted?' or 'What can I do with my "
        "instalment rate to get my loan application accepted?'.")
    FAIR = ("I understand that you are concerned about the decision "
            "fairness, however the decision is fair. None of the protected "
            "features has been used in the decisive process.")
    NO_ACTION = ("I am sorry but at the moment there is nothing you can do to "
                 "change the decision. Your loan is {}.")
    NO_ENQUIRY = ("I do not have any ongoing enquiry. Please give me a moment "
                  "to open one for you. ")
    NO_COUNTERFACTUALS = ("I am sorry. I am afraid that I cannot generate any "
                          "counterfactuals for you at the moment. Please try "
                          "again later.")
    QR_BUSY = ("You are applying as {} at the moment. Please say 'goodbye' "
               "to finish this application before scanning another card.")
    NOTHING_SAID = "I am sorry but I have not said anything yet."
    NO_MORE = {
        'why': ("I have no more explanations of this decision. Ask me again "
                "to hear them from the beginning."),
        'fairness': ("I have no more fairness concerns to report. Ask me "
                     "again to hear them from the beginning."),
        'action': ("I have no more suggestions for you. Ask me again to hear "
                   "them from the beginning.")}
