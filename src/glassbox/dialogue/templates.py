"""
Original answer templates of the dialogue and their corrected versions
(used with the ``wording`` fix).
"""
import copy

APPLICANT_NAMES = {
    "00": "Stu", "01": "Jonathan", "02": "Lurleen", "03": "Ruth",
    "04": "Joe", "05": "Kirk", "06": "Mona", "07": "Selma",
    "08": "Montgomery", "09": "Marge"}
NUMBERS = {"zero": "00", "one": "01", "two": "02", "three": "03",
           "four": "04", "five": "05", "six": "06", "seven": "07",
           "eight": "08", "nine": "09"}
OUTCOME_MAP = {"good": "accepted", "bad": "declined"}
FEATURE_ACTIONABLE = ["A17", "A8", "A14", "A6", "A1", "A5", "A2"]
FEATURE_UNFAIR = ["A20", "A9"]

FEATURE_TALK = {
    "loan_duration_in_months": {
        "part1": "requested your loan for {value} {value_type}",
        "part2": "{value} {value_type}",
        "property": {"singular": "month", "plural": "months"},
        "name": "loan duration"},
    "loan_amount": {
        "part1": "asked for {value} {value_type}",
        "part2": "{value} {value_type}",
        "property": {"singular": "pound", "plural": "pounds"},
        "name": "loan amount"},
    "installment_rate": {
        "part1": "received an installment rate of {value} {value_type}",
        "part2": "{value} {value_type}",
        "property": {"singular": "percent", "plural": "percent"},
        "name": "installment rate"},
    "age": {
        "part1": "been {value} {value_type}",
        "part2": "{value} {value_type}",
        "property": {"singular": "year old", "plural": "years old"},
        "name": "age"},
    "is_foreign_worker": {
        "part1": "{value} been a foreign worker",
        "value": {"yes": "", "no": "not"},
        "name": "migrant status"},
    "other_loans": {
        "part1": "{value} had other loans",
        "value": {"yes": "", "no": "not"},
        "name": "other loans"},
    "housing_type": {
        "part1": "{value} {a} house",
        "part2": "{value} {a} house",
        "value_hypothetical1": {"own": "owned", "rent": "rented",
                                "council": "lived in a council"},
        "value_hypothetical2": {"own": "owning", "rent": "renting",
                                "council": "living in a council"},
        "name": "housing type"},
    "job_type": {"part1": "been {value}", "part2": "being {value}",
                 "map": None, "name": "job type"},
    "marital_status_and_gender": {
        "part1": "been {value}", "part2": "being {value}", "map": None,
        "name": "marital status and gender"},
    "duration_of_present_employment": {
        "part1": "been {value}", "part2": "being {value}", "map": None,
        "name": "current employment duration"},
    "status_of_savings_account": {"part1": "{value}", "part2": "{value}",
                                  "map": None,
                                  "name": "savings account status"},
    "credit_history": {"part1": "{value}", "part2": "{value}", "map": None,
                       "name": "credit history"},
    "status_of_existing_current_account": {
        "part1": "{value}", "part2": "{value}", "map": None,
        "name": "current account status"}
}

# The wording fix adds the missing verb of the account/history features.
FEATURE_TALK_FIXED = copy.deepcopy(FEATURE_TALK)
for _name in ("status_of_savings_account", "credit_history",
              "status_of_existing_current_account"):
    FEATURE_TALK_FIXED[_name]["part1"] = "had {value}"
    FEATURE_TALK_FIXED[_name]["part2"] = "having {value}"


class Templates(object):
    """Answer templates (``fixed=True`` selects the corrected wording)."""

    DUNNO = "Sorry but I do not know what to say."
    GREET = "Hello. Welcome to Robo Bank."
    HOW_TO_APPLY = ("If you would like to apply for a loan please provide "
                    "your name or application number. Alternatively please "
                    "scan your application card.")
    BYE = "Thank you for your application. Goodbye."
    APPLICATION_RECEIVED = ("Thank you for your application, {name}. Please "
                            "bear with me a moment while I process you "
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
    COUNTERFACTUAL_STATEMENT_1 = ("{intro} {counterfactual_statement} "
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
                        "'goodbye' to finish this application first?")
    DIFFERENT_UNKNOWN_PERSON = (
        "You have already applied for a loan as a different person: {}. You "
        "are now trying to apply as {}, but we do not have this person on "
        "record. Continue you application or say 'goodbye' to end the "
        "process.")
    HELP_APPLIED = (
        "Now that you have applied for a loan you can ask why. Try asking one "
        "of the following questions: 'Why has my loan application been "
        "denied?' or 'Is the decision fair?' or 'What can I do to get my "
        "loan application accepted?' or 'Disregarding my age what can I do "
        "to get my loan application?' 'What can I do with my installment "
        "rate to get my loan application accepted?'.")
    FAIR = ("I understand that you are concerned about the decision "
            "fairness, however the decision is fair. None of the protected "
            "features has been used in the decisive process.")
    NO_ACTION = ("I am sorry but at the moment there is nothing you can do to "
                 "change the decision. Your loan is {}")
    NO_ENQUIRY = ("I do not have any ongoing enquiry. Please give me a moment "
                  "to open one for you. ")
    NO_COUNTERFACTUALS = ("I am sorry. I am afraid that I cannot generate any "
                          "counterfactuals for you at the moment. Please try "
                          "again later.")
    NOTHING_SAID = "I am sorry but I have not said anything yet."
    NO_MORE = {
        'why': ("I have no more explanations of this decision. Ask me again "
                "to hear them from the beginning."),
        'fairness': ("I have no more fairness concerns to report. Ask me "
                     "again to hear them from the beginning."),
        'action': ("I have no more suggestions for you. Ask me again to hear "
                   "them from the beginning.")}

    def __init__(self, fixed=False):
        self.fixed = fixed
        self.feature_talk = FEATURE_TALK_FIXED if fixed else FEATURE_TALK
        if fixed:
            self.APPLICATION_RECEIVED = self.APPLICATION_RECEIVED.replace(
                'process you application', 'process your application')
            self.DIFFERENT_UNKNOWN_PERSON = \
                self.DIFFERENT_UNKNOWN_PERSON.replace(
                    'Continue you application', 'Continue your application')
            self.DIFFERENT_PERSON = self.DIFFERENT_PERSON.replace(
                'application first?', 'application first.')
            self.HELP_APPLIED = self.HELP_APPLIED.replace(
                "to get my loan application?' 'What",
                "to get my loan application accepted?' or 'What")
            self.NO_ACTION = self.NO_ACTION + '.'
            # a comma between the condition and the outcome
            self.COUNTERFACTUAL_STATEMENT_1 = (
                "{intro} {counterfactual_statement}, {counterfactual_outcome}")
