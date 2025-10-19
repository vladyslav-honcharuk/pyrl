"""
Performance tracking for cognitive tasks.
"""
from collections import OrderedDict


class Performance2AFC:
    """Track performance for 2-alternative forced choice tasks."""

    def __init__(self):
        self.n_trials = 0
        self.n_decision = 0
        self.n_correct = 0

        # Track individual trial outcomes
        self.decisions = []  # Whether a decision was made
        self.choices = []    # The choice made ('L', 'R', or None)
        self.corrects = []   # Whether the choice was correct
        self.t_choices = []  # Timestep when choice was made

    def update(self, trial, status):
        """Update performance metrics based on trial outcome."""
        self.n_trials += 1

        if 'choice' in status:
            self.n_decision += 1
            self.decisions.append(True)
            self.choices.append(status['choice'])
            self.t_choices.append(status.get('t_choice', None))

            is_correct = status.get('correct', False)
            self.corrects.append(is_correct)
            if is_correct:
                self.n_correct += 1
        else:
            self.decisions.append(False)
            self.choices.append(None)
            self.corrects.append(None)
            self.t_choices.append(None)

    def display(self, output=True):
        """Display performance metrics."""
        items = OrderedDict()

        p_decision = self.n_decision / self.n_trials if self.n_trials > 0 else 0
        p_correct = self.n_correct / self.n_decision if self.n_decision > 0 else 0

        items['Decisions'] = f'{self.n_decision}/{self.n_trials} ({p_decision:.2%})'
        items['Correct'] = f'{self.n_correct}/{self.n_decision} ({p_correct:.2%})'

        if output:
            from .utils import print_dict
            print_dict(items)

        return items
