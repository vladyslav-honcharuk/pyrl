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

    def update(self, trial, status):
        """Update performance metrics based on trial outcome."""
        self.n_trials += 1
        if 'choice' in status:
            self.n_decision += 1
            if status.get('correct', False):
                self.n_correct += 1

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
