"""
Performance tracking for cognitive tasks.
"""
from collections import OrderedDict


class Performance2AFC:
    """Track performance for 2-alternative forced choice tasks."""

    def __init__(self):
        self.decisions = []
        self.corrects = []
        self.choices = []
        self.t_choices = []

    def update(self, trial, status):
        """Update performance metrics based on trial outcome."""
        if 'correct' in status:
            self.decisions.append(True)
            self.corrects.append(status['correct'])
            if 'choice' in status:
                self.choices.append(status['choice'])
            else:
                self.choices.append(None)
            if 't_choice' in status:
                self.t_choices.append(status['t_choice'])
            else:
                self.t_choices.append(None)
        else:
            self.decisions.append(False)
            self.corrects.append(False)
            self.choices.append(None)
            self.t_choices.append(None)

    @property
    def n_trials(self):
        return len(self.decisions)

    @property
    def n_decision(self):
        return sum(self.decisions)

    @property
    def n_correct(self):
        return sum(self.corrects)

    def display(self, output=True):
        """Display performance metrics."""
        n_trials = self.n_trials
        n_decision = self.n_decision
        n_correct = self.n_correct

        items = OrderedDict()
        items['P(choice)'] = f'{n_decision}/{n_trials} = {n_decision/n_trials:.3f}'
        if n_decision > 0:
            items['P(correct|choice)'] = f'{n_correct}/{n_decision} = {n_correct/n_decision:.3f}'

        if output:
            from .utils import print_dict
            print_dict(items)
        return items
