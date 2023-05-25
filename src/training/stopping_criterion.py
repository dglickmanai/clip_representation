class StoppingCriterion:
    def __init__(self, patience, higher_is_better=True):
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.num_consecutive_successes = 0
        self.num_no_improvement = 0
        self.best_so_far = float('-inf') if higher_is_better else float('inf')

    def __call__(self, result):
        if self.is_better(self.best_so_far, result):
            self.num_no_improvement += 1
        else:
            self.num_no_improvement = 0
            self.best_so_far = result

        return self.num_no_improvement >= self.patience

    def is_better(self, new_value, old_value):
        if self.higher_is_better:
            return new_value >= old_value
        return new_value <= old_value


# sc = StoppingCriterion(1, higher_is_better=True)
# assert not sc(0.2)
# assert not sc(0.3)
#
# sc = StoppingCriterion(1, higher_is_better=True)
# assert not sc(0.2)
# assert sc(0.1)
#
# sc = StoppingCriterion(3)
# sc(0.1)
#
# assert not sc(0.)
# assert not sc(0.)
# assert not sc(0.2)
#
# assert not sc(0.)
# assert not sc(0.)
# assert sc(0.)