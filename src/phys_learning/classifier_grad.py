from sklearn.ensemble import GradientBoostingClassifier
from .classifier import Classifier


class GradientBoosting(Classifier):
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None,
                 name="test", model=None, seed=None, verbose=0,
                 max_depth=3,
                 n_estimators=50, learning_rate=1.0):
        super().__init__(x_train, x_test, y_train, y_test, name, model,
                         seed, verbose)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def new_model(self):
        self.model = GradientBoostingClassifier(
            max_depth=3,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.seed)

    def model_info(self):
        print('GradientBoosting, max_depth={}'.format(self.max_depth))

    def learn(self):
        self.model.fit(self.x_train, self.y_train)

    def run_test(self):
        self.acc = self.model.score(self.x_test, self.y_test)
        if self.verbose > 0:
            print("test accuracy", self.acc)
