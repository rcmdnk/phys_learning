from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from .classifier import Classifier


class AdaBoost(Classifier):
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None,
                 name="test", model=None, seed=None, verbose=0,
                 max_depth=3,
                 n_estimators=50, learning_rate=1.0, algorithm='SAMME.R'):
        super().__init__(x_train, x_test, y_train, y_test, name, model,
                         seed, verbose)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm

    def new_model(self):
        dt = DecisionTreeClassifier(max_depth=self.max_depth)
        self.model = AdaBoostClassifier(base_estimator=dt,
                                        n_estimators=self.n_estimators,
                                        learning_rate=self.learning_rate,
                                        algorithm=self.algorithm,
                                        random_state=self.seed)

    def model_info(self):
        print('AdaBoost, max_depth={}'.format(self.max_depth))

    def learn(self):
        self.model.fit(self.x_train, self.y_train)

    def run_test(self):
        self.acc = self.model.score(self.x_test, self.y_test)
        if self.verbose > 0:
            print("test accuracy", self.acc)
