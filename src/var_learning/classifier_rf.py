from sklearn.ensemble import RandomForestClassifier
from .classifier import Classifier


class RandomForest(Classifier):
    def __init__(self, max_depth=3, **kw):
        super().__init__(**kw)
        self.max_depth = max_depth

    def new_model(self):
        self.model = RandomForestClassifier(max_depth=self.max_depth,
                                            random_state=self.seed)

    def model_info(self):
        print('RandomForestClassifier, max_depth={}'.format(self.max_depth))

    def learn(self):
        self.model.fit(self.x_train, self.y_train)

    def run_test(self):
        self.acc = self.model.score(self.x_test, self.y_test)
        if self.verbose > 0:
            print("test accuracy", self.acc)
