import numpy as np
from typing import Any, ClassVar
from features.prepare_for_training import prepare_for_training


class linearRegression:
    """
    手搓线性回归
    """

    def __init__(
        self,
        data,
        labels,
        alf: float,
        step: float,
        polynomial_degree: int = 0,
        sinusoid_degree: int = 0,
        normalize_data: bool = True,
    ):
        (data_processed, feature_mean, feature_deviation) = prepare_for_training(
            data, polynomial_degree, sinusoid_degree, normalize_data
        )
        self.data = data_processed
        self.feature_mean = feature_mean
        self.feature_deviation = feature_deviation
        self.normalize_data = normalize_data
        self.sinusoid_degree = sinusoid_degree
        self.polynomial_degree = polynomial_degree
        num_features = self.data.shape[1]
        self.labels = labels
        self.alf = alf
        self.step = step
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha: float = None, num_item: int = 500):
        if alpha is None:
            alpha = self.alf
        return self.theta, self.gradient_descent(alpha, num_item)

    def gradient_descent(self, alpha: float, num_item: int = 500) -> list:
        cost_fc = []
        for _ in range(num_item):
            self.gradient_step(alpha)
            cost_fc.append(self.cost_fc(self.data, self.labels))
        return cost_fc

    def gradient_step(self, alpha: float) -> None:
        num_data = self.data.shape[0]
        hypothesis = linearRegression.hypothesis(self.data, self.theta)
        delta = hypothesis - self.labels
        gradient = (1 / num_data) * (self.data.T.dot(delta))
        self.theta -= alpha * gradient

    @staticmethod
    def hypothesis(data: np.ndarray, theta: np.ndarray):
        return data.dot(theta)

    def cost_fc(self, data: np.ndarray, labels: np.ndarray):
        num_samples = data.shape[0]
        delta = linearRegression.hypothesis(data, self.theta) - labels
        cost = (1 / (2 * num_samples)) * np.dot(delta.T, delta)
        return cost[0][0]

    def get_cost(self, data: np.ndarray, labels):
        data_processed, _, _ = prepare_for_training(
            data,
            polynomial_degree=self.polynomial_degree,
            sinusoid_degree=self.sinusoid_degree,
            normalize_data=self.normalize_data,
        )
        return self.cost_fc(data_processed, labels)

    def predict(self, data: np.ndarray):
        data_processed, _, _ = prepare_for_training(
            data,
            polynomial_degree=self.polynomial_degree,
            sinusoid_degree=self.sinusoid_degree,
            normalize_data=self.normalize_data,
        )
        return linearRegression.hypothesis(data_processed, self.theta)
