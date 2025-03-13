import numpy as np
from typing import Any, ClassVar
from scipy.optimize import minimize
from features.prepare_for_training import prepare_for_training
from hypothesis.sigmoid import sigmoid


class logisticRegression:
    """
    手搓逻辑回归
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
        self.num_features = self.data.shape[1]
        num_union_lables = np.unique(labels).shape[0]
        # 在分类中需要得到label中唯一值，即为结果值
        self.union_lables = np.unique(labels)

        self.labels = labels
        self.alf = alf
        self.step = step
        self.theta = np.zeros((num_union_lables, self.num_features))

    def train(self, max_iter: int = 500):
        cost_list = []
        for index, labal in enumerate(self.union_lables):
            current_theta = np.copy(self.theta[index].reshape(self.num_features, 1))
            # 这里需要将 布尔矩阵转换为float
            current_labal = (self.theta == labal).astype(float)
            (res_theta, cost) = self.gradient_descent(
                self.data, current_labal, current_theta, max_iter
            )

    def gradient_descent(
        self, data: np.ndarray, labal: np.ndarray, theta: np.ndarray, max_iter: int
    ) -> list:
        cost_list = []
        minimize(lambda cur_theta: self.cost_fc(data, labal))

    def cost_fc(self, data: np.ndarray, labels: np.ndarray):
        hypothesis = logisticRegression.hypothesis(data, labels)

    def gradient_step(self, alpha: float) -> None:
        pass

    @staticmethod
    def hypothesis(data: np.ndarray, theta: np.ndarray):
        # 将预测值传入sigmoid中
        pre = sigmoid(data.dot(theta))
        return pre

    def get_cost(self, data: np.ndarray, labels):
        pass

    def predict(self, data: np.ndarray):
        pass
