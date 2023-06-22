import tensorflow as tf


class BalancedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, weight_positive, weight_negative, name='balanced_cross_entropy'):
        super(BalancedCrossEntropy, self).__init__(name=name)
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative

    def call(self, y_true, y_pred):
        # 计算平衡交叉熵损失
        loss = self.weight_positive * y_true * tf.math.log(y_pred) + self.weight_negative * (1 - y_true) * tf.math.log(
            1 - y_pred)
        loss = -tf.reduce_mean(loss)

        return loss


class WeightedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, positive_weight, negative_weight, name='weighted_cross_entropy'):
        super(WeightedCrossEntropy, self).__init__(name=name)
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def call(self, y_true, y_pred):
        # 计算加权交叉熵损失
        loss = self.positive_weight * y_true * tf.math.log(y_pred) + self.negative_weight * (1 - y_true) * tf.math.log(
            1 - y_pred)
        loss = -tf.reduce_mean(loss)

        return loss