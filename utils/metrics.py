def accuracy(*args, **kwargs):
    """
        Вычисление точности:
            accuracy = sum( predicted_class == ground_truth ) / N, где N - размер набора данных
        TODO: реализуйте подсчет accuracy
    """
    raise NotImplementedError


def balanced_accuracy(*args, **kwargs):
    """
        Вычисление точности:
            balanced accuracy = sum( TP_i / N_i ) / N, где
                TP_i - кол-во изображений класса i, для которых предсказан класс i
                N_i - количество изображений набора данных класса i
                N - количество классов в наборе данных
        TODO: реализуйте подсчет balanced accuracy
    """
    raise NotImplementedError
