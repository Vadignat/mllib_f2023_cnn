import torch


def accuracy(predictions, labels):
    """
        Вычисление точности:
            accuracy = sum( predicted_class == ground_truth ) / N, где N - размер набора данных
        TODO: реализуйте подсчет accuracy
    """
    correct_predictions = (predictions == labels).sum().item()
    total_samples = len(labels)
    accuracy_value = correct_predictions / total_samples
    return accuracy_value


def balanced_accuracy(predictions, labels, num_classes):
    """
        Вычисление точности:
            balanced accuracy = sum( TP_i / N_i ) / N, где
                TP_i - кол-во изображений класса i, для которых предсказан класс i
                N_i - количество изображений набора данных класса i
                N - количество классов в наборе данных
        TODO: реализуйте подсчет balanced accuracy
    """
    confusion_matrix = torch.zeros(num_classes, num_classes)

    for t, p in zip(labels.view(-1), predictions.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    TP = torch.diag(confusion_matrix)
    N_i = confusion_matrix.sum(dim=1)

    balanced_accuracies = TP / N_i
    balanced_accuracy_value = balanced_accuracies.mean().item()

    return balanced_accuracy_value
