import matplotlib.pyplot as plt


def plot_feature_importance(features, importance):
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()
