from data_processing import load_and_process_data
from decision_tree_model import train_decision_tree
from neural_network_model import train_neural_network

if __name__ == "__main__":
    file_path = "datas/Small sample data.xlsx"
    X_train, X_test, y_train, y_test = load_and_process_data(file_path)

    print("\n===== 决策树模型训练 =====")
    dt_model = train_decision_tree(X_train, X_test, y_train, y_test)

    print("\n===== 神经网络模型训练 =====")
    nn_model = train_neural_network(X_train, X_test, y_train, y_test)
