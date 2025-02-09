import torch
from neural_network_model import train_neural_network
from data_processing import load_and_process_data
from decision_tree_model import train_decision_tree, train_random_forest, train_xgboost
from attention_model import train_attention_model

if __name__ == "__main__":
    file_path = "datas/Small sample data.xlsx"
    X_train, X_test, y_train, y_test = load_and_process_data(file_path)

    print("\n===== 训练决策树模型 =====")
    dt_model = train_decision_tree(X_train, X_test, y_train, y_test)

    print("\n===== 训练随机森林模型 =====")
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    print("\n===== 训练XGBoost模型 =====")
    xgb_model = train_xgboost(X_train, X_test, y_train, y_test)

    print("\n===== 神经网络模型训练 =====")
    nn_model = train_neural_network(X_train, X_test, y_train, y_test)

    print("\n===== 训练注意力机制模型 =====")
    attn_model = train_attention_model(X_train, X_test, y_train, y_test)

    # 假设注意力机制模型是最佳模型
    best_model = attn_model

    # 保存最佳模型
    torch.save(best_model.state_dict(), "model/best_model.pth")

    # 转换为Torch Script
    example_input = torch.rand(1, X_train.shape[1])  # 示例输入
    scripted_model = torch.jit.trace(best_model, example_input)
    scripted_model.save("model/best_model_scripted.pt")

    # 转换为ONNX
    torch.onnx.export(
        best_model,
        example_input,
        "model/best_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )

    print("最佳模型已保存并转换为Torch Script和ONNX格式。")
