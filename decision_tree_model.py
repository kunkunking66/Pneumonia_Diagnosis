from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report


# 决策树训练函数
def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("决策树准确率:", accuracy)
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    rules = export_text(model, feature_names=list(X_train.columns))
    print("决策树规则:")
    print(rules)
    return model
