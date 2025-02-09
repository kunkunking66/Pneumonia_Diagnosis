from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib

# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_roc_curve(fpr, tpr, auc, label):
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
    # 在曲线上添加文字标注
    plt.text(fpr.mean(), tpr.mean(), label, fontsize=9, color='black', ha='center', va='center')


def train_decision_tree(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]  # 获取正类概率

    accuracy = accuracy_score(y_test, y_pred)
    print("最佳决策树参数:", grid_search.best_params_)
    print("决策树准确率:", accuracy)
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    rules = export_text(best_model, feature_names=list(X_train.columns))
    print("决策树规则:")
    print(rules)

    # 计算ROC和AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plot_roc_curve(fpr, tpr, auc, label="决策树")

    return best_model, auc


def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取正类概率

    accuracy = accuracy_score(y_test, y_pred)
    print("随机森林准确率:", accuracy)
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    # 计算ROC和AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plot_roc_curve(fpr, tpr, auc, label="随机森林")

    return model, auc


def train_xgboost(X_train, X_test, y_train, y_test):
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取正类概率

    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost准确率:", accuracy)
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    # 计算ROC和AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plot_roc_curve(fpr, tpr, auc, label="XGBoost")

    return model, auc


def train_and_plot_roc(X_train, X_test, y_train, y_test):
    # 训练决策树并绘制ROC曲线
    train_decision_tree(X_train, X_test, y_train, y_test)

    # 训练随机森林并绘制ROC曲线
    train_random_forest(X_train, X_test, y_train, y_test)

    # 训练XGBoost并绘制ROC曲线
    train_xgboost(X_train, X_test, y_train, y_test)

    # 绘制斜率为1的参考曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    # 添加图例
    plt.legend(loc="lower right")

    # 显示图形
    plt.show()