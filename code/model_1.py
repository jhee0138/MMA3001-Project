from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_decision_tree_env(X_train, y_train, max_depth=5, random_state=42):
    # Only keep environmental features if temporal ones are present
    env_features = ['co2', 'temperature', 'humidity', 'co2_rate_of_change']
    X_train_env = X_train[env_features] if all(col in X_train.columns for col in env_features) else X_train
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train_env, y_train)
    return model

def train_random_forest_env(X_train, y_train, n_estimators=100, max_depth=5, random_state=42):
    # Only keep environmental features if temporal ones are present
    env_features = ['co2', 'temperature', 'humidity', 'co2_rate_of_change']
    X_train_env = X_train[env_features] if all(col in X_train.columns for col in env_features) else X_train
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train_env, y_train)
    return model