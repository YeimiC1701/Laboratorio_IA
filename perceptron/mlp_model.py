from sklearn.neural_network import MLPClassifier

def create_mlp(hidden_layers=(10,), max_iter=500, learning_rate_init=0.01):
    return MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=max_iter, learning_rate_init=learning_rate_init, random_state=42)