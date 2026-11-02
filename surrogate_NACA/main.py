import numpy as np

from my_MLP_numpy import (
    X_train, Y_train, X_val, Y_val, X_test, Y_test,
    X_mean, X_std, Y_mean, Y_std,
    layer_dims,
    model_with_minibatch, evaluate, parity_plot,
    init_weights, gradient_check,
)

if __name__ == "__main__":

  # 1. Gradient check, on fresh weights (before training)
  test_params = init_weights(layer_dims)
  gradient_check(X_train[:200], Y_train[:200], test_params)

  # 2. Training
  params, train_c, val_c = model_with_minibatch(X_train, Y_train, X_val, Y_val, layer_dims,learning_rate=0.01, num_epochs=1000, batch_size=64)

  # 3. Metrics: validation, then test (once, at the very end)
  evaluate(X_val, Y_val, params, Y_mean, Y_std, name="VAL ")
  cl_t, cl_p, cd_t, cd_p = evaluate(X_test, Y_test, params, Y_mean, Y_std, name="TEST")

  # 4. Parity plot (saved as parity_test.png)
  parity_plot(cl_t, cl_p, cd_t, cd_p, title="TEST", save_path="parity_test.png")

  # 5. Save weights for the animation / benchmark
  np.savez("mlp_numpy_v2.npz", train_costs=train_c, val_costs=val_c,X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std, **params)
  print("\nWeights and loss curves saved in mlp_numpy_v2.npz")