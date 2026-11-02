import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1 : Data preparation
df = pd.read_csv("data_clean_v2.csv")

np.random.seed(42)
uniqueProfiles = df["profile_id"].unique()
np.random.shuffle(uniqueProfiles)

n_profiles = len(uniqueProfiles)
train_split = int(0.70*n_profiles)
val_split = int(0.85*n_profiles)

trainListProfile = uniqueProfiles[:train_split]
valListProfile = uniqueProfiles[train_split:val_split]
testListProfile = uniqueProfiles[val_split:]

df_train = pd.DataFrame(df[df["profile_id"].isin(trainListProfile)])
df_val = pd.DataFrame(df[df["profile_id"].isin(valListProfile)])
df_test = pd.DataFrame(df[df["profile_id"].isin(testListProfile)])

features = ['m','p','t','alpha']
targets = ['Cl','Cd']

def prepare_arrays(data_subset):
# Extract features : Matrix (N x 4)
  X = data_subset[features].values
# Extract targets : Matrix (N x 2)
# Reshaping into a (N x 1) vector
  cl = data_subset['Cl'].values.reshape(-1, 1)
# Take the logarithm of Cd to handle the large range of Cd values
  log_cd = np.log10(data_subset['Cd'].values.reshape(-1, 1))
# Horizontally stack the two target variables to form the matrix Y (N x 2)
  Y = np.hstack((cl, log_cd))
  return X, Y

X_train, Y_train = prepare_arrays(df_train)
X_val, Y_val = prepare_arrays(df_val)
X_test, Y_test = prepare_arrays(df_test)

# Standardisation on training data only
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0)

X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

Y_train = (Y_train - Y_mean) / Y_std
Y_val = (Y_val - Y_mean) / Y_std
Y_test = (Y_test - Y_mean) / Y_std


# Step 2 : Weights and bias initialisation

def init_weights(layer_dims):
  np.random.seed(42)
  parameters = {}
  L = len(layer_dims)  # Number of layers
  for l in range(1, L):
    # W_l and b_l are created for each layer
    # Xavier / Glorot Initialisation
    # layer_dims[l-1] => INPUT
    # layer_dims[l] => OUTPUT
    limit = np.sqrt(6/(layer_dims[l] + layer_dims[l-1]))
    W_l = np.random.uniform(-limit, limit, (layer_dims[l-1], layer_dims[l]))
    b_l = np.zeros((1,layer_dims[l]))
    parameters['W' + str(l)] = W_l
    parameters['b' + str(l)] = b_l
  return parameters

layer_dims = [4,64,64,2]
params = init_weights(layer_dims)

# Step 3 : Forward Propagation

def forward_propagation(X, parameters):
  caches = {}
  caches['Z1'] = np.dot(X,parameters['W1'])+parameters['b1']
  caches['A1'] = np.tanh(caches['Z1'])

  caches['Z2'] = np.dot(caches['A1'],parameters['W2'])+parameters['b2']
  caches['A2'] = np.tanh(caches['Z2'])

  caches['Z3'] = np.dot(caches['A2'],parameters['W3'])+parameters['b3']
  caches['A3'] = caches['Z3']

  return caches['A3'], caches

# Step 4 : Cost Function

def compute_cost(Y_hat, Y):
  m_samples = Y.shape[0]
  loss = np.sum((Y_hat - Y) ** 2) / (2 * m_samples) # MSE
  return loss

# Step 5 : Back Propagation

def backward_propagation(X, Y, parameters, caches):
  m_samples = X.shape[0]
  grads = {}

  A3 = caches['A3']
  A2 = caches['A2']
  A1 = caches['A1']

  # Gradient de la loss par rapport à la sortie A3 (activation linéaire)
  dZ3 = (A3 - Y) / m_samples

  # Gradients couche 3
  grads['dW3'] = np.dot(A2.T, dZ3)
  grads['db3'] = np.sum(dZ3, axis=0, keepdims=True)

  # Rétropropagation à travers la couche cachée 2 (activation tanh, dérivée = 1 - A2^2)
  dA2 = np.dot(dZ3, parameters['W3'].T)
  dZ2 = dA2 * (1 - A2**2)
  grads['dW2'] = np.dot(A1.T, dZ2)
  grads['db2'] = np.sum(dZ2, axis=0, keepdims=True)

  # Rétropropagation à travers la couche cachée 1 (activation tanh, dérivée = 1 - A1^2)
  dA1 = np.dot(dZ2, parameters['W2'].T)
  dZ1 = dA1 * (1 - A1**2)
  grads['dW1'] = np.dot(X.T, dZ1)
  grads['db1'] = np.sum(dZ1, axis=0, keepdims=True)

  return grads

# Step 6 : Updating weights (descente de gradient)
def update_parameters(parameters, grads, learning_rate):
  L = len(parameters) // 2 # Using //2 alows to directly have the number of layers
  for l in range(1, L + 1):
    parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
    parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
  return parameters

# Step 7 : Training loop
# AJOUT : X_val / Y_val en arguments, et suivi de la loss de validation.
# La validation n'est utilisée que pour CALCULER une loss : aucun gradient n'est calculé dessus, donc elle n'influence pas les poids.
 
def model(X, Y, X_val, Y_val, layer_dims, learning_rate=0.01, num_epochs=1000):
  # 1. Initialiser les poids et biais
  parameters = init_weights(layer_dims)
  train_costs = []
  val_costs = []
 
  for i in range(num_epochs):
    # 2. Forward propagation
    Y_hat, caches = forward_propagation(X, parameters)
 
    # 3. Cost function (loss)
    cost = compute_cost(Y_hat, Y)
 
    # 4. Backward propagation
    grads = backward_propagation(X, Y, parameters, caches)
 
    # 5. Updating parameters
    parameters = update_parameters(parameters, grads, learning_rate)
 
    # 6. Loss de validation (forward seulement, pas de backward)
    Y_hat_val, _ = forward_propagation(X_val, parameters)
    val_cost = compute_cost(Y_hat_val, Y_val)
 
    # On garde toutes les epochs (utile pour tracer la courbe / l'animation)
    train_costs.append(cost)
    val_costs.append(val_cost)
 
    if i % 100 == 0:
      print(f"Epoch {i:5d} | train {cost:.5f} | val {val_cost:.5f}")
 
  return parameters, train_costs, val_costs

def model_with_minibatch(X_train,Y_train,X_val,Y_val,layer_dims,learning_rate=0.01,num_epochs=1000,batch_size=64):
  # 1. Initialisation des poids
  parameters = init_weights(layer_dims)
  train_costs = []
  val_costs = []

  m_train = X_train.shape[0]

  for i in range(num_epochs):
    # Mélanger les données d'entraînement à chaque époque
    permutation = np.random.permutation(m_train)
    X_train_shuffled = X_train[permutation]
    Y_train_shuffled = Y_train[permutation]

    # Boucle sur les mini-batchs
    for j in range(0, m_train, batch_size):
      X_batch = X_train_shuffled[j : j + batch_size]
      Y_batch = Y_train_shuffled[j : j + batch_size]

      # Forward, Backward et Update sur le mini-batch courant
      _, caches = forward_propagation(X_batch, parameters)
      grads = backward_propagation(X_batch, Y_batch, parameters, caches)
      parameters = update_parameters(parameters, grads, learning_rate)

    # Suivi des performances à la fin de chaque époque (ou tous les X epochs)
    if i % 10 == 0 or i == num_epochs - 1:
      # Perte sur tout le jeu d'entraînement
      Y_hat_train, _ = forward_propagation(X_train, parameters)
      train_loss = compute_cost(Y_hat_train, Y_train)
      train_costs.append(train_loss)

      # Perte sur le jeu de validation
      Y_hat_val, _ = forward_propagation(X_val, parameters)
      val_loss = compute_cost(Y_hat_val, Y_val)
      val_costs.append(val_loss)

      if i % 100 == 0:
        print(
            f"Epoch {i}/{num_epochs} | Train Loss: {train_loss:.5f} | Val"
            f" Loss: {val_loss:.5f}"
        )
      epochs_axis = np.arange(len(train_costs)) * 10

  # Tracé des courbes sur la même figure
  plt.figure(figsize=(10, 6))
  plt.plot(epochs_axis, train_costs, label="Train loss", color="blue", linewidth=2)
  plt.plot(epochs_axis, val_costs, label="Validation loss", color="orange", linewidth=2, linestyle="--")
  plt.title("Loss evolution (mini-batch gradient descent)")
  plt.xlabel("Epochs")
  plt.ylabel("Loss (MSE)")
  plt.legend()
  plt.grid(True, linestyle=":", alpha=0.7)
  plt.tight_layout()
  plt.savefig("loss_curve.png", dpi=200)
  plt.show()

  return parameters, train_costs, val_costs
 
# Step 8 : Evaluation
# On revient aux unités physiques :
#   1. on inverse la standardisation : Y = Y_norm * std + mean
#   2. on repasse de log10(Cd) à Cd : Cd = 10 ** log10(Cd)
 
def evaluate(X, Y_norm, parameters, Y_mean, Y_std, name="TEST"):
  Y_hat_norm, _ = forward_propagation(X, parameters)
 
  Y_hat = Y_hat_norm * Y_std + Y_mean
  Y_true = Y_norm * Y_std + Y_mean
 
  cl_pred, cl_true = Y_hat[:, 0], Y_true[:, 0]
  cd_pred, cd_true = 10 ** Y_hat[:, 1], 10 ** Y_true[:, 1]
 
  def r2(t, p):
    return 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
 
  print(f"\n{name}  Cl : R2 = {r2(cl_true, cl_pred):.4f} | MAE = {np.mean(np.abs(cl_true - cl_pred)):.4f}")
  print(f"{name}  Cd : R2 = {r2(cd_true, cd_pred):.4f} | MAPE = {np.mean(np.abs(cd_true - cd_pred) / cd_true) * 100:.2f} %")
 
  return cl_true, cl_pred, cd_true, cd_pred

def gradient_check(X, Y, parameters, n_checks=5, eps=1e-5):
  # 1. Calculer les gradients analytiques avec votre backprop
  Y_hat, caches = forward_propagation(X, parameters)
  grads = backward_propagation(X, Y, parameters, caches)

  for _ in range(n_checks):
    # 2. Choisir un poids au hasard dans W1 (indices i, j)
    i = np.random.randint(parameters['W1'].shape[0])
    j = np.random.randint(parameters['W1'].shape[1])

    original = parameters['W1'][i, j]

    # 3. Décaler le poids de +eps, recalculer la loss
    parameters['W1'][i, j] = original + eps
    Y_hat_plus, _ = forward_propagation(X, parameters)
    loss_plus = compute_cost(Y_hat_plus, Y)

    # 4. Décaler de -eps, recalculer la loss
    parameters['W1'][i, j] = original - eps
    Y_hat_minus, _ = forward_propagation(X, parameters)
    loss_minus = compute_cost(Y_hat_minus, Y)

    # 5. IMPORTANT : remettre le poids d'origine
    parameters['W1'][i, j] = original

    # Gradient numérique (différence centrée) et gradient analytique (backprop)
    numeric = (loss_plus - loss_minus) / (2 * eps)
    analytic = grads['dW1'][i, j]

    # 6. Erreur relative
    rel_err = abs(numeric - analytic) / (abs(numeric) + abs(analytic) + 1e-12)
    print(f"W1[{i},{j}] analytique={analytic:.3e}  numérique={numeric:.3e}  erreur rel={rel_err:.2e}")

def parity_plot(cl_true, cl_pred, cd_true, cd_pred, title="TEST", save_path=None):
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  # --- Cl (échelle linéaire) ---
  ax = axes[0]
  ax.scatter(cl_true, cl_pred, s=8, alpha=0.5)
  lims = [min(cl_true.min(), cl_pred.min()), max(cl_true.max(), cl_pred.max())]
  ax.plot(lims, lims, "r--", label="y = x")
  ax.set_xlabel("Cl XFOIL")
  ax.set_ylabel("Cl predicted")
  ax.set_title(f"{title} : Cl")
  ax.legend()
  ax.grid(True, linestyle=":", alpha=0.7)
  ax.set_aspect("equal", adjustable="box")

  # --- Cd (échelle log) ---
  ax = axes[1]
  ax.scatter(cd_true, cd_pred, s=8, alpha=0.5)
  lims = [min(cd_true.min(), cd_pred.min()), max(cd_true.max(), cd_pred.max())]
  ax.plot(lims, lims, "r--", label="y = x")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("Cd XFOIL")
  ax.set_ylabel("Cd predicted")
  ax.set_title(f"{title} : Cd")
  ax.legend()
  ax.grid(True, which="both", linestyle=":", alpha=0.7)

  plt.tight_layout()
  if save_path is not None:
    plt.savefig(save_path, dpi=200)
  plt.show()