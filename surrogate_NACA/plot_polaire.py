import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data_clean_v2.csv')

unique_profiles = df['profile_id'].unique()
n_select = min(10, len(unique_profiles))
selected_profiles = np.random.choice(unique_profiles, size=n_select, replace=False)

print(f"Profile_id sélectionnés : {selected_profiles}")

plt.figure(figsize=(10, 6))

for pid in selected_profiles:
  # Filtrer et trier par alpha pour obtenir une courbe continue
  subset = df[df['profile_id'] == pid].sort_values('alpha')
  plt.plot(
      subset['alpha'],
      subset['Cd'],
      marker='o',
      linestyle='-',
      label=f'Profile ID {pid}',
  )

plt.title(
    'Coefficient de traînée $C_d$ en fonction de l’angle d’attaque $\\alpha$',
    fontsize=14,
)
plt.xlabel("Angle d'attaque $\\alpha$ (°)", fontsize=12)
plt.ylabel('Coefficient de traînée $C_d$', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('cd_alpha_plot.png', dpi=300)
plt.show()