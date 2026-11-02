import numpy as np

def naca4(m, p, t, n=100):
    """Coordonnées NACA 4 chiffres (m, p, t en fractions de corde), du BF extrados au BF intrados."""
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - 0.1036*x**4)  # bord de fuite fermé
    if m == 0:
        yc = np.zeros_like(x); dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m/p**2*(2*p*x - x**2),
                      m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
        dyc = np.where(x < p, 2*m/p**2*(p - x),
                       2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    X = np.concatenate([xu[::-1], xl[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y