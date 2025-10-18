
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import multivariate_normal, norm


class GMM:
    def __init__(self, n, d, m):
        self.n = n
        self.d = d
        self.m = m

        self.pi = np.ones(m) / m
        self.mu = np.random.rand(m, d) * 20 - 10
        self.sigma = np.zeros((m, d, d))

        for j in range(m):
            self.sigma[j] = np.eye(d, d)

        self.h = np.zeros((n, m))

    def e_step(self, x):
        for j in range(self.m):
            cov_stabilized = self.sigma[j] + np.eye(self.d) * 1e-6
            self.h[:, j] = self.pi[j] * multivariate_normal.pdf(x, mean=self.mu[j], cov=cov_stabilized, allow_singular=True)
        
        row_sums = self.h.sum(axis=1, keepdims=True)
        self.h = np.divide(self.h, row_sums, out=np.zeros_like(self.h), where=row_sums!=0)

    def m_step(self, x):
        sum_h = self.h.sum(axis=0)

        self.pi = sum_h / self.n
        self.mu = np.divide(np.matmul(self.h.T, x), sum_h[:, None], out=np.zeros_like(self.mu), where=sum_h[:, None]!=0)

        for j in range(self.m):
            diff = x - self.mu[j]
            
            if self.d == 1:
                # Per 1D, sum_h[j] potrebbe essere zero se un cluster è vuoto
                if sum_h[j] > 1e-6:
                    weighted_sq_diff = self.h[:, j] * diff.flatten()**2
                    self.sigma[j, 0, 0] = weighted_sq_diff.sum() / sum_h[j]
            else:
                if sum_h[j] > 1e-6:
                    diff_exp = np.expand_dims(diff, axis=2)
                    sq = np.matmul(diff_exp, diff_exp.transpose((0, 2, 1)))
                    weighted_sq = sq * self.h[:, j].reshape(-1, 1, 1)
                    self.sigma[j] = weighted_sq.sum(axis=0) / sum_h[j]

    def q(self, data):
        q_val = 0
        for i in range(data.shape[0]):
            log_likelihood = 0
            for j in range(self.m):
                cov_stabilized = self.sigma[j] + np.eye(self.d) * 1e-6
                pdf_val = self.pi[j] * multivariate_normal.pdf(data[i], mean=self.mu[j], cov=cov_stabilized, allow_singular=True)
                # Usiamo h calcolato per calcolare q
                if self.h[i, j] > 1e-9 and pdf_val > 1e-9: # Evitiamo log(0)
                    log_likelihood += self.h[i, j] * np.log(pdf_val)
            q_val += log_likelihood
        return q_val


def gen_data(m=3, d=2, points_per_cluster=200, lim=(-10, 10)):
    x = []
    mean = np.random.rand(m, d) * (lim[1] - lim[0]) + lim[0]
    for j in range(m):
        cov = np.random.rand(d, d + np.random.randint(1, 11)) * 2 - 1
        cov = np.matmul(cov, cov.T)
        x += list(np.random.multivariate_normal(mean[j], cov, points_per_cluster))
    return np.array(x)


def plot_gaussian(mean, cov, ax, n_std=2.0, **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot(x, gmm, title, colors, **kwargs):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.scatter(x[:, 0], x[:, 1], s=3, alpha=0.4)
    ax.scatter(gmm.mu[:, 0], gmm.mu[:, 1], c=colors)
    for j in range(gmm.m):
        plot_gaussian(gmm.mu[j], gmm.sigma[j], ax, edgecolor=colors[j], lw=3, **kwargs)
    ax.set_xlim((-12, 12))
    ax.set_ylim((-12, 12))
    plt.title(title)
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.clf()


def plot_1d_gmm(x, gmm, title):
    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=30, density=True, alpha=0.6, color='skyblue', label='Distribuzione Dati (Istogramma)')
    x_range = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    
    # Ordiniamo i cluster per la legenda del grafico in base alla media
    sorted_indices = np.argsort(gmm.mu.flatten())
    colors = plt.cm.viridis(np.linspace(0, 1, gmm.m))
    
    for i, j in enumerate(sorted_indices):
        mu = gmm.mu[j, 0]
        sigma = np.sqrt(gmm.sigma[j, 0, 0])
        pi = gmm.pi[j]
        pdf = pi * norm.pdf(x_range, mu, sigma)
        plt.plot(x_range, pdf, lw=3, color=colors[i], label=f'Stagione {i+1} (stima)')
        
    plt.title(title)
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Densità di Probabilità')
    plt.legend()
    plt.savefig('grafico_istogramma_temperature.png')
    plt.clf()
    print("\nGrafico istogramma salvato come 'grafico_istogramma_temperature.png'")

# --- NUOVA FUNZIONE PER IL GRAFICO A PUNTI COLORATI ---
def plot_1d_scatter_clusters(x, gmm, title):
    """
    Crea uno strip plot per un GMM a una dimensione.
    Ogni punto dato viene colorato in base al cluster di appartenenza.
    """
    plt.figure(figsize=(12, 4))
    
    # 1. Determina l'assegnazione di ogni punto al cluster più probabile (hard assignment)
    cluster_assignments = np.argmax(gmm.h, axis=1)
    
    # Ordiniamo i cluster per avere colori consistenti (dal più freddo al più caldo)
    sorted_indices = np.argsort(gmm.mu.flatten())
    # Creiamo una mappa per riordinare le assegnazioni
    # Es: se il cluster più freddo era il n.2, ora diventerà il n.0
    remap = {original_idx: new_idx for new_idx, original_idx in enumerate(sorted_indices)}
    remapped_assignments = np.array([remap[old_idx] for old_idx in cluster_assignments])
    
    # 2. Definiamo i colori per le stagioni
    # Blu (inverno), Verde (primavera), Giallo (autunno), Rosso (estate)
    # Li mettiamo in ordine cromatico per matchare le temperature
    colors = ['#0000FF', '#4CBB17', '#FFC72C', '#FF0000']
    
    # 3. Creiamo un "jitter plot"
    # L'asse X è la temperatura. L'asse Y è un valore casuale piccolo per non far sovrapporre i punti.
    jitter = np.random.uniform(-0.1, 0.1, size=x.shape[0])
    
    for i in range(gmm.m):
        # Seleziona solo i punti che appartengono al cluster corrente
        points_in_cluster = x[remapped_assignments == i]
        jitter_in_cluster = jitter[remapped_assignments == i]
        
        # Disegna i punti del cluster con il colore appropriato
        plt.scatter(points_in_cluster, jitter_in_cluster, alpha=0.5, color=colors[i], label=f'Stagione {i+1} (stima)')
        
    plt.yticks([]) # Nascondiamo l'asse Y che non ha significato
    plt.xlabel('Temperatura (°C)')
    plt.title(title)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Salva il grafico
    plt.savefig('grafico_punti_stagioni.png')
    plt.clf()
    print("Grafico a punti colorati salvato come 'grafico_punti_stagioni.png'")