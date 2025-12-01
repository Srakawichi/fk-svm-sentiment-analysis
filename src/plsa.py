# src/plsa.py

import numpy as np
from scipy.sparse import csr_matrix

class PLSA:
    def __init__(self, num_topics=20, iterations=20):
        self.num_topics = num_topics
        self.iterations = iterations
        self.P_w_z = None     # P(w | z)   shape: (K, V)
        self.P_z_d = None     # P(z | d)   shape: (D, K)

    def _em(self, X_dense, P_w_z, P_z_d, update_P_w_z=True, update_P_z_d=True):
        """
        Ein EM-Schritt (oder mehrere) für PLSA.
        X_dense: np.array mit shape (num_docs, num_words)
        P_w_z:   (K, V)
        P_z_d:   (D, K)
        """
        num_docs, num_words = X_dense.shape
        K = self.num_topics

        for it in range(self.iterations):
            print(f"[PLSA] Starting iteration {it+1}/{self.iterations}...")
            # E-Step: P(z | d, w)
            # shape: (D, V, K) -> Achtung: nur für kleinere D/V sinnvoll
            P_z_dw = np.zeros((num_docs, num_words, K), dtype=np.float64)

            for d in range(num_docs):
                for w in range(num_words):
                    if X_dense[d, w] == 0:
                        continue
                    # P(w | d) = sum_z P(z|d) P(w|z)
                    denom = np.dot(P_z_d[d, :], P_w_z[:, w])
                    if denom <= 0:
                        continue
                    for z in range(K):
                        P_z_dw[d, w, z] = P_z_d[d, z] * P_w_z[z, w] / denom
            print(f"[PLSA] E-Step {it+1}/{self.iterations} completed.")

            # M-Step: Parameter updaten
            if update_P_w_z:
                for z in range(K):
                    for w in range(num_words):
                        # Summe über alle Doks der erwarteten Count-Beiträge
                        s = 0.0
                        for d in range(num_docs):
                            if X_dense[d, w] == 0:
                                continue
                            s += X_dense[d, w] * P_z_dw[d, w, z]
                        P_w_z[z, w] = s
                    # normieren
                    row_sum = P_w_z[z, :].sum()
                    if row_sum > 0:
                        P_w_z[z, :] /= row_sum
            print(f"[PLSA] M-Step {it+1}/{self.iterations} completed.")

            if update_P_z_d:
                for d in range(num_docs):
                    for z in range(K):
                        s = 0.0
                        for w in range(num_words):
                            if X_dense[d, w] == 0:
                                continue
                            s += X_dense[d, w] * P_z_dw[d, w, z]
                        P_z_d[d, z] = s
                    row_sum = P_z_d[d, :].sum()
                    if row_sum > 0:
                        P_z_d[d, :] /= row_sum

        return P_w_z, P_z_d

    def fit(self, X: csr_matrix):
        """
        Trainiert PLSA auf der Trainings-BoW-Matrix X.
        X: csr_matrix (num_docs, vocab_size)
        Gibt zurück: P_z_d (Doc-Topic-Matrix), shape (num_docs, num_topics)
        """
        X_dense = X.toarray().astype(np.float64)
        num_docs, num_words = X_dense.shape

        # Zufällige Initialisierung
        P_w_z = np.random.rand(self.num_topics, num_words)
        P_w_z /= P_w_z.sum(axis=1, keepdims=True)  # P(w|z)

        P_z_d = np.random.rand(num_docs, self.num_topics)
        P_z_d /= P_z_d.sum(axis=1, keepdims=True)  # P(z|d)

        # EM vollständig (P_w_z und P_z_d updaten)
        P_w_z, P_z_d = self._em(
            X_dense,
            P_w_z,
            P_z_d,
            update_P_w_z=True,
            update_P_z_d=True
        )

        self.P_w_z = P_w_z
        self.P_z_d = P_z_d

        return self.P_z_d

    def transform(self, X: csr_matrix):
        """
        Berechnet P(z|d) für neue Dokumente X,
        bei fixiertem P(w|z) aus dem Training.
        """
        if self.P_w_z is None:
            raise ValueError("PLSA not fitted. Call fit() first.")

        X_dense = X.toarray().astype(np.float64)
        num_docs, num_words = X_dense.shape
        K, V = self.P_w_z.shape

        if num_words != V:
            raise ValueError("Vokabulargröße von X passt nicht zum trainierten PLSA.")

        P_z_d_new = np.random.rand(num_docs, K)
        P_z_d_new /= P_z_d_new.sum(axis=1, keepdims=True)

        _, P_z_d_new = self._em(
            X_dense,
            self.P_w_z,
            P_z_d_new,
            update_P_w_z=False,
            update_P_z_d=True
        )

        return P_z_d_new