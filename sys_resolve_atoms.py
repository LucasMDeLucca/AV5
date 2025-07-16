import numpy as np

class sys_resolve_atoms:
    def __init__(self, n):
        """
        Inicializa o sistema com n massas.

        Args:
            n (int): O número de massas (e o tamanho da matriz).
        """
        if n < 2:
            raise ValueError("O número de massas (n) deve ser pelo menos 2.")
        self.n = n
        self.matrix = np.zeros((n, n), dtype=float)
        self._autovalores = None
        self._autovetores = None

    def create_matrix(self, m: list, k: list, type = 'circular'):
        """
        Gera a matriz dinâmica D com base nas massas (m) e nas constantes de mola (k).
        """

        if type == 'circular':
            self.build_circular_matrix(m, k)

        elif type == 'linear':
            self.build_linear_matrix(m, k)
        else:
            raise ValueError("Tipo inválido. Use 'linear' ou 'circular'.")
        
        return self.matrix
    
    def build_linear_matrix(self, m: list, k: list) -> None:
        
        self._validate_inputs(m, k, expected_k_size=self.n - 1)

        self.matrix = np.zeros((self.n, self.n), dtype=float)

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    if i == 0:
                        self.matrix[i, j] = k[0] / m[i]
                    elif i == self.n - 1:
                        self.matrix[i, j] = k[i - 1] / m[i]
                    else:
                        self.matrix[i, j] = (k[i - 1] + k[i]) / m[i]
                elif abs(i - j) == 1:
                    k_index = min(i, j)
                    self.matrix[i, j] = -k[k_index] / m[i]

    
    def build_circular_matrix(self, m: list, k: list) -> None:

        self._validate_inputs(m, k, expected_k_size=self.n - 1)

        self.matrix = np.zeros((self.n, self.n), dtype=float)

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    if i == 0:
                        self.matrix[i, j] = (k[0] + k[self.n - 1])/ m[i]
                    elif i == self.n - 1:
                        self.matrix[i, j] = (k[i - 1] + k[self.n - 1]) / m[i]
                    else:
                        self.matrix[i, j] = (k[i - 1] + k[i]) / m[i]
                elif abs(i - j) == 1:
                    k_index = min(i, j)
                    self.matrix[i, j] = -k[k_index] / m[i]
                elif (i == self.n-1 and j == 0) or (i == 0 and j == self.n-1):
                    self.matrix[i, j] = -k[self.n - 1] / m[i]

    def _validate_inputs(self, m: list, k: list, expected_k_size: int) -> None:
        """Valida as listas de entrada."""
        if len(m) != self.n:
            raise ValueError(f"Lista 'm' deve ter {self.n} elementos.")
        if len(k) != expected_k_size:
            raise ValueError(f"Lista 'k' deve ter {expected_k_size} elementos.")
        
    def _calculate_system_values(self):
        """
        Método interno para calcular autovalores e autovetores
        """
        autovalores, autovetores = np.linalg.eig(self.matrix)
        
        sorted_indices = np.argsort(autovalores)
        self._autovalores = autovalores[sorted_indices]
        self._autovetores = autovetores[sorted_indices]

    def get_autovalores(self):
        """
        Calcula e retorna os autovalores (λ = ω²) da matriz dinâmica D.
        """
        self._calculate_system_values()
        return self._autovalores

    def get_autovetores(self):
        """
        Calcula e retorna os autovetores (modos normais de vibração) da matriz D.

        Returns:
            np.ndarray: Uma matriz onde cada coluna é um autovetor correspondente
                        a um autovalor (na mesma ordem retornada por get_autovalores).
        """
        self._calculate_system_values()
        return self._autovetores
    
    def get_frequencia_angular(self):
        self._calculate_system_values()
        autovalores_positivos = self._autovalores
        return autovalores_positivos
