import numpy as np
from pathlib import Path
from astropy.io import fits
from multiprocessing import Pool, cpu_count
import time

# Extraccion de caracteristicas

def extract_features(wav, flux):
    """Extrae 10 características espectrales robustas"""
    flux_norm = flux / np.max(flux)
    
    features = {}
    
    # 1. Índices de color
    idx_4200 = np.argmin(np.abs(wav - 4200))
    idx_5500 = np.argmin(np.abs(wav - 5500))
    idx_7000 = np.argmin(np.abs(wav - 7000))
    idx_8500 = np.argmin(np.abs(wav - 8500))
    
    features['ci_4200_7000'] = flux_norm[idx_4200] / flux_norm[idx_7000]
    features['ci_5500_8500'] = flux_norm[idx_5500] / flux_norm[idx_8500]
    
    # 2. Profundidad de líneas
    def line_depth(wav_center, width=50):
        idx = np.argmin(np.abs(wav - wav_center))
        linea = flux_norm[idx]
        cont = np.median(flux_norm[max(0, idx-width):min(len(flux_norm), idx+width)])
        return 1.0 - (linea / cont) if cont > 0 else 0
    
    features['caII_K'] = line_depth(3934, width=30)
    features['caII_H'] = line_depth(3968, width=30)
    features['Hbeta'] = line_depth(4861, width=50)
    features['Halpha'] = line_depth(6563, width=50)
    
    # 3. Pendiente espectral (Balmer jump)
    balmer_region = flux_norm[(wav > 3600) & (wav < 4200)]
    if len(balmer_region) > 0:
        features['balmer_jump'] = np.mean(balmer_region)
    else:
        features['balmer_jump'] = 0
    
    # 4. Continuidad infrarroja
    ir_region = flux_norm[(wav > 7000) & (wav < 8500)]
    features['ir_strength'] = np.mean(ir_region) if len(ir_region) > 0 else 0
    
    # 5. Gradiente UV-óptico
    uv_region = flux_norm[(wav > 3800) & (wav < 4500)]
    opt_region = flux_norm[(wav > 5500) & (wav < 6500)]
    features['uv_opt_ratio'] = np.mean(uv_region) / np.mean(opt_region) if np.mean(opt_region) > 0 else 0
    
    return np.array([features[k] for k in sorted(features.keys())])


def load_and_extract_single_file(ruta):
    """Función auxiliar para extraer características de UN archivo"""
    try:
        with fits.open(ruta) as hdul:
            hdr = hdul[1].header
            sptype = str(hdr.get("SPTYPE", "??"))
            real_type = sptype.split()[0]
            
            data = hdul[1].data
            wav = data["wavelength"][0]
            flux = data["spectrum"][0]
            flux = flux / np.max(flux)
            
            features = extract_features(wav, flux)
            return features, real_type, ruta.name
    except Exception as e:
        print(f"Error processing {ruta}: {e}")
        return None


class SimpleSOM:
    def __init__(self, map_width=13, map_height=13, feature_dim=9, learning_rate=0.5):
        self.map_width = map_width
        self.map_height = map_height
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        
        np.random.seed(42)
        self.weights = np.random.randn(map_height, map_width, feature_dim) * 0.5
        
        self.type_map = np.full((map_height, map_width), "?", dtype=object)
        self.count_map = np.zeros((map_height, map_width), dtype=int)
    
    def euclidean_distance_vectorized(self, x, weights_flat):
        """Calcula distancias vectorizadas para todas las neuronas"""
        return np.sqrt(np.sum((x - weights_flat) ** 2, axis=1))
    
    def find_bmu(self, x):
        """Encuentra la Best Matching Unit (neurona más cercana) - vectorizada"""
        weights_reshaped = self.weights.reshape(-1, self.feature_dim)
        distances = self.euclidean_distance_vectorized(x, weights_reshaped)
        
        bmu_flat_idx = np.argmin(distances)
        bmu_idx = np.unravel_index(bmu_flat_idx, (self.map_height, self.map_width))
        return bmu_idx, distances[bmu_flat_idx]
    
    def gaussian_kernel(self, center, position, sigma):
        """Kernel gaussiano para actualizar vecindario"""
        center = np.array(center)
        position = np.array(position)
        distance = np.sqrt(np.sum((center - position) ** 2))
        return np.exp(-(distance ** 2) / (2 * (sigma ** 2)))
    
    def train(self, data, epochs=100, n_processes=4):
        """Entrena el SOM con paralelización con multiprocessing
        
        n_processes: número de procesos paralelos para procesar muestras
        """
        n_samples = data.shape[0]
        
        print(f"[v0] Entrenando SOM con {n_processes} procesos paralelos")
        
        for epoch in range(epochs):
            sigma = 1.0 * np.exp(-epoch / (epochs / 3))
            learning_rate = self.learning_rate * np.exp(-epoch / epochs)
            
            indices = np.random.permutation(n_samples)
            
            if n_processes > 1:
                sample_data = [(data[idx], sigma, learning_rate) for idx in indices]
                
                with Pool(processes=n_processes) as pool:
                    updates = pool.starmap(self._process_sample_static, sample_data)
                
                # Aplicar updates en secuencia (el SOM es inherentemente secuencial)
                for bmu_idx, influence_map, delta in updates:
                    for i in range(self.map_height):
                        for j in range(self.map_width):
                            self.weights[i, j] += learning_rate * influence_map[i, j] * delta
            else:
                # Versión secuencial
                for idx in indices:
                    x = data[idx]
                    bmu_idx, _ = self.find_bmu(x)
                    
                    for i in range(self.map_height):
                        for j in range(self.map_width):
                            influence = self.gaussian_kernel(bmu_idx, (i, j), sigma)
                            self.weights[i, j] += learning_rate * influence * (x - self.weights[i, j])
            
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"[v0] Epoch {epoch + 1}/{epochs} - sigma: {sigma:.3f}, lr: {learning_rate:.4f}")
    
    def _process_sample_static(self, x, sigma, learning_rate):
        """Procesa una muestra para cálculo paralelo (debe ser static para multiprocessing)"""
        bmu_idx, _ = self.find_bmu(x)
        
        # Precomputar la influencia del vecindario
        influence_map = np.zeros((self.map_height, self.map_width))
        for i in range(self.map_height):
            for j in range(self.map_width):
                influence_map[i, j] = self.gaussian_kernel(bmu_idx, (i, j), sigma)
        
        return bmu_idx, influence_map, (x - self.weights[bmu_idx])
    
    def predict(self, x):
        """Predice el tipo basándose en el tipo asignado a la BMU"""
        bmu_idx, _ = self.find_bmu(x)
        return self.type_map[bmu_idx]
    
    def assign_types(self, training_data, training_labels):
        """Asigna tipos espectrales a las neuronas basándose en datos de entrenamiento"""
        self.type_map[:] = "?"
        self.count_map[:] = 0
        
        for x, label in zip(training_data, training_labels):
            bmu_idx, _ = self.find_bmu(x)
            if self.count_map[bmu_idx] == 0:
                self.type_map[bmu_idx] = label[0]
            self.count_map[bmu_idx] += 1


def procesar_carpeta_paralelo(carpeta, train_epochs=200, n_processes=4):
    """Procesa FITS y entrena SOM con paralelización con multiprocessing
    
    n_processes: procesos para entrenamiento SOM (extracción siempre secuencial)
    """
    
    rutas = sorted(Path(carpeta).glob("*.fits"))
    
    if len(rutas) == 0:
        print(f"No FITS files found in {carpeta}")
        return
    
    print("=" * 80)
    print("FASE 1: Extrayendo características espectrales (secuencial)...")
    print("=" * 80)
    
    start_time = time.time()
    
    training_data = []
    training_labels = []
    filenames = []
    
    for i, ruta in enumerate(rutas, 1):
        result = load_and_extract_single_file(ruta)
        if result is not None:
            features, label, filename = result
            training_data.append(features)
            training_labels.append(label)
            filenames.append(filename)
            print(f"  [{i}/{len(rutas)}] {filename}")
    
    training_data = np.array(training_data)
    extract_time = time.time() - start_time
    print(f"✓ Características extraídas en {extract_time:.2f}s")
    
    # Normalizar características
    feature_mean = np.mean(training_data, axis=0)
    feature_std = np.std(training_data, axis=0)
    feature_std[feature_std == 0] = 1.0
    training_data_norm = (training_data - feature_mean) / feature_std
    
    print(f"✓ Datos cargados: {len(training_data)} espectros")
    print(f"✓ Características extraídas: {training_data.shape[1]}")
    
    # Fase 2: Entrenar SOM con multiprocessing
    print("\n" + "=" * 80)
    print(f"FASE 2: Entrenando SOM ({n_processes} procesos)...")
    print("=" * 80)
    
    som = SimpleSOM(map_width=13, map_height=13, feature_dim=training_data.shape[1], learning_rate=0.5)
    
    start_time = time.time()
    som.train(training_data_norm, epochs=train_epochs, n_processes=n_processes)
    train_time = time.time() - start_time
    
    som.assign_types(training_data_norm, training_labels)
    print(f"✓ SOM entrenado en {train_time:.2f}s ({som.map_height}x{som.map_width}, {train_epochs} épocas)")
    
    # Fase 3: Evaluación
    print("\n" + "=" * 80)
    print("FASE 3: Clasificación y Evaluación")
    print("=" * 80)
    
    aciertos = 0
    total = 0
    
    print(f"{'Archivo'.ljust(25)} | {'Real'.ljust(10)} | {'Predicho'.ljust(10)} | {'Resultado'}")
    print("-" * 60)
    
    for fname, features, real in zip(filenames, training_data_norm, training_labels):
        pred = som.predict(features)
        correcto = (real == pred) or (real.startswith(pred) if pred != "?" else False)
        
        if correcto:
            aciertos += 1
            flag = "✓"
        else:
            flag = "✗"
        
        total += 1
        print(f"{fname.ljust(25)} | {real.ljust(10)} | {pred.ljust(10)} | {flag}")
    
    print("-" * 60)
    exactitud = aciertos / total if total > 0 else 0
    print(f"Exactitud Total: {aciertos}/{total} = {exactitud:.1%}")
    
    print("\n" + "=" * 80)
    print(f"Tiempos totales: Extracción={extract_time:.2f}s, Entrenamiento={train_time:.2f}s")
    print("=" * 80)
    
    return som, feature_mean, feature_std


if __name__ == "__main__":
    print("[v0] SOM con Multiprocessing - Clasificación de Espectros Astronómicos")
    
    # Usar 4 procesos para entrenamiento (ajusta según tu sistema)
    procesar_carpeta_paralelo("data/BINTABLE", train_epochs=100, n_processes=4)
