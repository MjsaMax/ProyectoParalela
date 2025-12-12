import numpy as np
from pathlib import Path
from astropy.io import fits
import time
import threading
import os
from queue import Queue


def extract_features(wav, flux):
    # ... existing code ...
    flux_norm = flux / np.max(flux)
    
    features = {}
    
    idx_4200 = np.argmin(np.abs(wav - 4200))
    idx_5500 = np.argmin(np.abs(wav - 5500))
    idx_7000 = np.argmin(np.abs(wav - 7000))
    idx_8500 = np.argmin(np.abs(wav - 8500))
    
    features['ci_4200_7000'] = flux_norm[idx_4200] / flux_norm[idx_7000]
    features['ci_5500_8500'] = flux_norm[idx_5500] / flux_norm[idx_8500]
    
    def line_depth(wav_center, width=50):
        idx = np.argmin(np.abs(wav - wav_center))
        linea = flux_norm[idx]
        cont = np.median(flux_norm[max(0, idx-width):min(len(flux_norm), idx+width)])
        return 1.0 - (linea / cont) if cont > 0 else 0
    
    features['caII_K'] = line_depth(3934, width=30)
    features['caII_H'] = line_depth(3968, width=30)
    features['Hbeta'] = line_depth(4861, width=50)
    features['Halpha'] = line_depth(6563, width=50)
    
    balmer_region = flux_norm[(wav > 3600) & (wav < 4200)]
    if len(balmer_region) > 0:
        features['balmer_jump'] = np.mean(balmer_region)
    else:
        features['balmer_jump'] = 0
    
    ir_region = flux_norm[(wav > 7000) & (wav < 8500)]
    features['ir_strength'] = np.mean(ir_region) if len(ir_region) > 0 else 0
    
    uv_region = flux_norm[(wav > 3800) & (wav < 4500)]
    opt_region = flux_norm[(wav > 5500) & (wav < 6500)]
    features['uv_opt_ratio'] = np.mean(uv_region) / np.mean(opt_region) if np.mean(opt_region) > 0 else 0
    
    return np.array([features[k] for k in sorted(features.keys())])


def _load_fits_file(ruta, output_queue):
    """Carga archivo FITS en hilo"""
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
            output_queue.put((ruta, features, real_type))
    except Exception as e:
        print(f"Error loading {ruta}: {e}")


def _compute_distances_row(row_idx, x, weights_row, output_queue):
    """Calcula distancias en hilo"""
    distances = np.array([np.sqrt(np.sum((x - w) ** 2)) for w in weights_row])
    output_queue.put((row_idx, distances))


def _compute_bmu_batch(x, weights, map_height, map_width, output_queue, idx):
    """Encuentra BMU en hilo"""
    distances = np.zeros((map_height, map_width))
    
    for i in range(map_height):
        for j in range(map_width):
            distances[i, j] = np.sqrt(np.sum((x - weights[i, j]) ** 2))
    
    bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
    output_queue.put((idx, bmu_idx, np.min(distances)))


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
        self.lock = threading.Lock()
    
    def find_bmu(self, x):
        """Encuentra BMU - versión con threading por filas"""
        num_threads = os.cpu_count()
        output_queue = Queue()
        threads = []
        
        # <CHANGE> Crear hilos para cada fila en lugar de Pool
        for i in range(self.map_height):
            thread = threading.Thread(
                target=_compute_distances_row,
                args=(i, x, self.weights[i], output_queue)
            )
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen todos los hilos
        for thread in threads:
            thread.join()
        
        # Recopilar resultados
        distances = np.zeros((self.map_height, self.map_width))
        while not output_queue.empty():
            row_idx, row_distances = output_queue.get()
            distances[row_idx] = row_distances
        
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx, np.min(distances)
    
    def gaussian_kernel(self, center, position, sigma):
        """Kernel gaussiano para actualizar vecindario"""
        center = np.array(center)
        position = np.array(position)
        distance = np.sqrt(np.sum((center - position) ** 2))
        return np.exp(-(distance ** 2) / (2 * (sigma ** 2)))
    
    def train(self, data, epochs=100):
        """Entrena SOM con actualización en threading"""
        n_samples = data.shape[0]
        num_threads = os.cpu_count()
        
        for epoch in range(epochs):
            sigma = 1.0 * np.exp(-epoch / (epochs / 3))
            learning_rate = self.learning_rate * np.exp(-epoch / epochs)
            
            # Procesar muestras en lotes con BMU en threading
            lote_size = max(1, n_samples // (num_threads * 2))
            sample_indices = np.random.permutation(n_samples)
            
            for batch_start in range(0, n_samples, lote_size):
                batch_end = min(batch_start + lote_size, n_samples)
                batch_indices = sample_indices[batch_start:batch_end]
                
                output_queue = Queue()
                threads = []
                
                # <CHANGE> Crear hilos para cada muestra en lugar de Pool
                for idx, sample_idx in enumerate(batch_indices):
                    thread = threading.Thread(
                        target=_compute_bmu_batch,
                        args=(data[sample_idx], self.weights, self.map_height, 
                              self.map_width, output_queue, idx)
                    )
                    threads.append(thread)
                    thread.start()
                
                # Esperar a que terminen
                for thread in threads:
                    thread.join()
                
                # Recopilar y aplicar actualizaciones
                bmu_results = {}
                while not output_queue.empty():
                    idx, bmu_idx, _ = output_queue.get()
                    bmu_results[idx] = bmu_idx
                
                for idx, sample_idx in enumerate(batch_indices):
                    if idx in bmu_results:
                        bmu_idx = bmu_results[idx]
                        x = data[sample_idx]
                        for i in range(self.map_height):
                            for j in range(self.map_width):
                                influence = self.gaussian_kernel(bmu_idx, (i, j), sigma)
                                self.weights[i, j] += learning_rate * influence * (x - self.weights[i, j])
    
    def predict(self, x):
        """Predice el tipo basándose en la BMU"""
        bmu_idx, _ = self.find_bmu(x)
        return self.type_map[bmu_idx]
    
    def assign_types(self, training_data, training_labels):
        """Asigna tipos espectrales a las neuronas"""
        self.type_map[:] = "?"
        self.count_map[:] = 0
        
        for x, label in zip(training_data, training_labels):
            bmu_idx, _ = self.find_bmu(x)
            if self.count_map[bmu_idx] == 0:
                self.type_map[bmu_idx] = label[0]
            self.count_map[bmu_idx] += 1
    
    def save_weights(self, filepath):
        """Guarda los pesos en formato numpy"""
        np.save(filepath, self.weights)
    
    def load_weights(self, filepath):
        """Carga los pesos desde archivo"""
        self.weights = np.load(filepath)


def cargar_datos_paralelo(rutas):
    """Carga todos los archivos FITS con threading"""
    num_threads = os.cpu_count()
    output_queue = Queue()
    threads = []
    
    # <CHANGE> Crear hilos para cargar archivos en lugar de Pool
    for ruta in rutas:
        thread = threading.Thread(target=_load_fits_file, args=(ruta, output_queue))
        threads.append(thread)
        thread.start()
        
        # Limitar número de hilos activos simultáneamente
        if len(threads) >= num_threads:
            for t in threads:
                t.join()
            threads = []
    
    # Esperar a los hilos restantes
    for thread in threads:
        thread.join()
    
    # Recopilar resultados
    results = []
    while not output_queue.empty():
        results.append(output_queue.get())
    
    if not results:
        return None, None, None
    
    rutas_validas, features_list, labels = zip(*results)
    return list(rutas_validas), np.array(features_list), list(labels)


def procesar_carpeta(carpeta, train_epochs=200):
    """Procesa FITS y entrena SOM con threading puro"""
    rutas = sorted(Path(carpeta).glob("*.fits"))
    
    if len(rutas) == 0:
        print(f"No FITS files found in {carpeta}")
        return
    
    # Fase 1: Cargar datos con threading
    print("=" * 80)
    print("FASE 1: Extrayendo características espectrales (threading puro)...")
    print("=" * 80)
    
    rutas_validas, training_data, training_labels = cargar_datos_paralelo(rutas)
    
    if training_data is None:
        print("No valid data loaded")
        return
    
    # Normalizar características
    feature_mean = np.mean(training_data, axis=0)
    feature_std = np.std(training_data, axis=0)
    feature_std[feature_std == 0] = 1.0
    training_data_norm = (training_data - feature_mean) / feature_std
    
    print(f"Datos cargados: {len(training_data)} espectros")
    print(f"Características extraídas: {training_data.shape[1]}")
    
    # Fase 2: Entrenar SOM
    print("\n" + "=" * 80)
    print("FASE 2: Entrenando SOM (threading puro)...")
    print("=" * 80)
    
    som = SimpleSOM(map_width=13, map_height=13, feature_dim=training_data.shape[1], learning_rate=0.5)
    som.train(training_data_norm, epochs=train_epochs)
    som.assign_types(training_data_norm, training_labels)
    
    print(f"SOM entrenado ({som.map_height}x{som.map_width}, {train_epochs} épocas)")
    
    # Fase 3: Evaluación
    print("\n" + "=" * 80)
    print("FASE 3: Clasificación y Evaluación")
    print("=" * 80)
    
    aciertos = 0
    total = 0
    
    print(f"{'Archivo'.ljust(25)} | {'Real'.ljust(10)} | {'Predicho'.ljust(10)} | {'Resultado'}")
    print("-" * 60)
    
    for ruta, features, real in zip(rutas_validas, training_data_norm, training_labels):
        pred = som.predict(features)
        
        correcto = (real == pred) or (real.startswith(pred) if pred != "?" else False)
        
        if correcto:
            aciertos += 1
            flag = "V"
        else:
            flag = "X"
        
        total += 1
        print(f"{ruta.name.ljust(25)} | {real.ljust(10)} | {pred.ljust(10)} | {flag}")
    
    print("-" * 60)
    exactitud = aciertos / total if total > 0 else 0
    print(f"Exactitud Total: {aciertos}/{total} = {exactitud:.1%}")
    
    return som, feature_mean, feature_std


if __name__ == "__main__":
    for epochs in [20, 30, 50, 80]:
        inicio = time.time()
        procesar_carpeta("data/BINTABLE", train_epochs=epochs)
        final = time.time()
        tiempo_total = final - inicio
        print(f"El entrenamiento tardó: {tiempo_total:.4f} s\n")
