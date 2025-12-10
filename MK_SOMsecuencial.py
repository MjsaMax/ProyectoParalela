import numpy as np
from pathlib import Path
from astropy.io import fits
import time
#Extraccion de caracteristicas ( )

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
    
    features['caII_K'] = line_depth(3934, width=30)        # Ca II K
    features['caII_H'] = line_depth(3968, width=30)        # Ca II H
    features['Hbeta'] = line_depth(4861, width=50)         # H-beta
    features['Halpha'] = line_depth(6563, width=50)        # H-alpha
    
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


class SimpleSOM:
    def __init__(self, map_width=13, map_height=13, feature_dim=9, learning_rate=0.5):
        self.map_width = map_width
        self.map_height = map_height
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        
        # Inicializar pesos aleatoriamente
        np.random.seed(42)
        self.weights = np.random.randn(map_height, map_width, feature_dim) * 0.5
        
        # Mapa de tipos espectrales (se llena durante entrenamiento)
        self.type_map = np.full((map_height, map_width), "?", dtype=object)
        self.count_map = np.zeros((map_height, map_width), dtype=int)
        
    def euclidean_distance(self, x, w):
        """Distancia euclidiana entre vector y pesos"""
        return np.sqrt(np.sum((x - w) ** 2))
    
    def find_bmu(self, x):
        """Encuentra la Best Matching Unit (neurona más cercana)"""
        distances = np.zeros((self.map_height, self.map_width))
        for i in range(self.map_height):
            for j in range(self.map_width):
                distances[i, j] = self.euclidean_distance(x, self.weights[i, j])
        
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx, np.min(distances)
    
    def gaussian_kernel(self, center, position, sigma):
        """Kernel gaussiano para actualizar vecindario"""
        center = np.array(center)
        position = np.array(position)
        distance = np.sqrt(np.sum((center - position) ** 2))
        return np.exp(-(distance ** 2) / (2 * (sigma ** 2)))
    
    def train(self, data, epochs=100):
        """Entrena el SOM"""
        n_samples = data.shape[0]
        
        for epoch in range(epochs):
            sigma = 1.0 * np.exp(-epoch / (epochs / 3))
            learning_rate = self.learning_rate * np.exp(-epoch / epochs)
            
            for sample_idx in np.random.permutation(n_samples):
                x = data[sample_idx]
                bmu_idx, _ = self.find_bmu(x)
                
                # Actualizar pesos en el vecindario
                for i in range(self.map_height):
                    for j in range(self.map_width):
                        influence = self.gaussian_kernel(bmu_idx, (i, j), sigma)
                        self.weights[i, j] += learning_rate * influence * (x - self.weights[i, j])
    
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
                self.type_map[bmu_idx] = label[0]  # Usar solo el tipo (F, G, K, M)
            self.count_map[bmu_idx] += 1
    
    def save_weights(self, filepath):
        """Guarda los pesos en formato numpy"""
        np.save(filepath, self.weights)
    
    def load_weights(self, filepath):
        """Carga los pesos desde archivo"""
        self.weights = np.load(filepath)

# Procesa y clasifica

def procesar_carpeta(carpeta, train_epochs=200): #200 iteraciones predeterminado
    """Procesa FITS y entrena SOM"""
    rutas = sorted(Path(carpeta).glob("*.fits"))
    
    if len(rutas) == 0:
        print(f"No FITS files found in {carpeta}")
        return
    
    # Fase 1: Cargar datos y extraer características
    print("=" * 80)
    print("FASE 1: Extrayendo características espectrales...")
    print("=" * 80)
    
    training_data = []
    training_labels = []
    
    for ruta in rutas:
        with fits.open(ruta) as hdul:
            hdr = hdul[1].header
            sptype = str(hdr.get("SPTYPE", "??"))
            real_type = sptype.split()[0]
            
            data = hdul[1].data
            wav = data["wavelength"][0]
            flux = data["spectrum"][0]
            flux = flux / np.max(flux)
            
            features = extract_features(wav, flux)
            training_data.append(features)
            training_labels.append(real_type)
    
    training_data = np.array(training_data)
    
    # Normalizar características
    feature_mean = np.mean(training_data, axis=0)
    feature_std = np.std(training_data, axis=0)
    feature_std[feature_std == 0] = 1.0
    training_data_norm = (training_data - feature_mean) / feature_std
    
    print(f"Datos cargados: {len(training_data)} espectros")
    print(f"Características extraídas: {training_data.shape[1]}")
    
    # Fase 2: Entrenar SOM
    print("\n" + "=" * 80)
    print("FASE 2: Entrenando SOM...")
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
    
    for ruta, features, real in zip(rutas, training_data_norm, training_labels):
        pred = som.predict(features)
        
        # Comparación: verificar si el tipo principal coincide
        correcto = (real == pred) or (real.startswith(pred) if pred != "?" else False)
        
        if correcto:
            aciertos += 1
            flag = "✓"
        else:
            flag = "✗"
        
        total += 1
        print(f"{ruta.name.ljust(25)} | {real.ljust(10)} | {pred.ljust(10)} | {flag}")
    
    print("-" * 60)
    exactitud = aciertos / total if total > 0 else 0
    print(f"Exactitud Total: {aciertos}/{total} = {exactitud:.1%}")
    
    # Guardar modelo
    # som.save_weights("som_model_weights.npy")
    # print("\nModelo SOM guardado en: som_model_weights.npy")
    
    return som, feature_mean, feature_std

if __name__ == "__main__":
    inicio=time.time()
    procesar_carpeta("data/BINTABLE", train_epochs=100)
    final=time.time()
    tiempo_total=final-inicio
    print(f"El entrenamiento tardó: {tiempo_total:.4f} s")
