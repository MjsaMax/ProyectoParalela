 #!/usr/bin/env python3
 """
 Script para inspeccionar archivos FITS y extraer información de clasificación MK
 """
 
 from astropy.io import fits
 import os
 import json
 
 def inspect_fits_file(filepath):
     """
     Inspecciona un archivo FITS y muestra toda su información
     """
     print(f"\n{'='*80}")
     print(f"INSPECCIONANDO: {filepath}")
     print(f"{'='*80}\n")
     
     try:
         # Abrir archivo FITS
         with fits.open(filepath) as hdul:
             print(f"Total de HDUs (extensiones): {len(hdul)}\n")
             
             # Iterar sobre todas las extensiones
             for i, hdu in enumerate(hdul):
                 print(f"\n{'─'*80}")
                 print(f"EXTENSIÓN {i}: {hdu.name if hdu.name else 'PRIMARY'}")
                 print(f"{'─'*80}")
                 print(f"Tipo: {type(hdu).__name__}")
                 
                 # Mostrar headers
                 if hdu.header:
                     print(f"\nHEADERS ({len(hdu.header)} campos):")
                     print("─" * 80)
                     
                     # Buscar campos relevantes
                     relevant_keywords = [
                         'SPTYPE', 'SPECTRAL', 'MK', 'SP_TYPE', 'SPECTRUM',
                         'TEFF', 'LOGG', 'FEH', 'STELLAR', 'OBJECT',
                         'COMMENT', 'HISTORY'
                     ]
                     
                     found_mk = False
                     
                     for key in relevant_keywords:
                         for card in hdu.header.cards:
                             if key.upper() in card.keyword.upper():
                                 print(f"{card.keyword:20s} = {str(card.value):50s} / {card.comment}")
                                 if key.upper() in ['SPTYPE', 'SPECTRAL', 'MK', 'SP_TYPE']:
                                     found_mk = True
                     
                     # Mostrar todos los headers si no encontramos nada
                     if not found_mk:
                         print("\nTODOS LOS HEADERS:")
                         for card in hdu.header.cards:
                             print(f"{card.keyword:20s} = {str(card.value):50s} / {card.comment}")
                 
                 # Mostrar datos si existen
                 if hdu.data is not None:
                     print(f"\nDATA:")
                     print(f"  Shape: {hdu.data.shape}")
                     print(f"  Dtype: {hdu.data.dtype}")
                     
                     # Si es tabla, mostrar columnas
                     if hasattr(hdu, 'columns'):
                         print(f"\n  Columnas:")
                         for col in hdu.columns:
                             print(f"    - {col.name}: {col.format}")
                         
                         # Mostrar primeras filas
                         print(f"\n  Primeras filas:")
                         for j in range(min(5, len(hdu.data))):
                             print(f"    Row {j}: {hdu.data[j]}")
     
     except Exception as e:
         print(f"ERROR al leer {filepath}: {str(e)}")
 
 def main():
     """
     Función principal
     """
     # Ruta de datos
     data_dir = "./data/BINTABLE"
     
     if not os.path.exists(data_dir):
         print(f"ERROR: Directorio {data_dir} no encontrado")
         return
     
     # Obtener lista de archivos FITS
     fits_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.fits')])[:5]
     
     if not fits_files:
         print(f"No se encontraron archivos .fits en {data_dir}")
         return
     
     print(f"\nEncontrados {len(fits_files)} archivos FITS para inspeccionar")
     print(f"{'='*80}\n")
     
     # Inspeccionar cada archivo
     for filename in fits_files:
         filepath = os.path.join(data_dir, filename)
         inspect_fits_file(filepath)
         print("\n")
 
 if __name__ == "__main__":
     main()
 
