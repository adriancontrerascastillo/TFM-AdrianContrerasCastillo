"""
Script para calcular métricas de ranking (Precision@10, Recall@10) 
para las variantes ItemKNN_k y SVD_f que actualmente solo tienen RMSE/MAE

VERSIÓN CORREGIDA: Calcula por usuario con threshold, igual que NCF
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, SVD
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("CÁLCULO DE MÉTRICAS DE RANKING PARA VARIANTES")
print("="*80)

DATA_PATH = Path("data/processed")
RESULTS_PATH = Path("results")

print("\n Cargando datos...")
test = pd.read_csv(DATA_PATH / "test.csv")
print(f"   Test set: {len(test)} interacciones")


def evaluate_ranking_metrics_corrected(model, test_df, train_df, k=10, threshold=3.5, verbose=False):
    """
    Calcula Precision@K y Recall@K CORRECTAMENTE
    
    Para cada usuario:
    1. Obtener sus ítems relevantes en test (rating >= threshold)
    2. Predecir scores para TODOS los ítems disponibles
    3. Tomar top-K predicciones
    4. Calcular precision y recall comparando con relevantes
    
    Esta es la forma correcta usada en el notebook NCF
    """
    
    all_items = sorted(train_df['movieId'].unique())
    test_users = test_df['userId'].unique()
    
    precisions = []
    recalls = []
    
    iterator = tqdm(test_users, desc="Evaluando usuarios") if verbose else test_users
    
    for uid in iterator:
        user_test_data = test_df[test_df['userId'] == uid]
        true_items = set(
            user_test_data[user_test_data['rating'] >= threshold]['movieId'].values
        )
        
        if len(true_items) == 0:
            continue
        
        true_items = true_items & set(all_items)
        
        if len(true_items) == 0:
            continue
        
        try:
            predictions = []
            for item in all_items:
                pred = model.predict(uid, item)
                predictions.append((pred.est, item))
            
            predictions.sort(reverse=True, key=lambda x: x[0])
            top_k_items = set([item for _, item in predictions[:k]])
            
            hits = top_k_items & true_items
            
            precision = len(hits) / k if k > 0 else 0.0
            recall = len(hits) / len(true_items) if len(true_items) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            
            if len(precisions) == 1 and (precision > 0 or recall > 0) and verbose:
                print(f"\n Primer hit encontrado!")
                print(f"   Usuario: {uid}")
                print(f"   Ítems relevantes: {len(true_items)}")
                print(f"   Top-{k} predichos: {list(top_k_items)[:5]}...")
                print(f"   Hits: {len(hits)}")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall: {recall:.4f}")
            
        except Exception as e:
            if verbose and len(precisions) < 5:
                print(f"\n  Error usuario {uid}: {e}")
            continue
    
    result = {
        'precision_at_k': np.mean(precisions) if precisions else 0.0,
        'recall_at_k': np.mean(recalls) if recalls else 0.0,
        'n_users_evaluated': len(precisions)
    }
    
    if verbose:
        print(f"\n Resultados de evaluación:")
        print(f"   Usuarios evaluados: {result['n_users_evaluated']}")
        print(f"   Precision@{k}: {result['precision_at_k']:.4f}")
        print(f"   Recall@{k}: {result['recall_at_k']:.4f}")
    
    return result


# ============================================================================
# PARTE 1: VARIANTES ItemKNN
# ============================================================================

print("\n" + "="*80)
print("PARTE 1: EVALUANDO VARIANTES ItemKNN")
print("="*80)

knn_results = []
k_values = [20, 30, 40]
fractions = [25, 50, 75]
K = 10
THRESHOLD = 3.5

for k_neighbors in k_values:
    for fraction in fractions:
        print(f"\n Evaluando ItemKNN k={k_neighbors}, fracción={fraction}%")
        
        train_file = DATA_PATH / "subsampled_informed" / f"ratings_top_items_{fraction}.csv"
        
        if not train_file.exists():
            print(f" Archivo no encontrado: {train_file}")
            continue
        
        train_frac = pd.read_csv(train_file)
        print(f"  Train: {len(train_frac)} ratings")
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_frac[['userId','movieId','rating']], reader)
        trainset = data.build_full_trainset()
        
        print(f"    Entrenando modelo...")
        model = KNNBasic(
            k=k_neighbors,
            sim_options={'name': 'cosine', 'user_based': False},
            verbose=False
        )
        model.fit(trainset)
        
        print(f"   Calculando Precision@{K} y Recall@{K} (threshold={THRESHOLD})...")
        metrics = evaluate_ranking_metrics_corrected(
            model, test, train_frac, k=K, threshold=THRESHOLD, verbose=True
        )
        
        result = {
            'method': f'ItemKNN_k{k_neighbors}',
            'k_neighbors': k_neighbors,
            'fraction': fraction,
            'precision_at_k': metrics['precision_at_k'],
            'recall_at_k': metrics['recall_at_k'],
            'n_users_evaluated': metrics['n_users_evaluated']
        }
        knn_results.append(result)
        
        print(f"    P@{K}={metrics['precision_at_k']:.4f}, R@{K}={metrics['recall_at_k']:.4f}")
        print(f"      Usuarios evaluados: {metrics['n_users_evaluated']}")

if knn_results:
    df_knn = pd.DataFrame(knn_results)
    output_file = RESULTS_PATH / "itemknn_variants_ranking.csv"
    df_knn.to_csv(output_file, index=False)
    print(f"\n Resultados ItemKNN guardados en: {output_file}")
    print("\n Resumen ItemKNN:")
    print(df_knn.to_string(index=False))


# ============================================================================
# PARTE 2: VARIANTES SVD
# ============================================================================

print("\n\n" + "="*80)
print("PARTE 2: EVALUANDO VARIANTES SVD")
print("="*80)

svd_results = []
n_factors_values = [20, 50, 100]
fractions = [25, 50, 75]
N_EPOCHS = 20
LEARNING_RATE = 0.005
K = 10
THRESHOLD = 3.5

for n_factors in n_factors_values:
    for fraction in fractions:
        print(f"\n Evaluando SVD n_factors={n_factors}, fracción={fraction}%")
        
        train_file = DATA_PATH / "subsampled_informed" / f"ratings_top_items_{fraction}.csv"
        
        if not train_file.exists():
            print(f"    Archivo no encontrado: {train_file}")
            continue
        
        train_frac = pd.read_csv(train_file)
        print(f"  Train: {len(train_frac)} ratings")
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_frac[['userId','movieId','rating']], reader)
        trainset = data.build_full_trainset()
        
        print(f"   Entrenando modelo...")
        model = SVD(
            n_factors=n_factors,
            n_epochs=N_EPOCHS,
            lr_all=LEARNING_RATE,
            random_state=42,
            verbose=False
        )
        model.fit(trainset)
        
        print(f"    Calculando Precision@{K} y Recall@{K} (threshold={THRESHOLD})...")
        metrics = evaluate_ranking_metrics_corrected(
            model, test, train_frac, k=K, threshold=THRESHOLD, verbose=True
        )
        
        result = {
            'method': f'SVD_f{n_factors}',
            'n_factors': n_factors,
            'fraction': fraction,
            'precision_at_k': metrics['precision_at_k'],
            'recall_at_k': metrics['recall_at_k'],
            'n_users_evaluated': metrics['n_users_evaluated']
        }
        svd_results.append(result)
        
        print(f"    P@{K}={metrics['precision_at_k']:.4f}, R@{K}={metrics['recall_at_k']:.4f}")
        print(f"      Usuarios evaluados: {metrics['n_users_evaluated']}")

if svd_results:
    df_svd = pd.DataFrame(svd_results)
    output_file = RESULTS_PATH / "svd_variants_ranking.csv"
    df_svd.to_csv(output_file, index=False)
    print(f"\n Resultados SVD guardados en: {output_file}")
    print("\n Resumen SVD:")
    print(df_svd.to_string(index=False))


# ============================================================================
# PARTE 3: INTEGRAR CON RESULTADOS EXISTENTES
# ============================================================================

print("\n\n" + "="*80)
print("PARTE 3: INTEGRANDO CON RESULTADOS EXISTENTES")
print("="*80)

knn_energy = pd.read_csv(RESULTS_PATH / "knn_energy_variants.csv")
svd_energy = pd.read_csv(RESULTS_PATH / "svd_energy_variants.csv")

if knn_results:
    df_knn_ranking = pd.DataFrame(knn_results)
    knn_complete = knn_energy.merge(
        df_knn_ranking[['method', 'fraction', 'precision_at_k', 'recall_at_k']],
        on=['method', 'fraction'],
        how='left'
    )
    output_file = RESULTS_PATH / "knn_energy_variants_complete.csv"
    knn_complete.to_csv(output_file, index=False)
    print(f"\n ItemKNN completo guardado en: {output_file}")
    print("\nPrimeras filas:")
    print(knn_complete.head(3).to_string(index=False))

if svd_results:
    df_svd_ranking = pd.DataFrame(svd_results)
    svd_complete = svd_energy.merge(
        df_svd_ranking[['method', 'fraction', 'precision_at_k', 'recall_at_k']],
        on=['method', 'fraction'],
        how='left'
    )
    output_file = RESULTS_PATH / "svd_energy_variants_complete.csv"
    svd_complete.to_csv(output_file, index=False)
    print(f"\n SVD completo guardado en: {output_file}")
    print("\nPrimeras filas:")
    print(svd_complete.head(3).to_string(index=False))

print("\n" + "="*80)
print(" PROCESO COMPLETADO")
print("="*80)
print("\nArchivos generados:")
print("  1. itemknn_variants_ranking.csv - Solo métricas de ranking")
print("  2. svd_variants_ranking.csv - Solo métricas de ranking")
print("  3. knn_energy_variants_complete.csv - TODAS las métricas")
print("  4. svd_energy_variants_complete.csv - TODAS las métricas")
print("\nAhora puedes actualizar la Tabla 3.1 con las métricas de ranking completas.")