


import torch
from scipy.spatial.distance import cdist
import math
import dice_ml
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
from dice_ml import Dice
import time
import pandas as pd
from xailib.explainers.lore_explainer import LoreTabularExplainer
from sklearn.metrics import accuracy_score,f1_score
import fatf.fairness.data.measures as fatf_dfm

import tensorflow as tf
from growingspheres import counterfactuals as cf
import multiprocessing as mp

from alibi.explainers.cfproto import CounterFactualProto

from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader

from ignite.metrics import Accuracy, Loss
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator



def stretch_dict_params(feature_names, weight_dict):

    stretched_means= []
    stretched_stds= []

    for feat_n in feature_names:
        cleaned_names= feat_n.replace("cat__", "").replace("num__", "")

        assigned= False
        
        for og_names, distribution_params in weight_dict.items():
            if og_names in cleaned_names:
                stretched_means.append(distribution_params[0])
                stretched_stds.append(distribution_params[1])
                assigned= True

                break

        if not assigned:
            print(f"ATTENZIONE: No params trovato per {feat_n}. Assegnato 0.0, std 0.05")
            stretched_means.append(0.0)
            stretched_stds.append(0.05)
            
    return np.array(stretched_means), np.array(stretched_stds)




def get_counterfactuals_DICE(df_log, x, wrapper, cats, target):
    """
    Genera spiegazioni controfattuali per un record  `x` utilizzando DICE.
    """
    query_instance = pd.DataFrame([x])
    start_time = time.time()

    df_log[target] = [bool(y) for y in df_log[target]]
    feature_names = [f for f in df_log.columns if f != target]


    if cats != []:
        dice_data = dice_ml.Data(
            dataframe=df_log,
            categorical_features=cats,
            continuous_features=[f for f in df_log.columns if f not in cats and f != target],
            outcome_name=target
        )
        dice_model = dice_ml.Model(model=wrapper, backend="sklearn")
        dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")
    else:
        dice_data = dice_ml.Data(
            dataframe=df_log,
            continuous_features=[f for f in df_log.columns if f != target],
            outcome_name=target
        )
        dice_model = dice_ml.Model(model=wrapper, backend="sklearn")
        dice_exp = dice_ml.Dice(dice_data, dice_model, method="kdtree")

    # Tentativi progressivi aumentando le feature da variare
    for i in range(1, len(feature_names)+1):
        features_to_vary = feature_names[:i]
        #print(f"Tentativo con le prime {i} feature: {features_to_vary}")
        try:
            counterfactuals = dice_exp.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class="opposite",
                verbose=True,
                features_to_vary=features_to_vary
            )
            print('Controfattuali trovati')
            elapsed_time = time.time() - start_time

            cfs = counterfactuals.cf_examples_list[0].final_cfs_df.drop([target], axis=1).to_dict(orient="records")
            cf_x = counterfactuals.cf_examples_list[0].final_cfs_df.drop([target], axis=1).values[0]

            sparsity = np.sum(np.array(list(x.values())) != cf_x)
            return cfs[0], elapsed_time, sparsity

        except Exception as e:
            #print(f"Tentativo fallito con {i} feature: {e}")
            continue

    # Fallback: se nessun controfattuale è stato trovato, usa l'esempio più simile con classe opposta
    #print("Controfattuali non trovati. Passo al fallback con l'esempio più simile.")
    X = df_log.loc[:, df_log.columns != target]
    X = list(X.to_dict(orient='index').values())

    min_d = float('inf')
    x_ = None
    for x_h in X:
        if wrapper.predict_one(x) != wrapper.predict_one(x_h):
            d = cdist(query_instance, pd.DataFrame([x_h]), metric='euclidean').flatten()
            if d[0] < min_d:
                min_d = d[0]
                x_ = x_h

    elapsed_time = time.time() - start_time
    sparsity = np.sum(np.array(list(x.values())) != np.array(list(x_.values())))
    return x_, elapsed_time, sparsity





def add_iteration(df, iteration, new_instance, new_cf, cf_method, model):
    updated_rows = []

    if cf_method == "DICE":
        print('Aggiornamento')

        # Seleziona righe precedenti all'attuale iterazione
        previous_rows = df[df["iteration"] != iteration]

        # Prendi solo l'ultima occorrenza per ogni controfattuale
        last_rows = previous_rows.drop_duplicates(subset=["counterfactual"], keep="last")

        for _, row in last_rows.iterrows():
            updated_validity = model.predict_one(row["counterfactual"])
            if row["validity"] != updated_validity:
                updated_rows.append({
                    "iteration": iteration,
                    "original_instance": row["original_instance"],
                    "counterfactual": row["counterfactual"],
                    "validity": updated_validity,
                    "cf_method": row["cf_method"]
                })

    print('Fine Aggiornamento')

    # Nuova riga per l'istanza corrente
    new_row = {
        "iteration": iteration,
        "original_instance": new_instance,
        "counterfactual": new_cf,
        "validity": model.predict_one(new_cf),
        "cf_method": cf_method
    }

    # Ritorna DataFrame aggiornato
    return pd.concat([df, pd.DataFrame(updated_rows + [new_row])], ignore_index=True)
        
def get_counterexamples_proto(x, df_log, model, target, cats):
    from classes import RiverModelWrapper
    # Disabilita eager execution all'inizio
    tf.compat.v1.disable_eager_execution()
    
    # Ottimizzazione: usa categorie solo se necessarie
    category_values = {
        df_log.columns.get_loc(col): df_log[col].nunique()
        for col in cats
    } if cats else {}
    
    # Ottimizzazione: seleziona solo le colonne necessarie
    cols_needed = [c for c in df_log.columns if c != target]
    X = df_log[cols_needed]
    
    # Ottimizzazione: usa array numpy invece di DataFrame dove possibile
    X_v = X.to_numpy(dtype=np.float32) 
    query_values = pd.DataFrame([x]).values.astype(np.float32)
    
    # Inizializza il modello wrapper
    feature_names = cols_needed
    river_model_wrapper = RiverModelWrapper(model, target, feature_names)
    
    # Calcola i range delle feature in modo efficiente
    X_min = X_v.min(axis=0)
    X_max = X_v.max(axis=0)
    feature_ranges = (X_min.reshape(1, -1), X_max.reshape(1, -1))
    
    # Funzione di predizione
    predict_fn = lambda x: river_model_wrapper.predict_proba(x)
    
    # Parametri di configurazione
    configs = [
        {'theta': 10.0, 'beta': 0.01, 'lr': 0.01, 'threshold': 0.0, 'c_init': 1.0},
        {'theta': 5.0, 'beta': 0.005, 'lr': 0.01, 'threshold': 0.01, 'c_init': 1.0},
        {'theta': 2.5, 'beta': 0.0025, 'lr': 0.05, 'threshold': 0.005, 'c_init': 1.0},
        {'theta': 1.25, 'beta': 0.00125, 'lr': 0.25, 'threshold': 0.0025, 'c_init': 1.0},
        {'theta': 0.625, 'beta': 0.000625, 'lr': 1.25, 'threshold': 0.001, 'c_init': 100.0}
    ]
    
    start_time = time.time()
    
    for attempt, config in enumerate(configs):
        print(f"Tentativo {attempt+1} con parametri: {config}")
        
        try:
            cf = CounterFactualProto(
                predict_fn,
                query_values.shape,
                beta=config['beta'],
                use_kdtree=True,
                cat_vars=category_values,
                ohe=False,
                theta=config['theta'],
                learning_rate_init=config['lr'],
                max_iterations=800,
                c_init=config['c_init'],
                c_steps=10,
                feature_range=feature_ranges
            )
            
            cf.fit(X_v)
            explanation = cf.explain(query_values, threshold=config['threshold'])
            
            if explanation.cf['X'] is not None:
                elapsed_time = time.time() - start_time
                cf_expl = explanation.cf['X'].flatten()
                def_sparsity = np.sum(query_values != explanation.cf['X'])
                
                # Pulizia memoria prima di restituire i risultati
                del cf, explanation
                
                return dict(zip(feature_names, cf_expl)), elapsed_time, def_sparsity
                
        except Exception as e:
            print(f"Errore al tentativo {attempt+1}: {str(e)}")
            continue
    
    # Se nessun controfattuale trovato, cerca il punto più vicino
    elapsed_time = time.time() - start_time
    min_d = float('inf')
    x_ = None
    
    # Ottimizzazione: usa array numpy per calcolare le distanze
    x_array = np.array([x[col] for col in cols_needed])
    
    for x_h in X.to_dict('records'):
        if model.predict_one(x) != model.predict_one(x_h):
            x_h_array = np.array([x_h[col] for col in cols_needed])
            d = np.linalg.norm(x_array - x_h_array)
            if d < min_d:
                min_d = d
                x_ = x_h
    
    if x_:
        sparsity = sum(x[col] != x_[col] for col in cols_needed)
        elapsed_time = time.time() - start_time
        return x_, elapsed_time, sparsity
    
    return None, elapsed_time, 0 

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj

def run_cf_fit(x, model, target, feature_names, return_dict):
    from classes import RiverModelWrapper
    river_model_wrapper = RiverModelWrapper(model, target, feature_names)
    query_instance = pd.DataFrame([x])
    CF = cf.CounterfactualExplanation(query_instance.values, river_model_wrapper.predict, method='GS')
    
    CF.fit(n_in_layer=100, first_radius=1.5, dicrease_radius=1.5, sparse=False, verbose=False)
    cf_x = CF.enemy
    return_dict['cf'] = cf_x

def get_counterfactuals_GS(x, df_log, model, target, timeout_seconds=300):
    query_instance = pd.DataFrame([x])
    feature_names = list(query_instance.columns)
    manager = mp.Manager()
    return_dict = manager.dict()
    
    process = mp.Process(target=run_cf_fit, args=(x, model, target, feature_names, return_dict))
    start_time = time.time()
    process.start()
    process.join(timeout_seconds)

    elapsed_time = time.time() - start_time

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"Timeout! Il calcolo ha superato {timeout_seconds} secondi, fermato.")
        # Fallback manuale
        X = df_log.loc[:, df_log.columns != target]
        X = list(X.to_dict(orient='index').values())
        min_d = float('inf')
        x_ = None
        for x_h in X:
            if model.predict_one(x) != model.predict_one(x_h):
                d = cdist(query_instance, pd.DataFrame([x_h]), metric='euclidean').flatten()
                if d[0] < min_d:
                    min_d = d[0]
                    x_ = x_h
        if x_:
            sparsity = np.sum(x != x_)
            return x_, elapsed_time, sparsity
        return None, elapsed_time, None

    if 'cf' in return_dict:
        cf_x = return_dict['cf']
        sparsity = np.sum(x != cf_x)
        x_d = dict(zip(feature_names, cf_x))
        return x_d, elapsed_time, sparsity
    else:
        return None, elapsed_time, None







def euclid_distance_fea(current_record, history_matrix):
        
        if history_matrix.size== 0:
            return 0.0
        
        x_current_2d = np.array(current_record).reshape(1, -1)
    
        # cdist(A, B, metric='euclidean') calcola la distanza tra tutte le righe in A e B, libreria cdist
        distances = cdist(x_current_2d, history_matrix, metric='euclidean')
        
        return distances.flatten()
    






def fading_emp_accuracy(record, history_matrix, ground_truth_p, past_prediction, current_prediction, current_index): # controlla che tipo di parametro passare
        """
        fading empirical accuracy formulation
        
        :param record: record corrente
        """
        if history_matrix.size == 0:
            return 0.0
        
        cardinality_hist_m = history_matrix.shape[0] # il termine al denominatore

        #nella formula della FEA la sommatoria ha una condizione: bisogna prendere solo i seen records appartenenti alla
        # hist matrix che hanno la stessa classe della predizione corrente f(x) = y_mach_pred
        predict_mask= (past_prediction == current_prediction)
        
        if not np.any(predict_mask): # se la maschera è vuota usciamo perchè non si può calcolare nulla
            return 0.0

        #applichiamo la maschera alle strutture ottenute

        hist_filt= history_matrix[predict_mask]
        past_p_filt= past_prediction[predict_mask]
        g_truth_filt= ground_truth_p[predict_mask]

        indexes= np.arange(cardinality_hist_m)
        filt_index= indexes[predict_mask]

        distance= euclid_distance_fea(record, hist_filt)
        weight = np.power(0.95, (current_index- filt_index))
        similarity= 1/(1+distance)
            
        kronecker_d= (past_p_filt == g_truth_filt).astype(int)

        summation_term = np.sum(kronecker_d * similarity* weight)
        weights_sum= np.sum(weight * similarity)

        fea_val=  summation_term/weights_sum if weights_sum >0 else 0.0
        print(f"Sum: {summation_term}, Card: {cardinality_hist_m}")
        
        return fea_val




#### XAI: FUNZIONI RIVISITATE PER FUNZIONARE CON modello TORCH

def get_cfe_DICE_MiC(log, curr_rec, torch_model, cats, target): 
    ## prima parte della f rimane praticamente invariata alla sua analoga usata in HiC dalla precedente implementazione
     
    query_instance =pd.DataFrame([curr_rec]) 
    start_time= time.time()

    features_names = [f for f in log.columns if f != target]

# 2. Calling DICE on filtered data + input torch model, method= gradiente
    dice_data= dice_ml.Data(
        dataframe= log,
        categorical_features= cats,
        continuous_features= [c for c in features_names if c not in cats],
        outcome_name =target
    )
    
    dice_model= dice_ml.Model(model=torch_model, backend='PyTorch')
    dice_exp= dice_ml.Dice(dice_data, dice_model, method='gradient')

#3. Generazione CFEs

    # questa porzione è lasciato invariato dall'implementazione precedente

    # Tentativi progressivi aumentando le feature da variare
    for i in range(1, len(features_names)+1):  
        features_to_vary = features_names[:i]
        #print(f"Tentativo con le prime {i} feature: {features_to_vary}")
        
        try:
            counterfactuals = dice_exp.generate_counterfactuals(  
                query_instance,
                total_CFs=1,
                desired_class="opposite",
                verbose=True,
                features_to_vary=features_to_vary
            )
            cf = counterfactuals.cf_examples_list[0]

            if cf.final_cfs_df is not None and not cf.final_cfs_df.empty:
                print('Controfattuali trovati')
                elapsed_time = time.time() - start_time

                cfs = cf.final_cfs_df.drop([target], axis=1).to_dict(orient="records").values[0] # così prendi il primo cf

                cf_vals = np.array([cfs[f] for f in curr_rec.keys()])
                curr_vals = np.array(list(curr_rec.values()))

                sparsity = np.sum(curr_vals != cf_vals)

                return cfs[0], elapsed_time, sparsity

        except Exception as e:
            print(f"Tentativo fallito con {i} feature: {e}")
            continue
         
 
#4. Fallback: se nessun controfattuale è stato trovato, usa l'esempio più simile con classe opposta
    print("Controfattuali non trovati. Passo al fallback con l'esempio più simile.")

    # nella funzione originale si chiamava nuovamente predict_one sia sull'istanza in esame x, che sulla x_h (che sarebbe la sua più vicina con classe opposta)
    # poi veniva applicato un filtro per prendere solo i record di class opposta
    #qui chiaramente siccome usiamo torch, dobbiamo richiamare tutti i passaggi necessari in MiC per dare la predizione

    # trasformiamo il record corrente di cui vogliamo la explanation in tensore
    rec_t = rec_t = torch.tensor(query_instance.values.astype(np.float32))
    
    # 4a. torch model produce la label per il current record
    with torch.no_grad():
        curr_pred= torch.argmax(torch_model(rec_t)).item()  


    # 4b. Stabiliamo la classe opposta 
    target_class= 1- curr_pred # maschera che usiamo dopo per filtrare in modo calcolare le distanze tra il record corrente e il dataframe di predizioni di classe opposta
   
    fallback_data= (log.drop(columns= target, axis= 1)) # this is basically the preprocessing we did when predicting the labels in MiC; drop the target col
    data_t= torch.tensor(fallback_data.values.astype(np.float32)) # transforming the batch in tensor 


    # 4c. Produce prediction per il restante batch log
    with torch.no_grad():  # produciamo tutte le predizioni di nuovo
        hist_pred= torch.argmax(torch_model(data_t), dim= 1).numpy()

    # 4d. Filtriamo il batch usando la mask per avere solo le righe di classe opposta
    mask= (hist_pred == target_class)
    data_opposite_classes= fallback_data[mask] 

#5. Calcolo distanze + sparsities
    if not data_opposite_classes.empty:   
        distances= cdist(query_instance, pd.DataFrame(data_opposite_classes), metric= 'euclidean').flatten() # calcolo distanza euclid tra il record corrente e il DF storico con la classe opposta
        min_d= np.argmin(distances) # assegnazione della distanza minima

        nearest_opp_rec= data_opposite_classes.iloc[min_d].to_dict() # lookup del record più 'vicino' di classe opposta rispetto a quello corrente 
        nearest_cf= np.array([nearest_opp_rec[f] for f in curr_rec.keys()])
        curr_vals= np.array(list(curr_rec.values()))

        sparsity = np.sum(curr_vals != nearest_cf)
        elapsed_time = time.time() - start_time

        return nearest_opp_rec, elapsed_time, sparsity

    return None, elapsed_time, 0


def nuova_fea_2(tried, got, last_index,  current_index,lambda_fading=0.999):
    if tried == 0: 
        return 0.0
    
    # 1. Calcoliamo l'accuratezza empirica base (0.0 - 1.0)
    base_accuracy = got / tried
    
    # 2. Calcoliamo la distanza temporale
    # Se 'tried' rappresenta l'indice dell'ultimo tentativo:
    delta_t = current_index - last_index if current_index > tried else 0
    
    # 3. Trasformiamo la distanza in un peso 0-1
    # Se delta_t è 0 (record appena visto), weight è 1.0
    # Man mano che delta_t cresce, weight tende a 0.0
    temporal_weight = np.power(lambda_fading, delta_t)
    
    # 4. Risultato finale (sempre tra 0 e 1)
    fea_val = base_accuracy * temporal_weight

    return fea_val


def get_counterfactuals_DICE(df_log, x, wrapper, cats, target): #lasciata quasi invariata, si passa il wrapper direttamente

    """
    Genera spiegazioni controfattuali per un record  `x` utilizzando DICE.
    """
    query_instance = pd.DataFrame([x])
    start_time = time.time()

    df_log[target] = [bool(y) for y in df_log[target]]
    feature_names = [f for f in df_log.columns if f != target]


    if cats != []:
        dice_data = dice_ml.Data(
            dataframe=df_log,
            categorical_features=cats,
            continuous_features=[f for f in df_log.columns if f not in cats and f != target],
            outcome_name=target
        )
        dice_model = dice_ml.Model(model=wrapper, backend="sklearn")
        dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")
    else:
        dice_data = dice_ml.Data(
            dataframe=df_log,
            continuous_features=[f for f in df_log.columns if f != target],
            outcome_name=target
        )
        dice_model = dice_ml.Model(model=wrapper, backend="sklearn")
        dice_exp = dice_ml.Dice(dice_data, dice_model, method="kdtree")

    # Tentativi progressivi aumentando le feature da variare
    for i in range(1, len(feature_names)+1):
        features_to_vary = feature_names[:i]
        #print(f"Tentativo con le prime {i} feature: {features_to_vary}")
        try:
            counterfactuals = dice_exp.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class="opposite",
                verbose=True,
                features_to_vary=features_to_vary
            )
            print('Controfattuali trovati')
            elapsed_time = time.time() - start_time

            cfs = counterfactuals.cf_examples_list[0].final_cfs_df.drop([target], axis=1).to_dict(orient="records")
            cf_x = counterfactuals.cf_examples_list[0].final_cfs_df.drop([target], axis=1).values[0]

            sparsity = np.sum(np.array(list(x.values())) != cf_x)
            return cfs[0], elapsed_time, sparsity

        except Exception as e:
            #print(f"Tentativo fallito con {i} feature: {e}")
            continue

    # Fallback: se nessun controfattuale è stato trovato, usa l'esempio più simile con classe opposta
    #print("Controfattuali non trovati. Passo al fallback con l'esempio più simile.")
    X = df_log.loc[:, df_log.columns != target]
    X = list(X.to_dict(orient='index').values())

    min_d = float('inf')
    x_ = None
    for x_h in X:
        if wrapper.predict_one(x) != wrapper.predict_one(x_h):
            d = cdist(query_instance, pd.DataFrame([x_h]), metric='euclidean').flatten()
            if d[0] < min_d:
                min_d = d[0]
                x_ = x_h

    elapsed_time = time.time() - start_time
    sparsity = np.sum(np.array(list(x.values())) != np.array(list(x_.values())))
    return x_, elapsed_time, sparsity


def generate_similars_lore(instance, log, target, wrapper):
    feat_names =log.columns.tolist()
    feat_names = [f for f in feat_names if f != target]

    bbox= sklearn_classifier_wrapper(wrapper)
    explainer = LoreTabularExplainer(bbox)
    config = {'neigh_type':'rndgen', 'size':50, 'ocr':0.1, 'ngen':5}
    explainer.fit(log, target, config)

    query_instance = pd.DataFrame([instance])
    query = np.array(query_instance).flatten()
    
    exp = explainer.explain(query)
    rules = exp.getRules()
    similars = exp.getExemplars()

    df_similars = pd.DataFrame(similars, columns=feat_names)
    
    

    return df_similars, rules





def filter_by_rules(instance, log, rules, wrapper, opposite_label= False):

    # così come la funzione originale, filtra il log rispetto le regole generate da LORE
    mask= pd.Series(True, index= log.index)

    for cond in rules['premise']:
        att = cond['att']
        op = cond['op']
        thr = cond['thr']
        
        if op == '>':
            mask &= (log[att] > thr)
        elif op == '<=':
            mask &= (log[att] <= thr)
    
    filtered_df = log[mask].copy() # log originale di instances compliant con le regole di LORE

    if filtered_df.empty:
        print("There are no instances following the same rule")
        return filtered_df
    
    # ora si filtra il log compliant per ottenere solamente istanze con la stessa classe target
    pred_instance = wrapper.predict(instance)[0]

    df_pred= wrapper.predict(filtered_df)

    if not opposite_label:
        mask_pred = (df_pred == pred_instance)
    else:
        mask_pred = (df_pred != pred_instance)
    
    return filtered_df[mask_pred]





def prepr_log_for_xai(log, attr_list, target_name, include_target= True):  

    features_only= [c for c in attr_list if c != target_name]
    
    if include_target:
        clean_log= log[features_only + [target_name]].copy()
        clean_log[target_name] = clean_log[target_name].astype(int) # cancellabile
    else:
        clean_log= log[features_only].copy()

    return clean_log

# se passo un DF come log, rimane un data frame




def calculate_sparsity(x_dict, cf_x_dict):
    key = list(x_dict.keys())
    x = np.array([x_dict[k] for k in key])
    cf_x = np.array([cf_x_dict[k] for k in key])
    sparsity = np.sum(x != cf_x)
    return sparsity




def calculate_distances(x_dict, examples, feature_ranges=None):
    """
    Calcola la distanza euclidea tra x e una lista di dizionari o un DataFrame.


    !! Modifica: in sostanza se passo come input dei DF bypasso totalmente il nested loop; lascio la f così perchè già funziona
    
    """
    # Estrai feature numeriche da x_dict
    numeric_features = [
        feat for feat, val in x_dict.items() 
        if isinstance(val, (int, float, np.number))
    ]
    if not numeric_features:
        raise ValueError("Nessuna feature numerica trovata in x_dict.")

    # Prepara array di x (solo feature numeriche)
    
    x_values = np.array([x_dict[feat] for feat in numeric_features], dtype=np.float64).reshape(1, -1)
    if len(examples)>0:
        # Gestione input: DataFrame o lista di dizionari
        
        if isinstance(examples, pd.DataFrame):
        # DataFrame -> estrai solo colonne numeriche corrispondenti a x_dict
            examples_numeric = examples[numeric_features].astype(np.float64)
            valid_examples = examples.to_dict('records')
            examples_array = examples_numeric.values

        else:
        # Lista di dizionari -> filtra valori numerici
            valid_examples = []
            examples_values = []
        
            for example in examples:
                ex_values = []
                valid = True
                for feat in numeric_features:
                    val = example.get(feat)
                    if isinstance(val, (int, float, np.number)):
                        ex_values.append(val)
                    else:
                        try:
                             ex_values.append(float(val))
                        except (ValueError, TypeError):
                            valid = False
                            break
            
                if valid:
                    examples_values.append(ex_values)
                    valid_examples.append(example)
        
            
            examples_array = np.array(examples_values, dtype=np.float64)

    # Normalizzazione 
        if feature_ranges:
            ranges = np.array([feature_ranges.get(feat, 1.0) for feat in numeric_features])
            x_values = x_values / ranges
            examples_array = examples_array / ranges

    # Calcola distanze con cdist
       
        distances = cdist(x_values, examples_array, 'euclidean')[0]
       

    # Combina risultati
        
        results = list(zip(valid_examples, distances))
    
        results = [item for item in results if item[1]!=0.0] 
        results.sort(key=lambda item: item[1])
       
    else:
       
        results = []
    return results



def evaluate_threshold(tau, max_conf, y_gt, y_preds):
   mask = max_conf >= tau
   if mask.sum() == 0:
      return 0.0, 0.0, 1.0 # acc=0, coverage=0, defer_rate=1
   acc_sel = (y_preds[mask] == np.array(y_gt)[mask]).astype(float).mean()
   coverage = mask.mean()
   defer_rate = 1.0 - coverage
   return acc_sel, coverage, defer_rate



   ## Nuova FEA:
def nuova_fea(tried, got, current_index):
    if tried == 0: 
        return 0.0
    if got == 0:
        return 0.0
    weight= 0.99 * (current_index - tried if tried > 1 else 0)
    num= got*weight
    fea_val= num / tried

    return fea_val


def stretch_dict_params(feature_names, weight_dict):

    stretched_means= []
    stretched_stds= []

    for feat_n in feature_names:
        cleaned_names= feat_n.replace("cat__", "").replace("num__", "")

        assigned= False
        
        for og_names, distribution_params in weight_dict.items():
            if og_names in cleaned_names:
                stretched_means.append(distribution_params[0])
                stretched_stds.append(distribution_params[1])
                assigned= True

                break

        if not assigned:
            print(f"ATTENZIONE: No params trovato per {feat_n}. Assegnato 0.0, std 0.05")
            stretched_means.append(0.0)
            stretched_stds.append(0.05)
            
    return np.array(stretched_means), np.array(stretched_stds)














"""
                                    #_, df_xai= get_percentage_and_df(None, self.processed, self.target) # prendiamo una fotografia utile esclusivamente per xai

                                    xai_log = pd.DataFrame([self.processed[record]['dict_form']])
                                    xai_log = xai_log[[c for c in self.attr_list if c in xai_log.columns]]

                                    # Aggiungi la label a mano se non c'è nel dizionario ma ti serve
                                    xai_log[self.target] = y
                                    
                                    #xai_log= prepr_log_for_xai(self.processed[record]['dict_form'], self.attr_list, self.target, self.processed[record]['ground_truth'])
                                    
                                    hic_wrapper= RiverModelWrapper(self.hic_model, self.target, self.feature_names)
                                    _, rules_dict, _, counter_rules_dict = generate_exp_lore(x, 
                                                                            xai_log, 
                                                                            self.target, 
                                                                            hic_wrapper,
                                    )
                                    
                                    
                                    ## ESEMPI LORE
                                    # EXPLAINATIONS FORNITE DA LORE

                                    
                                
                                
                                    #start_time = time.time()
                                    
                                    _, rules_dict, _, counter_rules_dict = generate_exp_lore(x, 
                                                                            xai_log, 
                                                                            self.target, 
                                                                            hic_wrapper,
                                    )

                                    
                                    ex_pro_df= filter_by_rules(x, xai_log, rules_dict, hic_wrapper)

                                    ex_against_df= filter_by_rules(x, xai_log, counter_rules_dict, hic_wrapper, True)

                                    n_examples= len(ex_pro_df) + len(ex_against_df)

                                    
                                   

                                    
                                    time_lore_mic= time.time()-start_time
                                
                                    # Calcolo metriche:
                                    
                                    #1. Proximity: distanza tra x e df similars

                                    distance_x_r_lore= calculate_distances(x, df_similars) # distance ritorna il formato lista = [(valido esempio:distanza)]
                                    proximity_lore= distance_x_r_lore[0][1]
                                    example_lore= distance_x_r_lore[0][0] # esempio più vicino all'istanza x
                                    
                                    # 2. Plausibility: distanza tra filtered_df e l'esempio più "vicino" --> 
                                    # quindi dobbiamo calcolare la dist tra x e il DF similars, prendere quello con dist minore

                                    # per la plausability ci serve il similar più vicino
                                    if not filtered_df.empty:
                                        distance_sim_log_lore= calculate_distances(example_lore, filtered_df)
                                        plausibility_lore= distance_sim_log_lore[0][1]
                                    else:
                                        print("Warning: No historical matches for the given rule.")
                                        plausibility_lore = np.nan

                                    #3. Sparsity
                                    sparsity_lore= calculate_sparsity(x, example_lore)
                                
                                    proximities_LORE.append({str(record):proximity_lore})
                                    times_LORE.append(time_lore_mic)
                                    sparsities_LORE.append({str(record):sparsity_lore})
                                    plausabilities_LORE.append({str(record): plausibility_lore})
                                    """