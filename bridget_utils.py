import numpy as np
import torch
from scipy.spatial.distance import cdist
import dice_ml
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
import time
import pandas as pd
from xailib.explainers.lore_explainer import LoreTabularExplainer
from sklearn.metrics import accuracy_score,f1_score
import fatf.fairness.data.measures as fatf_dfm
from sklearn.metrics import confusion_matrix
from growingspheres import counterfactuals as cf
import os
import pickle
import orjson
import random

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from ignite.metrics import Accuracy, Loss
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine


###########
# -- FUNZIONI NECESSARIE PER LA PREDIZIONE DEGLI ESPERTI (prese da Open L2D)
def sig(x):
    return 1/(1+np.exp(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

def invert_labels_with_probabilities(labels_arr, p_arr, seed):
    rng = np.random.default_rng(seed=seed)
    
    mask = rng.binomial(n=1, p=p_arr, size=np.atleast_1d(labels_arr).shape[0]).astype(bool)

    new_labels = np.atleast_1d(labels_arr).copy()
    
    new_labels[mask] = np.abs(new_labels[mask] - 1)
    

    return new_labels.astype(int)



###########
# -- FUNZIONI DI SUPPORTO HIC PHASE 

def get_index (attribute, attr_list):
    for j in range(len(attr_list)):
        if attr_list[j] == attribute:
            return j

def percentage(value, all_records):
    return round((value * all_records) / 100)

def ideal_record_test(rec, rule_att, rule_value):
    if rec[rule_att] > rule_value:
        return True
    else:
        return None


def get_value_swap_records(x, processed, protected, attr_list):
    if not processed:
        return [], None
    
    
    protected_inx = []
    for att in protected:
        protected_inx.append(get_index(att, attr_list))
                
    current = list(x.values())
    vs_records = []
    check = True
    vs_decision = None
    
    for record in processed:
        check = True
        for i in range(len(record)):
            if i not in protected_inx:
                if record[i] != current[i]:
                    check = False
            else:
                if record[i] == current[i]:
                    check = False
        if check:
            vs_records.append(record)
            vs_decision = processed[record]['decision']
    
    return vs_records, vs_decision


def get_fairness(model, protected, processed, protected_values):
    PP, PN, DP, DN = [], [], [], []
    PP_c, PN_c, DP_c, DN_c = 0, 0, 0, 0

    for rec in processed:
        og_rec = processed[rec]['dict_form']
        proba = model.predict_proba_one(og_rec)[True]
        if processed[rec]['decision'] == True:
            if processed[rec]['dict_form'][protected[0]] == protected_values[0]:
                PP_c = PP_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    PP.append(((proba, rec)))
            else:
                DP_c = DP_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    DP.append(((proba, rec)))
        else:
            if processed[rec]['dict_form'][protected[0]] == protected_values[0]:
                PN_c = PN_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    PN.append(((proba, rec)))
            else:
                DN_c = DN_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    DN.append(((proba, rec)))
                  
    try:
        fairness = (PP_c) / ((PP_c)+(PN_c)) - (DP_c) / ((DP_c)+(DN_c))
    except:
        fairness = 0
    
    if fairness != 0:
        fair_number = round(((DP_c)+(DN_c)) * ((PP_c)+(DP_c)) / ((PP_c)+(PN_c)+(DP_c)+(DN_c)))
        
    if fairness < 0:
        DN = PN
        PP = DP

    DN = [e for e in DN if e[0] > 0.5]
    PP = [e for e in PP if e[0] < 0.5]
    
    DN = sorted(DN, reverse=True)
    PP = sorted(PP)
    
    return DN, PP, fairness


def evaluation_human (processed, protected, Y, attr_list):
    DN, DP, PN, PP = 0, 0, 0, 0
    Y_final = []

    for r in processed:
        record = processed[r]['dict_form']
        sa = record[protected[0]]
        decision = processed[r]['decision']
        
        Y_final.append(decision) #for accuracy

        if decision == 0:
            if sa == 0:
                PN = PN + 1
            else:
                DN = DN + 1
        else:
            if sa == 0:
                PP = PP + 1
            else:
                DP = DP + 1

    try:
        human_fairness = (PP) / ((PP)+(PN)) - (DP) / ((DP)+(DN))
    except:
        human_fairness = 0
        
    human_acc = accuracy_score(Y_final, Y[:len(Y_final)])
    
    processed_df = pd.DataFrame.from_dict(list(processed.keys()))
    processed_df.columns = attr_list[:-1]
    data_fairness_matrix = fatf_dfm.systemic_bias(np.array(list(processed_df.to_records(index=False))), np.array(Y_final), protected)
    is_data_unfair = fatf_dfm.systemic_bias_check(data_fairness_matrix)
    unfair_pairs_tuple = np.where(data_fairness_matrix)
    unfair_pairs = []
    for i, j in zip(*unfair_pairs_tuple):
        pair_a, pair_b = (i, j), (j, i)
        if pair_a not in unfair_pairs and pair_b not in unfair_pairs:
            unfair_pairs.append(pair_a)
    if is_data_unfair:
        unfair_n = len(unfair_pairs)
    else:
        unfair_n = 0
        
    return human_fairness, human_acc, unfair_n

def evaluation_frank (X_test, Y_test, model, protected, preprocessor):
    frank_preds = []

    PP, DP, PN, DN = 0, 0, 0, 0

    for x_t, y_t in zip(X_test, Y_test):
        if preprocessor is not None:
            x_t = preprocessor.transform_one(x_t)
        test_pred = model.predict_one(x_t)
        #print('test pred : ',test_pred)
        frank_preds.append(test_pred)
        

        if test_pred == True:
            if x_t[protected[0]] == 0: #0 Male, 1 Female in our tests
                PP = PP + 1
            else:
                DP = DP + 1
        else:
            if x_t[protected[0]] == 0:
                PN = PN + 1
            else:
                DN = DN + 1

    try:
        frank_fairness = (PP) / ((PP)+(PN)) - (DP) / ((DP)+(DN))
    except:
        frank_fairness = 0
    #print('frank_preds,Y_test:',frank_preds, Y_test)
    frank_acc = accuracy_score(frank_preds, Y_test)
    frank_f1 = f1_score(Y_test, frank_preds, average='macro')
    #print('accuracy',frank_acc)
    
    frank_cm = confusion_matrix(Y_test, frank_preds)
    #print('Confusion_matrix : ',confusion_matrix(Y_test, frank_preds))
    
    return frank_fairness, frank_acc,frank_f1,frank_cm


def calculate_metrics(X_test, Y_test, model):
    frank_preds = [model.predict_one(x_t) for x_t in X_test]
    #print(frank_preds)
    frank_acc = accuracy_score(Y_test, frank_preds)
    #print(Y_test)
    frank_f1 = f1_score(Y_test, frank_preds, average='macro')
   
    from sklearn.metrics import confusion_matrix
    frank_cm = confusion_matrix(Y_test, frank_preds)
    

    return frank_acc,frank_f1,frank_cm

def convert_dict_list_to_float32(dict_list):
    return [
        {
            k: np.float32(v) if isinstance(v, float) else v
            for k, v in d.items()
        }
        for d in dict_list
    ]


def get_percentage_and_df(df_train, processed, target): # modificata leggermente aggiungendo le colonne di provider, model conf ecc

    #  issue: è necessario aggiungere il caso in cui il log sia vuoto (cioè la primissima riga del ciclo start HiC)
    user_truth= 'expert prediction'
    machine_prediction= 'machine prediction'
    g_truths= 'ground truth'
    provider_f= 'provider'
    model_conf= 'proba_model'
    processed_c = processed.copy()
    rows = []

    #query_instance = pd.DataFrame([x])
    df_log  = pd.DataFrame()
    #print('Processed_C',processed_c)

    if len(processed_c)>0:

        for entry in processed_c.values():
                
                row = entry['dict_form'].copy() # features originali 

                # colonne aggiuntive necessarie per HiC e MiC
                row[user_truth]= entry['user']
                row[machine_prediction]= entry['machine']
                row[g_truths]= entry['ground_truth']
                row[model_conf]= entry['proba_model']
                row[target] = entry['decision']
                row[provider_f]= entry['provider_flag']               

                rows.append(row)
                
        df_proc = pd.DataFrame(rows)

        if df_train is None or df_train.empty: # se il log è vuoto
            df_log= df_proc
        
        else:
            df_log = pd.concat([df_train,df_proc], ignore_index=True)

    else:
        df_log= df_train if df_train is not None else pd.DataFrame()
    
    percentage_dict= dict()
    
    if not df_log.empty and target in df_log.columns:
        percentage_dict = (df_log[target].value_counts(normalize=True) * 100).round(2).to_dict()

    return  percentage_dict, df_log

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





###########
# -- DRIFT CHECK and FEA

def exit_HiC(available_budget, current_budget, fea_vals, desired_performance):

    if not fea_vals:
        return False
    
    avg_fea= np.mean(fea_vals)

    if current_budget < available_budget and avg_fea < desired_performance:
        return False
    else: 
        return True


def exit_MiC(fea_vals, user_patience, low_belief_count, desired_performance):
    # same in form, different concept, this time the user patience is an input param given deliberately by the user

    # if we wanted to replicate the effect of the budget in the MiC phase, we could theorize that after a certain no. of low belief instances
    # produced by the machine, the user gets dissatisfied and looks into it every single time there is a new low belief count, to assess whether to switch phase or not
    # the effect on the exit condition is basically like p_max of the first formulation, with a new name this time

    if not fea_vals:
        return False
    
    avg_fea= np.mean(fea_vals)
    
    if low_belief_count > user_patience and avg_fea < desired_performance:
        return True

    return False
    





###########
# -- UTILITIES / PREPROCESSING FUNCTIONS

def clean_compas(data, col_to_delete, col_to_strip= None, drop_duplicates=True):

    if col_to_strip is not None:
        data[col_to_strip] = data[col_to_strip].str.strip("()")
    
    data= data.drop(columns= col_to_delete)

    if drop_duplicates:
        data= data.drop_duplicates()

    return data.reset_index(drop=True)


def stratif(start_point, end_point, class_0, class_1):
    class_0_perc= class_0.iloc[int(len(class_0)*start_point) : int(len(class_0)*end_point)]
    class_1_perc= class_1.iloc[int(len(class_1)*start_point) : int(len(class_1)*end_point)]

    total= pd.concat([class_0_perc, class_1_perc]).sample(frac=1, random_state= 42).reset_index(drop=True)
    #chiaramente se c'è il concat bisogna rifare lo shuffle

    return total

def scale_df(data, pipe, target_c):  # a quanto pare usando questa pipeline di River la label viene persa per strada quindi la devo riattaccarre
    processed_r= []
    labels= data[target_c].values
    X = data.drop(columns=[target_c])

    for i,r in enumerate(X.to_dict(orient='records')):
        scaled_r = pipe.transform_one(r)
        scaled_r[target_c] = labels[i]
        processed_r.append(scaled_r)
    return pd.DataFrame(processed_r)

def apply_map(data, col_mapping):

    for col, mapping in col_mapping.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)
    return data

def apply_order(data):

    order= [c for c in data if c not in ['ground truth', 'proba_model', 'provider', 'machine prediction', 'expert prediction']]
    return data[order]

def x_y_split(df, target):
    X = df.drop(columns=target)
    y = df[target]
    return X, y


def set_all_seeds(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
   
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_data(directory, prefix, data_dict):
    os.makedirs(directory, exist_ok=True)

    for suffix, data in data_dict.items():
        path = os.path.join(directory, f"{prefix}{suffix}")

        if suffix.endswith('.pkl'):
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        
        elif suffix.endswith('.json'):
            with open(path, 'wb') as f:
                f.write(orjson.dumps(convert_numpy(data)))

        elif suffix.endswith('.txt'):
            with open(path, 'w') as f:
                if isinstance(data, (list, tuple, range)):
                    for i, val in enumerate(data):
                        f.write(f"{i} {val}\n")
                else:
                    f.write(f"{data}\n")

def create_loader(df, features, target, batch_size=128, shuffle=False):

    #establish external order, pass target col
    # then pass df acc t or whatever

    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df[target].values, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return X, y, loader







###########
# -- UTILITIES DEFERRAL NET CALIBRATION

# condense the ignite environment in a single f so the notebook is not polluted
# major events to log: 
# 1. create supervised trainer and evaluator for training/validation
# 2. set logging interval as param
# 3. trainer event (iteration completed) to log training loss
# 4. trainer event (epoch completed) to log training and validation metrics
# 5. trainer event (epoch completed) to update lr scheduler
# 6. return score function
# 7. call early stopping (plug params patience, max epochs)
# 8. add the event handler and checkpoint
# 9. call trainer run with train loader and max epochs as params

def net_trainer(net, optimizer, criterion, device, acc_t_loader, val_loader, scheduler, iter, model_prefix, directory_name,
                log_interval=100, patience=25, max_epochs=50):

    # first set the structures

    training_history = {'accuracy':[],'loss':[]}
    validation_history = {'accuracy':[],'loss':[]}
    val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}


    # set trainer and evaluators
    trainer = create_supervised_trainer(net, optimizer, criterion, device)
    train_evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=device)

    # setting events
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(acc_t_loader)
        metrics = train_evaluator.state.metrics
        training_history['accuracy'].append(metrics['accuracy']*100)
        training_history['loss'].append(metrics['loss'])
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        validation_history['accuracy'].append(metrics['accuracy']*100)
        validation_history['loss'].append(metrics['loss'])
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr_scheduler(trainer):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"End of Epoch {trainer.state.epoch}: Learning Rate {lr}")


    # get score function

    def score_function(engine):
        return engine.state.metrics["accuracy"]
    
    # call early stopping and model checkpoint

    handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)

    checkpoint = ModelCheckpoint(
        dirname=directory_name,
        filename_prefix=model_prefix,
        n_saved=1,
        create_dir=True,
        require_empty=False,
        global_step_transform=global_step_from_engine(trainer) # helps fetch the trainer's state
    )

    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint, {'model': net})

    # finally call run

    trainer.run(acc_t_loader, max_epochs= max_epochs)

    return training_history, validation_history








###########
# -- CALIBRAZIONE THRESHOLD STRAT 1 E 2

def evaluate_threshold(tau, max_conf, y_gt, y_preds):
   mask = max_conf >= tau
   if mask.sum() == 0:
      return 0.0, 0.0, 1.0 # acc=0, coverage=0, defer_rate=1
   acc_sel = (y_preds[mask] == np.array(y_gt)[mask]).astype(float).mean()
   coverage = mask.mean()
   defer_rate = 1.0 - coverage
   return acc_sel, coverage, defer_rate





###########
# -- FUNZIONI DI SUPPORTO MIC PHASE 
    


def deferral_loss(r, pred_correct, C):
    """
    :param r: output della rete di deferral 
    :param pred_correct: 1 se la macchina aveva azzeccato in HiC, altrimenti 0
    :param C: costi del deferral basati sui log
    """

    # r: tensor (batch,)
    s0 = torch.zeros_like(r) # score per usare il modello
    s1 = -r # score per deferire all'umano
    stacked = torch.stack([s0, s1], dim=0)

    Z = torch.logsumexp(stacked, dim=0)
    log_p0 = s0 - Z # log P(use model)
    log_p1 = s1 - Z # log P(defer)

    term_model = pred_correct * (-log_p0)
    term_expert = C * (-log_p1)

    return (term_model + term_expert).mean()

def p_defer(x_tensor, net):

    net.eval()

    with torch.no_grad():

        if not torch.is_tensor(x_tensor):
            x_tensor = torch.tensor(x_tensor, dtype=torch.float32)

        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)
            
        r = net(x_tensor)
        s0 = torch.zeros_like(r)
        s1 = -r

        stacked = torch.stack([s0, s1], dim=0)
        Z = torch.logsumexp(stacked, dim=0)
        log_p1 = s1 - Z
        
        return torch.exp(log_p1).cpu().numpy()





###########
# -- FUNZIONI XAI
# -- rivisitate per il wrapper della rete torch
    
def prepr_log_for_xai(memory, log, attr_list, target_name):  

    """
    ## qui l'idea è che: siccome LORE durante i primi esperimenti non riusciva a generare il neighborhood,
    ai fini esclusivi di generare le regole e contro regole così da filtrare il df originale, si usano anche le istanze
    del dataset di avviamento che
    """

    cols= [c for c in attr_list]

    if log is not None:
        rows= []
        for rec_id in log:
            rec_values= log[rec_id]['dict_form'].copy()
            rec_values[target_name]= log[rec_id]['ground_truth']
            rows.append(rec_values)
        
        current_memory = pd.DataFrame(rows)
        current_memory = current_memory[cols]
    else:
        current_memory= pd.DataFrame(columns= cols) # se è vuoto 

    avv_memory =  memory[cols].copy()
    xai_memory= pd.concat([avv_memory, current_memory], ignore_index=True)
    return xai_memory



def filter_df_by_counter_rules(instance, log, counter_rules, wrapper): # aggiunto solo il wrapper come parametro input
   
    mask = pd.Series(False, index=log.index)  # Inizializza tutto False

    if isinstance(counter_rules, list):    
    # Applica OR tra le regole
        for rule in counter_rules:
            rule_mask = pd.Series(True, index=log.index)  # Inizializza tutto True per questa regola
        
        # Applica AND tra le condizioni della singola regola
            for cond in rule['premise']:
                att = cond['att']
                op = cond['op']
                thr = cond['thr']
            
                if op == '>':
                    rule_mask &= (log[att] > thr)
                elif op == '<=':
                    rule_mask &= (log[att] <= thr)
        
            mask |= rule_mask  # OR logico tra regole

    
    # Filtra per predizione opposta
        pred_mask = log.apply(
            lambda row: wrapper.predict_one(row.to_dict()) != wrapper.predict_one(instance),
            axis=1
        )
    
        return log[mask & pred_mask]  # Combina maschere
    
    else:
        check=False
        
        return filter_by_rules(instance,counter_rules,wrapper,check)
    


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


def filter_by_rules(instance, log, rules, wrapper, check= False):

    # così come la funzione originale, filtra il log rispetto le regole generate da LORE
    mask= pd.Series(True, index= log.index)
    
    if isinstance(rules, dict):
        rules = [rules]
    else:
        # Se è già una lista (come counter_rules_dict), la usiamo così com'è
        rules = rules

    for condition in rules['premise']:
        att = condition['att']
        op = condition['op']
        thr = condition['thr']
        
        if op == '>':
            mask &= (log[att] > thr)
        elif op == '<=':
            mask &= (log[att] <= thr)
    
    filtered_df = log[mask].copy() # log originale di instances compliant con le regole di LORE

    if filtered_df.empty:
        print("There are no instances following the same rule")
        return filtered_df
    
    # ora si filtra il log compliant per ottenere solamente istanze con la stessa classe target
    pred_instance = wrapper.predict(instance)

    df_pred= wrapper.predict(filtered_df)

    predictions= np.array(df_pred).flatten()
    target_val = np.array(pred_instance).flatten()[0]

    if check:
        return filtered_df[predictions == target_val]
    else:
        return filtered_df[predictions != target_val]
   
 
def filter_df_by_rules(instance, log, rules, wrapper, check=False):
    
    # 1. Filtra per la regola (AND tra condizioni)
    mask = pd.Series(True, index=log.index)
    for cond in rules['premise']:
        att = cond['att']
        op = cond['op']
        thr = cond['thr']
        
        if op == '>':
            mask &= (log[att] > thr)
        elif op == '<=':
            mask &= (log[att] <= thr)
    
    filtered_by_rule = log[mask].copy()  # DataFrame già filtrato per la regola
    
    # 2. PRE-CALCOLO DELLA PREDIZIONE TARGET (Fuori dal loop per la CPU)
    # Assicuriamoci che instance sia un dizionario per il wrapper
    instance_dict = instance.iloc[0].to_dict() if isinstance(instance, pd.DataFrame) else instance
    target_pred = int(wrapper.predict_one(instance_dict))

    # 3. FILTRAGGIO PREDITTIVO LINEARE (Niente .apply() o funzioni annidate)
    # Trasformiamo il log in una lista di dizionari per velocizzare l'accesso
    log_dicts = filtered_by_rule.to_dict(orient='records')
    
    # Calcoliamo le predizioni per tutte le righe in un colpo solo
    # (Se il wrapper supporta i batch, potresti passare tutto il log insieme qui)
    row_preds = [int(wrapper.predict_one(row)) for row in log_dicts]

    # 4. CREAZIONE DELLA MASCHERA FINALE
    if not check:
        final_mask = [p == target_pred for p in row_preds]
    else:
        final_mask = [p != target_pred for p in row_preds]

    return filtered_by_rule[final_mask]


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
    if x_ is None:
        return None, 0, 0
    elapsed_time = time.time() - start_time
    sparsity = np.sum(np.array(list(x.values())) != np.array(list(x_.values())))

    return x_, elapsed_time, sparsity


def calculate_sparsity(x_dict, cf_x_dict):
    key = list(x_dict.keys())
    x = np.array([x_dict[k] for k in key])
    cf_x = np.array([cf_x_dict[k] for k in key])
    sparsity = np.sum(x != cf_x)
    return sparsity


def generate_exp_lore(instance, log, target, wrapper): 
    # cambiata leggermente da quella originale per includere il wrapper 
    # + avoid calling explainer.explain twice, so for each record evaluated both rules and counter rules are
    # extracted straight away and returned 

    feat_names =log.columns.tolist()
    feat_names = [f for f in feat_names if f != target]


    bbox= sklearn_classifier_wrapper(wrapper)
    explainer = LoreTabularExplainer(bbox)
    config = {'neigh_type':'geneticp', 'size':100, 'ocr':0.1, 'ngen':2}
    explainer.fit(log, target, config)
    

    query_instance = pd.DataFrame([instance])
    query = np.array(query_instance).flatten()
   
    exp = explainer.explain(query)
    
    rules = exp.getRules()
    similars = exp.getExemplars()
    
    rule_dicts = []
    df_similars = pd.DataFrame()
    if similars is not None:
        for array in similars:
            dizionario = dict(zip(feat_names, array))
            rule_dicts.append(dizionario)
        df_similars = pd.DataFrame(rule_dicts)    
    
    
    
    counter_rules= exp.getCounterfactualRules()
    opposites= exp.getCounterExemplars()

    
    c_rules_dicts = []
    df_opp = pd.DataFrame(c_rules_dicts)
    if opposites is not None:
        for array in opposites:
            dizionario = dict(zip(feat_names, array))
            c_rules_dicts.append(dizionario)
        df_opp = pd.DataFrame(c_rules_dicts)  
    
    

    return exp, df_similars, rules, df_opp, counter_rules

def get_cfe_DICE_MiC(log, curr_rec, torch_model, cats, target): 
    ## prima parte della f rimane praticamente invariata alla sua analoga usata in HiC dalla precedente implementazione
    
    query_instance =pd.DataFrame([curr_rec]) 
    start_time= time.time()
    elapsed_time= 0
    features_names = [f for f in log.columns if f != target]

# 2. Calling DICE on filtered data + input torch model, method= gradiente
    dice_data= dice_ml.Data(
        dataframe= log,
        categorical_features= cats,
        continuous_features= [c for c in features_names if c not in cats],
        outcome_name =target
    )
    
    dice_model = dice_ml.Model(model=torch_model, backend='PYT')
    dice_exp= dice_ml.Dice(dice_data, dice_model, method='gradient')

#3. Generazione CFEs

    # questa porzione è lasciata invariato dall'implementazione precedente

    # Tentativi progressivi aumentando le feature da variare in abtch di 5 

    trials= [
    features_names[:5],   
    features_names[:10],  
    features_names        
]

    for features_to_vary in trials:          
        try:
            counterfactuals = dice_exp.generate_counterfactuals(  
                query_instance,
                total_CFs=1,
                desired_class="opposite",
                verbose=False,
                features_to_vary=features_to_vary,
                sample_size=100,      
                proximity_weight=0.5, 
                sparsity_weight=1.0
            )
            
            cf = counterfactuals.cf_examples_list[0]

            if cf.final_cfs_df is not None and not cf.final_cfs_df.empty:
                #print('Controfattuali trovati')
                elapsed_time = time.time() - start_time

                cfs = cf.final_cfs_df.drop([target], axis=1).to_dict(orient="records")[0] # così prendi il primo cf

                cf_vals = np.array([cfs[f] for f in curr_rec.keys()])
                curr_vals = np.array(list(curr_rec.values()))

                sparsity = np.sum(curr_vals != cf_vals)

                return cfs, elapsed_time, sparsity

        except Exception as e:
            #print(f"Tentativo fallito con {i} feature: {e}")
            continue
         
 
#4. Fallback: se nessun controfattuale è stato trovato, usa l'esempio più simile con classe opposta
    #print("Controfattuali non trovati. Passo al fallback con l'esempio più simile.")

    # nella funzione originale si chiamava nuovamente predict_one sia sull'istanza in esame x, che sulla x_h (che sarebbe la sua più vicina con classe opposta)
    # poi veniva applicato un filtro per prendere solo i record di class opposta
    #qui chiaramente siccome usiamo torch, dobbiamo richiamare tutti i passaggi necessari in MiC per dare la predizione

    # trasformiamo il record corrente di cui vogliamo la explanation in tensore
    rec_t = torch.tensor(query_instance.values.astype(np.float32))
    
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
        
        #nearest_cf= np.array([nearest_opp_rec[f] for f in curr_rec.keys()])
        #curr_vals= np.array(list(curr_rec.values()))

        nearest_cf = np.array([nearest_opp_rec[f] for f in curr_rec.keys() if f in nearest_opp_rec])
        curr_vals = np.array([val for f, val in curr_rec.items() if f in nearest_opp_rec])

        sparsity = np.sum(curr_vals != nearest_cf)
        elapsed_time = time.time() - start_time

        return nearest_opp_rec, elapsed_time, sparsity
    
    elapsed_time = time.time() - start_time
    return curr_rec, elapsed_time, 0


def get_neighbors(record, record_g_truth, target, cache, relevance_window= 100, n_neighbors= 1): # la cache senza le colonne provider ecc mi raccomando
    
    # !! remember to pass record as a DICT
    
    # record is a dict, called as x= self.X[i]
    # g_truth likewise
    # target, its self.target (str)
    # cache is a DF
    
    # 100 records should be fine and not too much demanding

    # returns 2 vals
    
    ## per evitare il nested loop la cache deve essere un DF, quindi chiama la funzione prepr xai beforehand

    # cosa dobbiamo fare qua dentro? diverse cose
    # Goal --> extract the nearest neighbor of the same and opposite class, meaning i'll need to know beforehand the ground truth of the record
    # first, to be more efficient: get only the relevant portion of the data, according to the relevancy window
    # Then, in order not to repeat the distance computation, we'll calculate the euclid similarity
    # Only afterwards is the DF filtered with past ground truth == current g truth or =! and then the results are returned

    numeric_features = [feat for feat, val in record.items() 
                        if isinstance(val, (int, float, np.number))]
    # in the dict record, get the col if the vals are either int, float or np.number
    # else raise value error
    
    if not numeric_features:
        raise ValueError("No numerical features found in the record.")
    

    # 1. relevancy filter straight away

    relevance= max(0, len(cache) - relevance_window)  
    # get the relevance dinamically based to the relevance window and the current cache
    # extract only the relevant portion of the cache
    
    valid_cache = cache.iloc[relevance:].copy()

    # if theres no valid cache exit immediately
    if valid_cache is None or len(valid_cache)==0:
        return [],[], [], []

    # 2. computing distances on the valid portion only

    rec_values = np.array([record[feat] for feat in numeric_features], dtype=np.float64).reshape(1, -1) 
    # here we build an array of shape (1, number of numeric features) and cast everything to float
    
        # anche questa logica mantenuta dalla vecchia f
    cache_values = valid_cache[numeric_features].values.astype(np.float64) # casting everything in the cache for the same reason

    distances = cdist(rec_values, cache_values, 'euclidean')[0] # compute the distances

    valid_cache['distances'] = distances  # ok questa era un problema, dopo avere sortato va tolta subito

    # 3. now filter
    same_class = valid_cache[valid_cache[target] == record_g_truth] 
    opp_class= valid_cache[valid_cache[target] != record_g_truth] 

    # here we apply the the filter for the same and opposite class so we're comparing a value with the column and putting everything in a data frame

    # 4. then sort temp cache and get the n_neighbors you want

    k_neighbors_same= same_class.sort_values('distances').head(n_neighbors)
    k_neighbors_opp= opp_class.sort_values('distances').head(n_neighbors)

    # 5. ok now we want to get the sparsity just like the other metrics 
    # note that sparsity is sparsity = np.sum(curr_vals != cf_vals) so get vals for the first row if there are more
    
    curr_rec_vals= np.array(list(record.values()))

    if k_neighbors_same is not None and len(k_neighbors_same) > 0: # ok no this must be changed, forgot this wasnt proper
        k_neighbors_same= k_neighbors_same.drop(columns= 'distances') # drop the distance column after sorting the cache, otherwise it will be included in the sparsity calculation
        nearest_same= k_neighbors_same.iloc[0] # plus this is df use iloc
        same_class_vals= nearest_same[numeric_features].values# ok usando iloc hai una serie quindi non serve transformare in np
        sparsity_same= np.sum(curr_rec_vals != same_class_vals)
        

    if k_neighbors_opp is not None and len(k_neighbors_opp) > 0:
        k_neighbors_opp= k_neighbors_opp.drop(columns= 'distances')
        nearest_opp= k_neighbors_opp.iloc[0] # first row 
        opp_class_vals= nearest_opp[numeric_features].values # get the vals as array
        sparsity_opp= np.sum(curr_rec_vals != opp_class_vals)
        


    return k_neighbors_same, k_neighbors_opp, sparsity_same, sparsity_opp


# recap the function had severall issues: first, forgot .iloc cause i got tangled up over the constant changes
# then, !!!! the xai log function returned the DF with the label because its needed to filter over sam/opp class
# but it must be removed, along with the auxiliary col distance after sorting the cache
# or just use [numerical_features] so the same order is enforced everywhere





def calculate_distances(x_dict, examples, feature_ranges=None): # originale
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



# ok allora nel py di growing spheres c'è la classe che richiede questi input:
# def __init__(self, obs_to_interprete, prediction_fn, method='GS', target_class=None, random_state=None):
# nell'esempio chiamata così: CF = cf.CounterfactualExplanation(obs, clf.predict, method='GS')
# quindi passo il wrapper o il modello stesso .predict


def get_GS_cfe(log, curr_rec, torch_model, cats, target): 
    ## prima parte della f rimane praticamente invariata alla sua analoga usata in HiC dalla precedente implementazione
    
    # x_rec called like this  x = self.df_batch3[self.feature_names].iloc[record].to_dict() not a tensor
    query_instance =pd.DataFrame([curr_rec]) 
    start_time= time.time()
    elapsed_time= 0
    features_names = [f for f in log.columns if f != target]

    rec= np.array(list(curr_rec[f] for f in features_names)).reshape(1, -1)
        
    try:
        
        exp = cf.CounterfactualExplanation(rec, torch_model.predict, method= 'GS')
        exp.fit(n_in_layer=200, first_radius=0.1, sparse=True, verbose=False)
        cf_x = exp.enemy.reshape(-1)

        # NOTA sull'implementazione originale della classe CounterfactualExplanation
        # Da counterfactuals.py abbiamo che self.enemy = cf.find_counterfactuals() che è un metodo della classe GS
        # ritorna il closest enemy e performa già il reshape 
        #closest_ennemy_ = sorted(ennemies_,key= lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))[0] 

        # trasformazione lambda tra x e self.obs_to_interprete (che sarebbe questo self.obs_to_interprete = obs_to_interprete
        #self.prediction_fn = prediction_fn
        #self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
        # con self.obs_to_inteprete definito così obs_to_interprete: instance whose prediction is to be interpreded

        # il formato non è menzionato, però quando chiamo il torch_model.predict ottengo un'array delle due predizioni
        # infatti la dimensione era (8, ) e (2, ) quindi il 2 è la dimensione dell'array label (sbagliato quindi perchè ne deve prendere uno)

        # tra i parametri c'è anche prediction_fn: prediction function, must return an integer label
        # quindi la funzione predict deve ritornare una label invece dell'array 

        # lets try with the predictone function of the wrapper?

    
        cfs= {}

        for idx, feature in enumerate(features_names): # if counterfactual is found we add to the dict
            
            val = cf_x[idx]

            if feature in cats: # papabile modifica se con le categoriche vi sono problemi
                val= round(val)

            cfs[feature]= val

        # ok erano indentate dentro quindi per questo mi dava l'errore di chiave 
        elapsed_time = time.time() - start_time
        cf_vals= np.array([cfs[f] for f in curr_rec.keys()]) # we take the vals for the counterfactual found
        curr_vals = np.array(list(curr_rec.values()))
        sparsity = np.sum(curr_vals != cf_vals)

        return cfs, elapsed_time, sparsity

    except Exception as e:
        print(f"GS fallito {e}")
         
 
#4. Fallback: se nessun controfattuale è stato trovato, usa l'esempio più simile con classe opposta
    #print("Controfattuali non trovati. Passo al fallback con l'esempio più simile.")

    # nella funzione originale si chiamava nuovamente predict_one sia sull'istanza in esame x, che sulla x_h (che sarebbe la sua più vicina con classe opposta)
    # poi veniva applicato un filtro per prendere solo i record di class opposta
    #qui chiaramente siccome usiamo torch, dobbiamo richiamare tutti i passaggi necessari in MiC per dare la predizione

    # trasformiamo il record corrente di cui vogliamo la explanation in tensore
    rec_t = query_instance.values.astype(np.float32)
    
    # 4a. torch model produce la label per il current record
    with torch.no_grad():
        curr_pred= int(torch_model.predict(rec_t)[0])

    # 4b. Stabiliamo la classe opposta 
    target_class= 1- curr_pred # maschera che usiamo dopo per filtrare in modo calcolare le distanze tra il record corrente e il dataframe di predizioni di classe opposta
   
    fallback_data= (log.drop(columns= target, axis= 1)) # this is basically the preprocessing we did when predicting the labels in MiC; drop the target col
    #data_t= fallback_data.values.astype(np.float32) 


    # 4c. Produce prediction per il restante batch log
    with torch.no_grad():  # produciamo tutte le predizioni di nuovo
        hist_pred = torch_model.predict(fallback_data)

    # 4d. Filtriamo il batch usando la mask per avere solo le righe di classe opposta
    mask= (hist_pred == target_class)
    data_opposite_classes= fallback_data[mask] 

#5. Calcolo distanze + sparsities
    if not data_opposite_classes.empty:   
        distances= cdist(query_instance, data_opposite_classes, metric= 'euclidean').flatten() # calcolo distanza euclid tra il record corrente e il DF storico con la classe opposta
        min_d= np.argmin(distances) # assegnazione della distanza minima

        nearest_opp_rec= data_opposite_classes.iloc[min_d].to_dict() # lookup del record più 'vicino' di classe opposta rispetto a quello corrente 
        
        #nearest_cf= np.array([nearest_opp_rec[f] for f in curr_rec.keys()])
        #curr_vals= np.array(list(curr_rec.values()))

        nearest_cf = np.array([nearest_opp_rec[f] for f in curr_rec.keys() if f in nearest_opp_rec])
        curr_vals = np.array([val for f, val in curr_rec.items() if f in nearest_opp_rec])

        sparsity = np.sum(curr_vals != nearest_cf)
        elapsed_time = time.time() - start_time

        return nearest_opp_rec, elapsed_time, sparsity
    
    elapsed_time = time.time() - start_time
    return curr_rec, elapsed_time, 0






# with open(os.path.join(dir,'User_'+self.name+str(self.mic_model_name)+'model.pkl'), 'wb') as file:
                            #pickle.dump(self.mic_model, file)



def assess_risk(ground_truth, mach_pred, current_confidence, threshold_conf= 0.8):
        # ok so basically the idea is to create a priority queue storing problematic instances
        # basically, during the main loop in BRIDGET, once a record is process, we update the log and have access to the current
        # conf level of the machine for that specific record

        # the issue is basically to correct cases when the machine got the final decision, and it was wrong
        # we want to revisit these instances again, and we could store them in a queue that gets evaluated at the start of the next iteration

        # we could design it as a ranking system, i.e. 
        # 1. low priority instance is one the machine had predicted wrongly with LOW confidence
        # 2. high priority instance if the machine was wrong ultimately, with confidence > threshold (def 0.8 seems reasonable?)

        # now the issue is how the instances would be evaluated at the next iter since we ideally dont want to add instances to the queue
        # so to empy it we could aim to re evaluate 4/5 instances max (according to an extra budget allocated by the user)
        # or at least empty the high priority ones, so when inserting an item to the queue just add a counter to the tuple
        # store it like instance (priority_level, counter)
        # the counter increases everytime the item stays in the queue

        #also if two instance have the exact same priority level and counter, 
        # the one with the higher counter gets to be evaluated first

        # wrt to the ranking, since heapq natively sorts in increasing order, for the priority 
        # we'll do 1 = Max priority, 2 Low priority, 0 for non problematic instances
        if mach_pred != ground_truth:
            if current_confidence >= threshold_conf:
                return 1 
            else:
                return 2
        return 0






