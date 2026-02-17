from river import metrics
import random
import numpy as np
import time
from tqdm import tqdm
from classes import BetaUser, DeferralNet, RiverModelWrapper, PyTorchWrapper, PriorityManager
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
from copy import deepcopy

import pickle
import os 
from collections import Counter
import orjson
from river import metrics
from bridget_utils import *
import heapq



class BRIDGET:
    def __init__(self, 
                 dataset_name, user_name, 
                 df_batch1, df_batch3, batch1_test, 
                 target, 
                 user_model, 
                 protected, cats, num,
                 preprocessor=None):
        
        self.dataset_name= dataset_name
        self.name= user_name
        self.user_model = user_model 
        
        # -- Batch 1: 60 %
        self.df_batch1 = df_batch1 ## il batch 1 , cioè lo stato iniziale prima del decision making di BRIDGET
        # WARNING!! questo batch1 viene già assunto separato totalmente dalla porzione di avviamento usata nella funzione train

        # -- Batch 3: 20% usato esclusivamente per testare la fase MiC
        # inserito all'inizio perchè funge da "lookup" dict da quale attingere la riga giusta 
        # da appendere al log con i valori delle nuove colonne nella fase MiC
        self.df_batch3= df_batch3

        # -- Slicing over batch 1
        self.X = self.df_batch1.loc[:, self.df_batch1.columns != target]
        self.X = list(self.X.to_dict(orient='index').values())

        # self. X è una lista di dizionari, è comodo per costruire il logging, 
        # quindi questo va fatto pure per il df batch3 separatamente
        # ogni elemento della lista è un dizionario dove le chiavi sono i nomi delle colonne e i valori sono i dati della riga
        
        self.Y = list(self.df_batch1[target])
        self.Y = [int(y) for y in self.Y] 

        
        self.X_test1= batch1_test.loc[:, batch1_test.columns != target]  
        self.X_test1= list(self.X_test1.to_dict(orient= 'index').values())
    
        self.Y_test1= list(batch1_test[target])
        self.Y_test1 = [int(y) for y in self.Y_test1]
        

        self.target = target # str format
        self.attr_list = list(df_batch1.columns) # list of str
        self.protected= protected  # list of str
        self.protected_values=df_batch1[protected[0]].unique()

        self.cats = cats
        

        self.num= num
        self.preprocessor= preprocessor
        self.feature_names = [c for c in self.attr_list if c!=self.target]
        
        
        self.train_check = False
              
        
        # -- Stats Dictionary to get coverage and stuff
        self.stats = dict()
        self.stats[False] = dict()
        self.stats[True] = dict()
        for e in ['user', 'machine']:
            self.stats[False][e] = dict()
            self.stats[True][e] = dict()
            self.stats[False][e]['tried'] = 0
            self.stats[True][e]['tried'] = 0
            self.stats[False][e]['got'] = 0
            self.stats[True][e]['got'] = 0
           
            if e == 'user':
                self.stats[False][e]['conf'] = 1
                self.stats[True][e]['conf'] = 1
            else:
                self.stats[False][e]['conf'] = 0
                self.stats[True][e]['conf'] = 0

       

    def switch_phase(self, drift_detected, current_phase, current_log, strat=None):
    
        if drift_detected:
            print("\n ALERT \n")
            print("Drift detected, loading metrics...")
            print(f"Current phase: {current_phase}")
            

            if current_phase == 'HiC':
                print(f"Current desired perfomance: {self.desired_performance}")
                print(f"Last 5 FEA values: {self.machine_fea[-5:]}")

            else: 
                print(f"Current belief_threshold: {self.belief_threshold}")
                print(f"Last 5 FEA values: {self.fea_mic[-5:]}")


            current_log.to_csv(f".\processed_data\{self.dataset_name}\drift_checkpoints\{current_phase}_{self.name}_{strat}", 
                               index= False)

            return True
        
        return False
            


class HiC(BRIDGET):

    def __init__(self,
                 RULE, PAST, SKEPT, GROUP, EVA, 
                 n_bins, n_var, maxc, 
                 rule_att, rule_value, 
                 hic_model_name, hic_model,
                 warm_start_performance, # we input the accuracy/f1 score obtained by the incremental model during pre-training
                 allocated_budget = 200,  # so we can fix it sorta as a benchmark, meaning we ideally want to achieve a +5% within the available budget):
                 emergency_budget= 50,
                 user_patience= 5,
                 performance_delta= 0.05,
                 **kwargs):  # budget (representing more like fatigue) of the user, works like an exit condition
                
        

        super().__init__(**kwargs)
         

        self.RULE = RULE
        self.PAST = PAST
        self.SKEPT = SKEPT
        self.GROUP = GROUP
        self.EVA = EVA
        self.n_bins = n_bins
        self.n_var = n_var
        self.maxc = maxc

        self.rule_att = rule_att
        self.rule_value = rule_value
       
        self.warm_start_performance= warm_start_performance
        self.allocated_budget= allocated_budget
        self.emergency_budget= emergency_budget
        self.user_patience=user_patience

        self.desired_performance= self.warm_start_performance + (self.warm_start_performance*performance_delta)
        self.spent_budget= 0
        self.spent_emergency_budget= 0
        self.priority_q= PriorityManager()
        
        
        self.hic_model= hic_model
        self.hic_model_name= hic_model_name
        self.machine_fea= []
        self.user_fea= []

        self.initial_model = pickle.loads(pickle.dumps(self.hic_model))

        self.hic_evaluation_results = []
        self.fea_num_u_hic =0
        self.fea_num_m_hic =0

        #various counters for testing/debugging purposes

        self.rules_count = 0
        self.past_count = 0
        self.ok_count = 0
        self.no_count = 0
        self.xai_check= 0
        self.xai_ok = 0
        self.xai_no = 0
        self.skept_count = 0
        self.agree_count = 0
        self.disagree_count = 0
        self.hic_acc = metrics.Accuracy()
        self.hic_F1 = metrics.F1()
        self.fairness_records = [len(self.X) - 1]
        for i in range(0, 100, 5)[1:]:
            self.fairness_records.append(percentage(i, len(self.X)))

        self.retrain_count= 0

        self.q_stats= dict()

        self.q_stats={
            'instances_processed': 0,
            'user_confirmed' : 0,
            'user_rethinked' : 0,
            'priority_1': 0,
            'priority_2': 0
            }



    def train(self, x_avv, y_avv, x_test, y_test):
        # QUESTA FUNZIONE VIENE CHIAMATA PRIMA DI ENTRARE IN BRIDGET !!!

        # i dati di x_data e y_data fanno parte di un piccolo dataset di avviamento per fittare il modello incrementale
        # che non entra assolutamente nel processo decisionale di BRIDGET
        # e di conseguenza non sono la base sulla quale si costruisce il log
        
        """
        Funzione per calibrare il modello incrementale che poi diventa self.initial model
        
        :param self: Description
        :param x_data: data stream necessario
        :param y_data: data stream con le labels
        """

        accuracy = metrics.Accuracy()
        f1= metrics.F1()
        predictions = []
    
        
        for x, y in zip(x_avv, y_avv):
            y = int(y)

            if self.preprocessor is not None:
                self.preprocessor.learn_one(x)
                x= self.preprocessor.transform_one(x)
        
            self.hic_model.learn_one(x, y)
    

        for x, y in zip(x_test, y_test):
            y = int(y)
            
            if self.preprocessor is not None:
                x = self.preprocessor.transform_one(x)

            y_pred = self.hic_model.predict_one(x)
            
            accuracy.update(y, y_pred)
            f1.update(y, y_pred)
            predictions.append(y_pred)
    

        print(f"{accuracy}")
        print(f"{f1}")
        print(f"Distribution of predictions: {Counter(predictions)}")

        self.train_check = True
        print(f"{self.hic_model} trained")
        return self 




    def start_HiC(self, df_avv):
            
            self.processed= dict()
            queue_log= []

            machine_predictions = []
            machine_conf_lvls= []
            accuracy_score = []#Lista usata per salvare i vari punteggi di accuracy
            f1_score = []#Lista salvata per salvare i vari punteggi di f1
            
            frank_cms = []
            equality = []#Lista usata per verificare quando cambiano i modelli


            # FEA structures
            skepticisms = []
            fea_num_machine= 0.0
            fea_num_user= 0.0

            fea_den_machine= 0.0
            fea_den_user= 0.0


            # XAI Structures

            proximities_KNN_SAME=[]
            proximities_KNN_OPP= []
            times_KNN= []
            sparsities_KNN_SAME= []
            sparsities_KNN_OPP= []

            similar_nn = []
            opposite_nn = []


            self.X = convert_dict_list_to_float32(self.X)

            for i in tqdm(range(len(self.X))):      

                relabel = False #When this is set to True, Re-Labelling is triggered

                # - BOOSTING- INSPIRED error treating procedure (hopefully)
                curr_patience= 0
                
                while not self.priority_q.is_empty() and curr_patience <= self.user_patience and self.spent_emergency_budget < self.emergency_budget:
                    print("processing the queue...")
                    prio, old_idx= self.priority_q.exit_queue()

                    old_x = self.X[old_idx]
                    old_y= self.Y[old_idx]
                    old_rec= tuple(list(old_x.values()))

                    # now, to keep it even more realistic we could distinguish the two cases
                    # meaning, we ask the user to provide the new prediction to see if they changed their mind
                    
                    user_pred= int(self.user_model.predict(old_rec, old_y, old_idx))
                    prev_user= self.processed[old_rec]['user']
                    provider= 'H'

                    if user_pred == prev_user: # since it was the same, we dont change the prediction given by the user
                        # we do however change the flag to H
                        decision= user_pred
                        self.q_stats['user_confirmed'] += 1
                        
                               

                    else: # now this is a little bit controversial since we assume that this time the user is fully onboard
                        # and rethinks its past decision every time
                        # but to keep it even more realistic, the function .confirm() could be called, just like we do for
                        # duplicated records and regular instances
                        # since this is just a test, i wont call it, because i want to assess if the q structure works as intended
                        decision= user_pred
                        self.processed[old_rec]['user']= user_pred
                        self.q_stats['user_rethinked'] += 1
                        

                        #we leave the got/tried counters as it is and we dont adjust the FEA (maybe? or it could be cheating)


                    # Updating logs
                    self.processed[old_rec]['decision'] = decision
                    self.processed[old_rec]['provider_flag']= provider

                    self.q_stats['instances_processed'] += 1
                    if prio == 1:
                        self.q_stats['priority_1'] += 1
                    else:
                        self.q_stats['priority_2'] += 1  


                    self.hic_model.learn_one(old_x, decision)

                    # incrementing    
                    curr_patience += 1
                    self.spent_emergency_budget+= 1



                x = self.X[i]
                y = int(self.Y[i])
 
                
                if self.preprocessor is not None:
                    self.preprocessor.learn_one(x)
                    x= self.preprocessor.transform_one(x) 

                
                record = tuple(list(x.values()))
                user_truth = int(self.user_model.predict(record, y, i))

                machine_prediction = int(self.hic_model.predict_one(x)) # forzato int perchè adwin produce true/false            
                machine_predictions.append(machine_prediction)

                if record in self.processed: #Duplicated record
                    self.processed[record]['times'] += 1
                    self.spent_emergency_budget += 1

                    #print("Record already processed...")
                    old_decision = int(self.processed[record]['decision'])                    

                    if user_truth == old_decision:
                        #print("And you are consistent! Decision accepted.")
                        decision = old_decision
                    
                    else:
                        #print("Inconsistent. You previously said:", old_decision, "Want to change old decision?")
                        confirm = random.choices(population=[False, True], weights=[0.8, 0.2], k=1)[0]

                        if confirm == False:
                            decision = old_decision
                            
                        else:
                            decision = user_truth
                            relabel = True
                    
                    self.stats[user_truth]['user']['tried'] += 1
                    self.stats[machine_prediction]['machine']['tried'] += 1
            
                    if decision == user_truth:
                        self.stats[user_truth]['user']['got'] += 1

                    if decision == machine_prediction: 
                        self.stats[machine_prediction]['machine']['got'] += 1
                    
                    
                
                else:  # LOGIC FOR UNSEEN RECORDS

                    try:
                        pred_proba = self.hic_model.predict_proba_one(x)[machine_prediction]

                    except:
                        pred_proba = 0

                    try:
                        user_proba = self.hic_model.predict_proba_one(x)[user_truth]
                    except:
                        #print("Still unlearned...")
                        user_proba = 0.8

                    self.stats[machine_prediction]['machine']['tried'] += 1
                    self.stats[user_truth]['user']['tried'] += 1
                                        
                    machine_conf_lvls.append(pred_proba)
                    
                    # FEA COMPUTATION (con la nuova formula il for loop ammazza tutto)
                    
                    """
                    if i == 0:
                        user_fea= 1.0
                        mach_fea= 0.0
                    else:
                        user_fea= fea_computation(i, user_truth, self.processed, 0.90, False, 1800)           
                        mach_fea= fea_computation(i, machine_prediction, self.processed, 0.90, 1800)
                    """

                    fea_num_machine *= 0.99
                    fea_num_user *= 0.99

                    fea_den_machine *= 0.99
                    fea_den_user *= 0.99

                    if machine_prediction == y:
                        fea_num_machine += 1 

                    if user_truth == y:
                        fea_num_user += 1

                    fea_den_machine += 1
                    fea_den_user += 1

                    mach_fea= fea_num_machine / fea_den_machine if fea_den_machine > 0 else 0.5
                    user_fea= fea_num_user/ fea_den_user if fea_den_user > 0 else 1.0

                    self.user_fea.append(user_fea)
                    self.machine_fea.append(mach_fea)
                    
                    

                    # - UPDATING LOG
                    self.processed[record] = dict()
                    self.processed[record]['notes'] = []
                    self.processed[record]['vs'] = None
                    self.processed[record]['ideal'] = None
                    self.processed[record]['times'] = 1

                    self.processed[record]['idx']= i
                    self.processed[record]['dict_form'] = x
                    self.processed[record]['user'] = user_truth
                    self.processed[record]['machine'] = machine_prediction
                    self.processed[record]['ground_truth'] = y
                    self.processed[record]['proba_model'] = pred_proba

                    self.spent_budget += 1


                    # - LABELING LOGIC

                    provider= 'H'
                    ideal_value = ideal_record_test(x, self.rule_att, self.rule_value) #Is record covered by Ideal Rule Check?

                    vs_records, vs_decision = get_value_swap_records(x, self.processed,
                                                                    self.protected, self.attr_list) #Is record covered by Individual Fairness Check?

                    if user_truth == machine_prediction:
                        skepticism = 0
                    else:
                        skepticism = mach_fea * pred_proba - user_fea * user_proba
                    skepticisms.append({str(i):skepticism})


                    if ideal_value is not None and user_truth != ideal_value and self.RULE: #User is not consistent w.r.t. Ideal Rule
                        self.rules_count += 1
                        decision = ideal_value
                        self.processed[record]['ideal'] = False
                        if machine_prediction == ideal_value:
                            provider= 'M'
                            self.stats[machine_prediction]['machine']['got'] += 1


                    elif ideal_value is not None and user_truth == ideal_value and self.RULE: #User is consistent w.r.t. Ideal Rule
                        decision = ideal_value
                        provider= 'H'  # perchè l'umano aveva ragione, c'è scritto sopra
                        self.processed[record]['ideal'] = True
                        if machine_prediction == ideal_value:
                            self.stats[machine_prediction]['machine']['got'] += 1



                    elif vs_decision is not None and user_truth != vs_decision and self.PAST: #IRC not triggered. User not consistent w.r.t. Individual Fairnesss
                        self.processed[record]['vs'] = True
                        self.past_count += 1
                        for rec in vs_records:
                            self.processed[rec]['vs'] = True
                        confirm = random.choices(population=[False, True], weights=[0.8, 0.2], k=1)[0]
                        if confirm in [0, "0", False]:
                            decision = vs_decision
                            if machine_prediction == vs_decision:
                                provider= 'M'
                                self.stats[machine_prediction]['machine']['got'] += 1

                        elif confirm in [1, "1", True]:
                            decision = user_truth
                            provider= 'H'
                            self.stats[user_truth]['user']['got'] += 1
                            if machine_prediction == user_truth:
                                self.stats[machine_prediction]['machine']['got'] += 1
                            for rec in vs_records:
                                self.processed[rec]['decision'] = user_truth
                            relabel = True

                    elif vs_decision is not None and user_truth == vs_decision and self.PAST: #IRC not triggered. User not consistent w.r.t. Individual Fairnesss
                        self.processed[record]['vs'] = True
                        for rec in vs_records:
                            self.processed[rec]['vs'] = True
                        decision = vs_decision
                        provider= 'H'
                        if machine_prediction == vs_decision:
                            self.stats[machine_prediction]['machine']['got'] += 1
                    
                    else: #Other conditions not triggered. Skeptical Learning Check
                        if user_truth != machine_prediction and self.SKEPT:

                            if skepticism > 0.6 :
                                self.skept_count += 1
                                self.xai_check = 0
                                confirm = self.user_model.believe() 

                                #confirm = None
                                
                                if confirm == None:
                                    
                                    start_time = time.time()
                                    xai_log = prepr_log_for_xai(df_avv, self.processed, self.attr_list, self.target)

                                    nearest_ex, nearest_opp, sparsity_ex, sparsity_opp = get_neighbors(record, 
                                                                            y, 
                                                                            xai_log, 
                                                                            relevance_window= 50, 
                                                                            n_neighbors= 2
                                                                            
                                                                            )
                                    
                                    time_KNN= time.time()-start_time

                                    similar_nn.append(nearest_ex)
                                    opposite_nn.append(nearest_opp)

                                    # no plausibility perchè li pesco dal log anyways 

                                    # Metrics:
                    
                                    #1. Proximity: distanza tra x e df, SAME CLASS VER

                                    distance_x_nn_ex= calculate_distances(x, nearest_ex) # distance ritorna il formato lista = [(valido esempio:distanza)]
                                    proximity_nn_ex= distance_x_nn_ex[0][1]
                                    

                                    #2. Proximity: distanza tra x e df, OPPOSITE CLASS VER
                                    distance_x_nn_opp= calculate_distances(x, nearest_opp) # distance ritorna il formato lista = [(valido esempio:distanza)]
                                    proximity_nn_opp= distance_x_nn_opp[0][1]

                                    proximities_KNN_SAME.append({str(record):proximity_nn_ex})
                                    proximities_KNN_OPP.append({str(record):proximity_nn_opp})

                                    times_KNN.append(time_KNN)

                                    sparsities_KNN_SAME.append({str(record):sparsity_ex})
                                    sparsities_KNN_OPP.append({str(record):sparsity_opp})
                                    
                                    
                                    n_examples= len(nearest_ex) + len(nearest_opp)
                                    
                                    if n_examples > 0:

                                        for e in nearest_ex.values:
                                            # ora devo riprendere nuovamente la colonna della gt perchè il mio user
                                            # necessita della gt originale assolutamente per predirre
                                            gt_col = e[-1] # riprendo nuovamente la colonna della gt
                                            rec_feats= e[:-1]

                                            user_opinion = self.user_model.predict(rec_feats, gt_col, i)
                                            if user_opinion == machine_prediction:
                                                self.xai_check += 1
                                                self.xai_ok += 1
                                            else:
                                                self.xai_no += 1

                                        for e in nearest_opp.values:

                                            gt_col = e[-1] # riprendo nuovamente la colonna della gt
                                            rec_feats= e[:-1]

                                            user_opinion = self.user_model.predict(rec_feats,gt_col, i)
                                            if user_opinion != machine_prediction:
                                                self.xai_check += 1
                                                self.xai_ok += 1
                                            else:
                                                self.xai_no += 1

                                        if (self.xai_check / n_examples) > 0.5:
                                            confirm = True
                                        else:
                                            confirm = False
                                    
                                    
                                """  
                                    hic_wrapper= RiverModelWrapper(self.hic_model, self.target, self.feature_names)
                                    
                                    start_time = time.time()

                                    _, df_similars, rules_dict, _, counter_rules_dict = generate_exp_lore(x, 
                                                                            xai_log, 
                                                                            self.target, 
                                                                            hic_wrapper,
                                    )


                                    self.rules.append(rules_dict)
                                    self.counter_rules.append(counter_rules_dict)
                                    
                                    ex_pro_df= filter_df_by_rules(x, xai_log, rules_dict, hic_wrapper)

                                    time_lore_mic= time.time()-start_time
                                
                                    # Calcolo metriche dei similari:
                                    
                                    #1. Proximity: distanza tra x e df similars

                                    distance_x_r_lore= calculate_distances(x, df_similars) # distance ritorna il formato lista = [(valido esempio:distanza)]
                                    proximity_lore= distance_x_r_lore[0][1]
                                    example_lore= distance_x_r_lore[0][0] # esempio più vicino all'istanza x
                                    
                                    # 2. Plausibility: distanza tra filtered_df e l'esempio più "vicino" --> 
                                    # quindi dobbiamo calcolare la dist tra x e il DF similars, prendere quello con dist minore

                                    # per la plausability ci serve il similar più vicino
                                    if not ex_pro_df.empty:

                                        distance_sim_log_lore= calculate_distances(example_lore, ex_pro_df)
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



                                    ex_against_df= filter_df_by_counter_rules(x, xai_log, counter_rules_dict, hic_wrapper)

                                    n_examples= len(ex_pro_df) + len(ex_against_df)

                                    
                                    if n_examples > 0:

                                        for e in ex_pro_df.values:
                                            # ora devo riprendere nuovamente la colonna della gt perchè il mio user
                                            # necessita della gt originale assolutamente per predirre
                                            gt_col = e[-1] # riprendo nuovamente la colonna della gt
                                            rec_feats= e[:-1]

                                            user_opinion = self.user_model.predict(rec_feats, gt_col, i)
                                            if user_opinion == machine_prediction:
                                                self.xai_check += 1
                                                self.xai_ok += 1
                                            else:
                                                self.xai_no += 1

                                        for e in ex_against_df.values:

                                            gt_col = e[-1] # riprendo nuovamente la colonna della gt
                                            rec_feats= e[:-1]

                                            user_opinion = self.user_model.predict(rec_feats,gt_col, i)
                                            if user_opinion != machine_prediction:
                                                self.xai_check += 1
                                                self.xai_ok += 1
                                            else:
                                                self.xai_no += 1

                                        if (self.xai_check / n_examples) > 0.5:
                                            confirm = True
                                        else:
                                            confirm = False
                                    """
            

                                if confirm in [0, "0", False]:
                                    self.no_count += 1
                                    decision = user_truth
                                    provider= 'H'
                                    self.stats[user_truth]['user']['got'] += 1
                                    
                                                            
                                else:
                                    self.ok_count += 1
                                    decision = machine_prediction
                                    provider= 'M'
                                    self.stats[machine_prediction]['machine']['got'] += 1

                                    
                            else:
                                self.disagree_count += 1
                                decision = user_truth
                                provider= 'H'
                                self.stats[user_truth]['user']['got'] += 1
                                
                        else:
                            self.agree_count += 1
                            decision = user_truth
                            provider='H'
                            self.stats[user_truth]['user']['got'] += 1
                            self.stats[machine_prediction]['machine']['got'] += 1
                            
                            

                    #Once the final decision has been taken, the model is updated. Internal data structure is also updated
                    
                    self.processed[record]['decision'] = int(decision) # sometimes AdwinBagging and Adaboost provide bool, so cast to int
                    self.processed[record]['provider_flag'] = provider

                    self.hic_model.learn_one(x, decision)
                                        
                    
                   
                    try:
                        self.hic_acc.update(decision,machine_prediction)
                        self.hic_F1.update(decision,machine_prediction)
                        

                    except:
                        print('err', x, decision)
                        self.hic_model = pickle.loads(pickle.dumps(self.initial_model))

                        #for x_train_sample, y_train_sample in zip(x_avv, y_avv):
                            #self.hic_model.learn_one(x_train_sample, y_train_sample)

                        for data in self.processed.values():
                        
                            #x_relabel = self.processed[proc]['dict_form']
                            #y_relabel = self.processed[proc]['decision']
                            self.retrain_count += 1
                            self.hic_model.learn_one(data['dict_form'], data['decision']) #x_relabel, y_relabel)

                        # se learn one fallisce facciamo il re-training 

                    
                    
                
                try:
                    self.stats[user_truth]['user']['conf'] = self.stats[user_truth]['user']['got'] / self.stats[user_truth]['user']['tried']
                except:
                    self.stats[user_truth]['user']['conf'] = 1

                try:
                    self.stats[machine_prediction]['machine']['conf'] = self.stats[machine_prediction]['machine']['got'] / self.stats[machine_prediction]['machine']['tried']
                except:
                    self.stats[machine_prediction]['machine']['conf'] = 0



                if relabel == True:

                    self.hic_model = deepcopy(self.initial_model)

                    #for x_train_sample, y_train_sample in zip(x_avv, y_avv):
                     #   self.hic_model.learn_one(x_train_sample, y_train_sample)

                    for proc in (self.processed.keys()):
                    
                        x_relabel = self.processed[proc]['dict_form']
                        y_relabel = self.processed[proc]['decision']
                        self.retrain_count += 1
                        self.hic_model.learn_one(x_relabel, y_relabel)

                
                            
                if i in self.fairness_records and self.GROUP:
                    DN, PP, _ = get_fairness(self.hic_model, self.protected, self.processed, self.protected_values)
                    fairnes_relabel = DN[:round(len(DN) * 0.25)] + PP[:round(len(PP) * 0.25)]
                    for e in fairnes_relabel:
                        #self.processed[e[1]][self.target] = not self.processed[e[1]][self.target] per un booleano
                        self.processed[e[1]]['decision'] = 1 - self.processed[e[1]]['decision']
                    
                    self.hic_model = pickle.loads(pickle.dumps(self.initial_model))

                    #for x_train_sample, y_train_sample in zip(x_avv, y_avv):
                     #   self.hic_model.learn_one(x_train_sample, y_train_sample)
                    
                    for proc in (self.processed.keys()):
                        
                        x_relabel = self.processed[proc]['dict_form']
                        y_relabel = self.processed[proc]['decision']
                        self.retrain_count += 1
                        self.hic_model.learn_one(x_relabel, y_relabel)
                        
                    #percentage_dict,_ = get_percentage_and_df(None,self.processed,self.target)
                

                #hic_drift= drift_check_HiC(self.machine_fea_lst, self.drift_threshold, iter_count= i, k_max=1000)

                hic_drift= exit_HiC(self.allocated_budget, self.spent_budget, self.machine_fea, self.desired_performance)


                # prima di passare avanti, chiamiamo l'assess risk e vediamo se va inserito nella priority queue

                if provider== 'M' and decision != y:
                    priority_v= assess_risk(y, machine_prediction, pred_proba)
                    if priority_v >0:
                        self.priority_q.add_to_queue(priority_v, i)

                        queue_log.append({
                            'idx': i,
                            'feats': x,
                            'priority': priority_v
                        })


                if hic_drift:
                    #print(f"Drift here! Record index {record}")
                    dir = os.path.join("HIC_res", f"{self.dataset_name}", f"results_{self.name}")
                   
                    hic_pref = f"HIC_DRIFT_User_{self.name}_{self.hic_model_name}"

                    metrics_knn = {
                    "time_steps": times_KNN,
                    "sparsity_SAME": sparsities_KNN_SAME,
                    "sparsity_OPP": sparsities_KNN_OPP,
                    "proximity_SAME": proximities_KNN_SAME,
                    "proximity_OPP": proximities_KNN_OPP,
                    "method": "KNN",  # Aggiungi il name del metodo
                    "dataset": self.name
                    }


                    skept = {
                        "skept":skepticisms
                    }

                    hic_d = {
                    "model.pkl": self.hic_model,
                    "Accuracy.txt": accuracy_score,
                    "F1.txt": f1_score,
                    "Machine_Confidence.txt": machine_conf_lvls,
                    "times_KNN.txt": times_KNN,
                    "KNN_metrics.json": metrics_knn,
                    "skept.json": skept,
                    "HiC_FEA_machine.txt": self.machine_fea,
                    "HiC_FEA_user.txt": self.user_fea,
                    "HiC_queue_log.txt": queue_log,
                    "HiC_queue_stats.txt": self.q_stats
                    }
                    
                    save_data(dir, hic_pref, hic_d)
                    
                    _, df_drift_log = get_percentage_and_df(None, self.processed,self.target) 

                    super().switch_phase(hic_drift, current_phase= 'HiC', current_log=df_drift_log)
                    return df_drift_log,  self.hic_evaluation_results, equality, self.skept_count
                

                    
                if self.EVA:
                    human_fairness, human_acc, systemic = evaluation_human(self.processed, self.protected, self.Y,
                                                                        self.attr_list)
                
                    frank_fairness, frank_acc,frank_f1,_ = evaluation_frank(self.X_test1, self.Y_test1, self.hic_model, self.protected, self.preprocessor)
                    accuracy_score.append(frank_acc)
                    f1_score.append(frank_f1)
                    self.hic_evaluation_results.append([human_fairness, human_acc, systemic, frank_fairness, frank_acc,
                                                    self.rules_count, self.past_count,
                                                    self.ok_count, self.no_count,
                                                    self.xai_ok, self.xai_no,
                                                    self.skept_count, self.agree_count, self.disagree_count
                                                    ])
                    

                if i ==  (len(self.X)-1):
                    
                    dir = os.path.join("HIC_res", f"{self.dataset_name}", f"results_{self.name}")

                    hic_pref = f"User_{self.name}_{self.hic_model_name}"

                     

                    metrics_knn = {
                    "time_steps": times_KNN,
                    "sparsity_SAME": sparsities_KNN_SAME,
                    "sparsity_OPP": sparsities_KNN_OPP,
                    "proximity_SAME": proximities_KNN_SAME,
                    "proximity_OPP": proximities_KNN_OPP,
                    "method": "KNN",  
                    "dataset": self.name
                    }


                    skept = {
                        "skept":skepticisms
                    }

                    hic_d = {
                    "model.pkl": self.hic_model,
                    "Accuracy.txt": accuracy_score,
                    "F1.txt": f1_score,
                    "Machine_Confidence.txt": machine_conf_lvls,
                    "times_KNN.txt": times_KNN,
                    "KNN_metrics.json": metrics_knn,
                    "skept.json": skept,
                    "HiC_FEA_machine.txt": self.machine_fea,
                    "HiC_FEA_user.txt": self.user_fea,
                    "HiC_queue_log.txt": queue_log,
                    "HiC_queue_stats.txt": self.q_stats
                    }
                    
                    save_data(dir, hic_pref, hic_d)



            _, df_final_hic = get_percentage_and_df(None, self.processed,self.target) 
            
            return df_final_hic, self.hic_evaluation_results, equality, self.skept_count


class MiC(BRIDGET):

    def __init__(self,
                 mic_model, 
                 mic_model_name,
                 benchmark_performance,
                 user_patience= 200,
                 performance_delta= 0.05,
                 belief_threshold= 0.7,
                 tau_threshold= None,
                 anqi_mao_thresh=None,
                 **kwargs
                 ):
        
        
        super().__init__(**kwargs)

        
        self.mic_model= mic_model  ## modello già trainato
        self.mic_model_name= mic_model_name
        self.benchmark_performance=benchmark_performance
        
        self.performance_delta= performance_delta
        self.user_patience= user_patience

        self.belief_threshold= belief_threshold
        self.tau= tau_threshold
        self.anqi_mao_thresh= anqi_mao_thresh
        # works just like the one in HiC, however like a lower performance bound
        self.performance_thresh= self.benchmark_performance - (self.benchmark_performance * self.performance_delta) 

        # containers delle metriche di valutazione custom
        self.mic_acc= 0.0
        self.model_acc= 0.0

        self.mic_preds= []  # decisioni dell'intero sistema MiC
        self.low_belief_count= 0
        self.deferred_decisions= 0

        self.fea_mic= []


    def start_MiC(self, x_stream, y_stream, df_switch, r_net=None, two_step_deferral= None): 
        ## Df_switch è l'output di HiC, ricorda di passarlo come DATA FRAME

        # df_batch3 è il terzo batch di dati che usiamo esclusivamente per la valutazione della strategia di deferral
        # viene usato per continuare a costruire il log

        # x_stream e y_stream sono tensori preprocessati in orchestrazione per la net del batch3

        # in sostanza sarebbe df_batch3 attributo della classe, ma trasformato esternamente in tensore


        # -- Log
        self.processed= dict()

        fea_mic_num = 0
        fea_mic_den = 0
      
        mach_confidence= [] # lista contenente i valori di predict proba per le singole istanze processate dalla machine
        mach_predictions= [] # lista contenente tutte le predizioni della machine

        # GROWING SPHERES structures
        times_GS= []
        sparsities_GS = []
        proximities_GS = []  # queste metriche sono tutte calcolate su dati già scalati usando self.scaler
        plausabilities_GS = []

       

        ## Main Loop
        
        
        for record, (x_rec, y_gt) in enumerate(tqdm(zip(x_stream, y_stream), total=len(x_stream))):
                
            self.processed[record]= dict() # analogo a quando in HiC si inizializza vuoto per gli unseen records
    
            x = self.df_batch3[self.feature_names].iloc[record].to_dict()
           
            y_gt= int(y_gt.item())

            self.processed[record]['dict_form'] = x
            self.processed[record]['ground_truth'] = y_gt
            self.processed[record]['idx'] = record

            ## si ottengono i risultati della net
            with torch.no_grad():

                if x_rec.ndim == 1:
                    x_rec = x_rec.unsqueeze(0)

                outputs= self.mic_model(x_rec) ##  qui ritorniamo i logits scores delle classi

                ## da pytorch: non è necessario scrivere model.forward, ma basta passare direttamente model(x)
                
                ## quindi qui ora da outputs dobbiamo ricavare la classe con output in logit maggiore

                net_output= torch.argmax(outputs, dim=1).item()
                mach_predictions.append(net_output)

            ## - PREDICT PROBA, LOGGING
            try:
                probas= self.mic_model.predict_proba_nn(x_rec)
                max_conf = float(np.max(probas))
                mach_confidence.append(max_conf)

            except Exception as e:
                print(f"Error: {e}") 
                max_conf = 0.0
            
            self.stats[net_output]['machine']['tried']+=1
    

            ## - DEFERRAL LOGIC 
            
            if two_step_deferral: ## ANQI MAO LOGIC
                 
                deferral_proba= p_defer(x_rec, r_net)
                p_val = deferral_proba.item()

                if p_val >= self.anqi_mao_thresh:
                    
                    
                    x_input_user = x_rec.squeeze().cpu().numpy()    
                    user_pred = self.user_model.predict(x_input_user, y_gt, record)
                    decision = user_pred
                    provider= 'H'
                    self.deferred_decisions +=1
                    self.stats[user_pred]['user']['tried'] +=1
                    self.stats[user_pred]['user']['got'] += 1

            
                
                else:
                    provider= 'M'
                    decision = int(net_output)
                    
                    x_input_user = x_rec.squeeze().cpu().numpy()
                    user_pred = self.user_model.predict(x_input_user, y_gt, record)

                    self.stats[net_output]['machine']['got']+=1
                    


            else:  # CONFIDENCE BASED DEFERRAL
                 
                # 1. No Deferral, Explanation Provided ? these records are reliable according to the results obtained, 
                # so we supply explanations + similar records to maybe "boost" confidence 
                
                if max_conf >= self.tau:

                    decision = int(net_output)
                    x_input_user = x_rec.squeeze().cpu().numpy()
                    user_pred= self.user_model.predict(x_input_user, y_gt, record) 
                    ## prendiamo ugualmente la predizione che avrebbe fornito l'user in questo caso
                    provider= 'M'

                    self.stats[net_output]['machine']['got']+=1




                # 2. Deferral to user: the model was unrealiable here, show counterexamples
                else:
                    
                    x_input_user = x_rec.squeeze().cpu().numpy()
                                
                    user_pred = self.user_model.predict(x_input_user, y_gt, record)
                   
                    decision = user_pred
                    provider = 'H'

                    self.deferred_decisions +=1
                    self.stats[user_pred]['user']['tried'] +=1
                    self.stats[user_pred]['user']['got'] += 1

               
            self.mic_preds.append(decision) # updating the structure



            ### Explanation logic: GROWING SPHERES

            xai_log= prepr_log_for_xai(df_switch, self.processed, self.attr_list, self.target)
            torch_wrapper= PyTorchWrapper(self.mic_model,self.target, self.feature_names)

            cfs, time_GS, sparsity_GS= get_GS_cfe(xai_log, x, torch_wrapper, self.cats, self.target)

            try:

                cf_df = pd.DataFrame([cfs]).drop(columns=[self.target], errors='ignore')
                x_df = pd.DataFrame([x]).drop(columns=[self.target], errors='ignore')
                        
                cf_df = cf_df[x_df.columns]
                        
                d = cdist(cf_df, x_df, metric='euclidean').flatten()[0] # computing distances
                proximity_GS = d

            except Exception as e:
                print(f"Errore calcolo distanza: {e}")
                proximity_GS = np.nan

            distance_cf_hist_GS = calculate_distances(cfs, xai_log, feature_ranges=None)  
            plausability_GS = distance_cf_hist_GS[0][1]
                    
                    ## logging
            proximities_GS.append({str(record):proximity_GS})
            times_GS.append(time_GS)
            sparsities_GS.append({str(record):sparsity_GS})
            plausabilities_GS.append({str(record): plausability_GS})
            


            """
            Implementation for LORE and DICE (works)


            if provider== 'H':
                  
                    xai_log= prepr_log_for_xai(df_switch, self.processed, self.attr_list, self.target)



                    # 1- EXPLANATIONS FORNITE DA DICE
                    # ricorda, queste liste di seguito contengono valori scalati, quindi se si vuole capire bene bisogna ri-trasformarli invertendo lo scaler
                    cf_dice_mic, time_dice_mic, sparsity_dice_mic= get_cfe_DICE_MiC(xai_log, 
                                                                                    x, 
                                                                                    self.mic_model, 
                                                                                    self.cats, 
                                                                                    self.target)
                    
                    
                    
                    try:
                        # Creiamo i DataFrame e rimuoviamo il target se presente
                        cf_df = pd.DataFrame([cf_dice_mic]).drop(columns=[self.target], errors='ignore')
                        x_df = pd.DataFrame([x]).drop(columns=[self.target], errors='ignore')
                        
                        # Ordiniamo le colonne nello stesso identico modo
                        cf_df = cf_df[x_df.columns]
                        
                        # Calcolo distanza
                        d = cdist(cf_df, x_df, metric='euclidean').flatten()[0]
                        proximity_dice_mic = d

                    except Exception as e:
                        print(f"Errore calcolo distanza: {e}")
                        proximity_dice_mic = np.nan

                    distance_cf_dice_hist = calculate_distances(cf_dice_mic, xai_log, feature_ranges=None)  # questa sarebbe la "distanza storica"

                    # xai log è già pandas DF

                    # dist tra il counterfactual prodotto e il df log (composito di df switch prodotto da hic + i record valutati da MiC di volta in volta)
                    # per verificare la plausability del cf e verificare se ha similitudini con record già visti

                    plausability_dice_mic = distance_cf_dice_hist[0][1]
                    ## possibile trasformazione con invert scaler (se utile)


                    ## logging
                    proximities_DICE_mic.append({str(record):proximity_dice_mic})
                    times_DICE_mic.append(time_dice_mic)
                    sparsities_DICE_mic.append({str(record):sparsity_dice_mic})
                    plausabilities_DICE_mic.append({str(record): plausability_dice_mic})

                   
            else:
                    # 1. qui estrazione di esempi simili sulla base di belief or simili in termini di examples estratti con XAI

                    # la cosa importante è ricalcolare nuovamente le metriche di proximity, sparsity e plausability
                        # plausability: verifichiamo quanto è distante l'esempio prodotto, dal log storico( df_switch)
                        # proximity: verify the distance between the current record vs the synthetic neighbor generated by LORE
                    
                    if random.random() < 0.1: 
                    
                        net_wrapper = PyTorchWrapper(self.mic_model, self.target, self.feature_names)

                        xai_log= prepr_log_for_xai(df_switch, self.processed, self.attr_list, self.target)

                        # EXPLAINATIONS FORNITE DA LORE
                        start_time = time.time()
                        
                        _, df_similars, rules, _, _ = generate_exp_lore(x, 
                                                                            xai_log, 
                                                                            self.target, 
                                                                            net_wrapper)
                        
                        
                    

                        filtered_df= filter_df_by_rules(x, xai_log, rules, net_wrapper)
                        time_lore_mic= time.time()-start_time
                    
                        # Calcolo metriche:
                        
                        #1. Proximity: distanza tra x e df similars

                        distance_x_r_lore= calculate_distances(x, df_similars) # distance ritorna il formato lista = [(valido esempio:distanza)]
                        proximity_lore_mic= distance_x_r_lore[0][1]
                        example_lore= distance_x_r_lore[0][0] # esempio più vicino all'istanza x
                        
                        # 2. Plausibility: distanza tra filtered_df e l'esempio più "vicino" --> 
                        # quindi dobbiamo calcolare la dist tra x e il DF similars, prendere quello con dist minore

                        # per la plausability ci serve il similar più vicino
                        if not filtered_df.empty:
                            distance_sim_log_lore= calculate_distances(example_lore, filtered_df)
                            plausibility_lore_mic= distance_sim_log_lore[0][1]
                        else:
                            print("Warning: No historical matches for the given rule.")
                            plausibility_lore_mic = np.nan

                        #3. Sparsity
                        sparsity_lore_mic= calculate_sparsity(x, example_lore)
                    
                        proximities_LORE_mic.append({str(record):proximity_lore_mic})
                        times_LORE_mic.append(time_lore_mic)
                        sparsities_LORE_mic.append({str(record):sparsity_lore_mic})
                        plausabilities_LORE_mic.append({str(record): plausibility_lore_mic})
                """
            

            ## - FEA COMPUTATION

            fea_mic_num *= 0.99  # decay factor could also be passed as input instead of hardcoded, its a detail tho i guess
            fea_mic_den *= 0.99
                

            if net_output == y_gt:
                fea_mic_num += 1 


            fea_mic_den += 1

            fea_mic_model= fea_mic_num / fea_mic_den if fea_mic_den > 0 else 0.5
              
            self.fea_mic.append(fea_mic_model)
                        

            ## - UPDATING LOG
           
            self.processed[record]['user'] = user_pred
            self.processed[record]['machine'] = net_output
            self.processed[record]['proba_model']= max_conf
            self.processed[record]['decision'] = decision
            self.processed[record]['provider_flag'] = provider

            #fea_mic_model= fea_computation(record, net_output, self.processed, 0.99)
          

            ## - DRIFT CHECK

            if torch.is_tensor(max_conf):
                    max_conf = max_conf.item()

            belief= max_conf * fea_mic_model
            

            if belief <= self.belief_threshold:
                self.low_belief_count += 1


            mic_drift= exit_MiC(self.fea_mic,
                                self.user_patience, 
                                self.low_belief_count, 
                                self.performance_thresh
            )
            

            if mic_drift:
                #print(f"Drift here! Record index {record}")

                y_mic = np.array(self.mic_preds)
                y_mach = np.array(mach_predictions)
                y_true = self.df_batch3[self.target].iloc[:len(y_mic)].to_numpy()
                
                self.mic_acc = (y_mic == y_true).mean()
                self.model_acc = (y_mach == y_true).mean()
                

                strat = "Mao" if two_step_deferral else "Confidence"
                dir = os.path.join("MIC_res", f"{self.dataset_name}", f"{self.name}_{strat}")
                
                mic_pref= f"MIC_DRIFT_User_{self.name}_{self.mic_model_name}"

                metrics_gs = {
                            "time_steps": times_GS,
                            "sparsity": sparsities_GS,
                            "proximity": proximities_GS,
                            "plausability":plausabilities_GS,
                            "method": "GS",  
                            "dataset": self.name
                            }
                
                mic_res = {
                    "model.pkl": self.mic_model,
                    "System_Accuracy.txt": self.mic_acc,
                    "Model_Accuracy.txt":self.model_acc,
                    "Model_Confidence.txt.": mach_confidence,
                    "times_GS.txt": times_GS,
                    "GS_metrics.json": metrics_gs,
                    "MiC_FEA.txt": self.fea_mic                    
                    }
                
                save_data(dir, mic_pref, mic_res)

                _, df_log= get_percentage_and_df(df_switch, self.processed, self.target) 

                super().switch_phase(mic_drift, current_phase= 'MiC', current_log= df_log, strat= strat)

                return df_log
           
            
           
        ## - ACCURACY MEASURES
        # misura custom da inserire alla fine come fotografia

        y_mic = np.array(self.mic_preds)
        y_mach = np.array(mach_predictions)
        y_true = self.df_batch3[self.target].iloc[:len(y_mic)].to_numpy()
                
        self.mic_acc = (y_mic == y_true).mean()
        self.model_acc = (y_mach == y_true).mean()
        
        ## - UPDATING SAVE STRUCTURES
        

        strat = "Mao" if two_step_deferral else "Confidence"
        dir = os.path.join("MIC_res", f"{self.dataset_name}", f"{self.name}_{strat}")

        mic_pref= f"User_{self.name}_{self.mic_model_name}"

        metrics_gs = {
                        "time_steps": times_GS,
                        "sparsity": sparsities_GS,
                        "proximity": proximities_GS,
                        "plausability":plausabilities_GS,
                        "method": "GS",  
                        "dataset": self.name
                    }
                
        mic_res = {
                    "model.pkl": self.mic_model,
                    "System_Accuracy.txt": self.mic_acc,
                    "Model_Accuracy.txt":self.model_acc,
                    "Model_Confidence.txt": mach_confidence,
                    "times_GS.txt": times_GS,
                    "GS_metrics.json": metrics_gs,
                    "MiC_FEA.txt": self.fea_mic
                    }
                
        save_data(dir, mic_pref, mic_res)


        _, df_log_final = get_percentage_and_df(df_switch, self.processed, self.target)

        return df_log_final
    








 