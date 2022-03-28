# DO NOT MODIFY

import sys
import os
sys.path.append(os.getcwd())

import numpy as np # si consiglia vivamente di utilizzare i np.array al posto delle liste
import pandas as pd # si può estrarre un numpy array da un oggetto DataFrame con .values
import argparse
import warnings
warnings.filterwarnings('ignore')

from aim_machine_learning.base_regressor import Regressor

import dataset_gen
# questo script genera dei dataset nella cartella data/

f= open("output/logs.txt","w+")
f.close()
f = open("output/logs.txt", "a") # salveremo tutto l'output in un file .txt

# argparse contiene un oggetto di tipo ArgumentParser che ci permetterà di inserire argomenti da linea di codice
# esempio: python scr/main.py "<PATH_TO_DATASET>"
# i path disponibilis sono data/dataset1.csv, data/dataset2.csv, data/dataset3.csv

parser = argparse.ArgumentParser(description='Input parameters.')
parser.add_argument('data_path', type=str, help='path to csv file containing data.')
args = parser.parse_args()
globals().update(vars(args)) # tutti gli argomenti di input diventano variabili globali

# Regressor è una classe base, astratta, che colleziona placeholders per tutti i metodi più importanti che un regressore di Machine Learning dovrebbe contenere.
# Grazie all'utilizzo del modulo nativo 'abc' possiamo definirla come classe astratta e ricevere un errore qualora volessimo provare ad istanziare un oggetto 
# di quel tipo:
try:
    r = Regressor() 
except TypeError:
    print('Regressor e\' una classe astratta.')

# Lo scopo del progetto è implementare dei regressori, figli della classe astratta Regressor con molteplici funzionalità.
# Implementeremo anche una classe Evaluator che utilizzeremo per valutare gli output di un modello.
from aim_machine_learning.metrics import Evaluator

eval_obj = Evaluator(supported_metrics=['mse','mae','corr']) # useremo mean squared error, mean absolute error e correlazione di pearson.
eval_obj.set_metric(new_metric='mse')
print(eval_obj, file=f)
#esempio
y_true = np.array([1,2,3,4,5])
y_pred = np.array([5,4,3,2,1])
print(eval_obj(y_true=y_true,y_pred=y_pred), file=f)
try:
    eval_obj.set_metric('abc')
except NameError:
    print('abc non e\' una metrica supportata',file=f)
print(eval_obj.set_metric('corr')(y_true,y_pred), file=f)

# leggiamo un dataset usando pandas, dal percorso specificato in input allo script
data = pd.read_csv(data_path, index_col=0)
X = data.loc[:,~data.columns.isin(['y'])].values
y = data['y'].values
print(data.head().round(2), file=f)

# adesso creiamo una classe figlia di regressor che implementi una semplice regressione (PiecewiseRegression) che funziona così:
#      il modello, dato un nuovo x, cercherà nel dato storico su cui è stato addestrato con .fit() il dato x_hat più vicino (in termini di distanza euclidea)
#      predirrà x con il valore y_hat corrispondente a x_hat
# questo regressore è un esempio di modello nonparametrico.
from aim_machine_learning.neighbor_regressor import NeighborRegressor

pw_model = NeighborRegressor()
if not issubclass(NeighborRegressor,Regressor):
        raise NameError('Deve essere figlio di regressor.')
pw_model.fit(X, y)
y_pred = pw_model.predict(X)
print(pw_model.evaluate(X, y, eval_obj.set_metric('mse')), file=f)
print(pw_model.evaluate(X, y, eval_obj.set_metric('mae')), file=f)
print(pw_model.evaluate(X, y, eval_obj.set_metric('corr')), file=f)

# adesso rendiamo questo modello dipendente da un singolo parametro k: invece che predirre utilizzando soltanto il punto più vicino ad x (sempre in termini di
# distanza euclidea), utilizzeremo la media degli y dei k punti più vicini ad x. In letteratura questo è noto come k-Neighbors regressor.

pw_model = NeighborRegressor(k=4)
pw_model.fit(X, y)
y_pred = pw_model.predict(X)
print(pw_model.evaluate(X, y, eval_obj.set_metric('mse')), file=f)
print(pw_model.evaluate(X, y, eval_obj.set_metric('mae')), file=f)
print(pw_model.evaluate(X, y, eval_obj.set_metric('corr')), file=f)

# FATE IN MODO CHE IL CODICE SOPRA (LA PRIMA VERSIONE DI NeighborRegressor, giri ancora nello stesso modo senza dover modificare il codice).

# Perché il primo predittore (k=1) performa meglio di quello più generale con k=4? Questo fenomeno si chiama overfitting. Stiamo infatti valutando le performances
# del nostro algoritmo sullo stesso dataset su cui lo abbiamo "addestrato".
# Per valutare le performances di un algoritmo in modo pulito ricorreremo al train-test split. Ovvero divideremo i dati in due insiemi: il primo, il train set,
# sarà quello che l'algoritmo utilizzerà per addestrarsi. Il secondo invece sarà quello su cui valuteremo le predizioni, il test set, che l'algoritmo non avrà mai 
# "visto" prima di quel momento.

# L'obiettivo è creare una classe ModelEvaluator che si occupi a tutto tondo della valutazione di un modello su un dataset.
# Essa riceverà una classe di modello (e.g. NeighborsRegressor), i parametri e un dataset. Dovrà restituirci una stima fair delle sue performances usando 
# train-test split
from aim_machine_learning.model_evaluator import ModelEvaluator

eval_obj = Evaluator(supported_metrics=['mse','mae','corr']) 
eval_obj.set_metric('mse')
# il valutatore effettuerà un train-test split prendendo il primo 20% del dataset come samples di test, mentre il restante 80% come train. Questo si applica
# sia ad X che ad y. Dopodiché addestrerà il modello su X_train,y_train e utilizzerà una metrica di valutazione fornita esternamente per restituire il dizionario
# con le performance.
full_eval = ModelEvaluator(model_class=NeighborRegressor, params={'k':1}, X=X, y=y)
print(full_eval.train_test_split_eval(eval_obj=eval_obj,test_proportion=0.2), file=f)

full_eval = ModelEvaluator(model_class=NeighborRegressor, params={'k':4}, X=X, y=y)
print(full_eval.train_test_split_eval(eval_obj=eval_obj,test_proportion=0.2), file=f)

# Quale predittore performa meglio adesso?
# vogliamo che il nostro ModelEvaluator supporti altri tipi di split del dataset, in particolare la k-fold cross validation.
# La filosofia della k-fold cross validation è quella di dividere il dataset in k parti di egual dimensione, quindi addestrare il modello k volte su k-1 parti e
# di valutarlo sul test che sarà composto dalla k-esima parte su cui non è stato addestrato. Ripetendo k volte questa procedura si otterranno k stime differenti
# dell'errore, si restituisca quindi la media di queste stime come score.
full_eval = ModelEvaluator(model_class=NeighborRegressor, params={'k':1}, X=X, y=y)
print(full_eval.kfold_cv_eval(eval_obj=eval_obj,K=5), file=f)

full_eval = ModelEvaluator(model_class=NeighborRegressor, params={'k':4}, X=X, y=y)
print(full_eval.kfold_cv_eval(eval_obj=eval_obj,K=5), file=f)
# Questa stima è ancora più stabile della precedente.
eval_obj = Evaluator(supported_metrics=['mse','mae','corr']) 
eval_obj.set_metric('corr')

full_eval = ModelEvaluator(model_class=NeighborRegressor, params={'k':1}, X=X, y=y)
print(full_eval.kfold_cv_eval(eval_obj=eval_obj,K=5), file=f)

full_eval = ModelEvaluator(model_class=NeighborRegressor, params={'k':4}, X=X, y=y)
print(full_eval.kfold_cv_eval(eval_obj=eval_obj,K=5), file=f)

# Adesso siamo interessati a scoprire il valore ottimale (in termini di errore minimo) del parametro k del neighbors regressor.
# Per fare ciò, scriviamo una classe ParametersTuner che si occupi, data una classe di modello, un criterio di validazione (e.g. train-test split o k-fold cv) e dei dati
# di stimare il miglior set di parametri di quel modello.
# Utilizzeremo il miglior upper bound del MSE, ovvero il modello con la minore somma di MSE medio + deviazione standard del MSE.
from aim_machine_learning.parameter_tuning import ParametersTuner

eval_obj = Evaluator(supported_metrics=['mse','mae','corr']) 
eval_obj.set_metric('mse')

tuner = ParametersTuner(model_class=NeighborRegressor, X=X, y=y, supported_eval_types=['ttsplit','kfold'])
print(tuner.tune_parameters({'k':[1,2,3,4,5]}, eval_type='ttsplit', eval_obj=eval_obj, **{'K':5, 'test_proportion':0.2}), file=f)

# Utilizzando la libreria matplotlib, si possono plottare liste di numeri
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1,2,3,4,5,6],[1,4,6,8,10,6])
plt.title('A simple plot.')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('output/simple_plot.png') #abbiamo la possibilità di salvare i plot

# utilizzate matplotlib per salvare in output/<name>.png un plot dell'andamento dell'errore al variare di k all'interno della chiamata a tune_parameters
tuner = ParametersTuner(model_class=NeighborRegressor, X=X, y=y, supported_eval_types=['ttsplit','kfold'], output_path='output/')
print(tuner.tune_parameters({'k':np.arange(1,80)}, eval_type='ttsplit', eval_obj=eval_obj, fig_name='ttsplit.png', **{'K':5, 'test_proportion':0.2}),file=f)
best_params_kfold = tuner.tune_parameters({'k':np.arange(1,80)}, eval_type='kfold', eval_obj=eval_obj, fig_name='kfold.png', **{'K':5, 'test_proportion':0.2})
print(best_params_kfold, file=f)

# La nota libreria per il Machine Learning scikit-learn implementa K-nearest neighbors regressor in modo nativo, con la stessa interfaccia del nostro:
from sklearn.neighbors import KNeighborsRegressor
skl_model = KNeighborsRegressor(n_neighbors=best_params_kfold['k'])
my_model = NeighborRegressor(**best_params_kfold)
skl_model.fit(X, y)
my_model.fit(X, y)
# daranno le stesse previsioni a parità di parametri?
print(np.all(my_model.predict(X)==skl_model.predict(X)),file=f)
# vorremmo utilizzare la nostra classe ParametersTuner sull'implementazione di sklearn di KNeighbors.
try:
    tuner = ParametersTuner(model_class=KNeighborsRegressor, X=X, y=y, supported_eval_types=['kfold'], output_path='output/')
    best_params_skl = tuner.tune_parameters({'n_neighbors':np.arange(1,80)}, eval_type='kfold', eval_obj=eval_obj, **{'K':5, 'test_proportion':0.2})
except AttributeError:
    print('KNeighborsRegressor di Scikit-Learn non espone alcun metodo "evaluate"!')

from aim_machine_learning.neighbor_regressor import MySklearnNeighborRegressor
print(issubclass(MySklearnNeighborRegressor, KNeighborsRegressor), file=f)
tuner = ParametersTuner(model_class=MySklearnNeighborRegressor, X=X, y=y, supported_eval_types=['kfold'], output_path='output/')
best_params_skl = tuner.tune_parameters({'n_neighbors':np.arange(1,80)}, eval_type='kfold', eval_obj=eval_obj, **{'K':5, 'test_proportion':0.2})
# se il nostro modello NeighborRegressor è implementato correttamente, oltre all'output anche il parametro ottimale coinciderà con quell di sklearn
print(best_params_skl['n_neighbors']==best_params_kfold['k'], file=f)

# FATE IN MODO CHE FUNZIONI ANCHE LA CHIAMATA PRECEDENTE
# Quale tipo di validazione vi sembra dare un output più stabile?

# DA QUI IN POI SOLO CON DATASET3 E TEST CASE NASCOSTO DATASET5
if data_path in ['data/dataset3.csv','data/dataset5.csv'] :   
    # Utilizzando il dataset3 notiamo come questo regressore sia inadatto a questa task.
    # Creiamo un nuovo regressore UNIVARIATO chiamato MultipleRegressor che cerchi l'intero a tale che, moltiplicato per X, approssimi al meglio y al netto di una costante.
    # (In sostanza una regressione lineare univariata -> y = b+a*X)
    # Invece che utilizzare i minimi quadrati come al solito, sfrutteremo ParametersTuner per addestrare il modello.
    # Infatti MultipleRegressor sarà anch'esso un 'lazy' predictor e noi useremo a e b come veri e propri iperparametri, tunandoli noi.
    from aim_machine_learning.multiple_regressor import MultipleRegressor

    m_model = MultipleRegressor(a=10,b=20)
    if not issubclass(MultipleRegressor,Regressor):
        raise NameError('Deve essere figlio di regressor.')

    X_ = np.reshape(X[:,0], (X.shape[0],1)) #prendiamo solo la prima colonna di variabili
    m_model.fit(X_, y) 
    y_pred = m_model.predict(X_)
    print(m_model.evaluate(X_, y, eval_obj.set_metric('corr')), file=f)


    eval_obj.set_metric('mse')
    tuner = ParametersTuner(model_class=MultipleRegressor, X=X_, y=y, supported_eval_types=['ttsplit','kfold'], output_path='output/')
    print(tuner.tune_parameters({'a':np.linspace(0,5,11),'b':np.linspace(0,5,11)}, eval_type='ttsplit', eval_obj=eval_obj, **{'K':5, 'test_proportion':0.2}),file=f)

    best_params = tuner.tune_parameters({'a':np.linspace(0,5,11),'b':np.linspace(0,5,11)}, eval_type='kfold', eval_obj=eval_obj, **{'K':5, 'test_proportion':0.2})

    # Proviamo a fare una regressione multivariata sommando i modelli
    X1 = np.reshape(X[:,1], (X.shape[0],1)) #prima covariata
    X2 = np.reshape(X[:,1], (X.shape[0],1)) #seconda covariata
    tuner = ParametersTuner(model_class=MultipleRegressor, X=X1, y=y, supported_eval_types=['ttsplit','kfold'], output_path='output/')
    best_params1 = tuner.tune_parameters({'a':np.linspace(0,5,11),'b':[0]}, eval_type='kfold', eval_obj=eval_obj,  fig_name='kfold_mult_regressor',**{'K':5, 'test_proportion':0.2})
    # notiamo come si comporta il parametro a rispetto all'errore (fig. kfold_mult_regressor.png)
    tuner = ParametersTuner(model_class=MultipleRegressor, X=X2, y=y, supported_eval_types=['ttsplit','kfold'], output_path='output/')
    best_params2 = tuner.tune_parameters({'a':np.linspace(0,5,11),'b':np.linspace(0,5,11)}, eval_type='kfold',eval_obj=eval_obj, **{'K':5, 'test_proportion':0.2})
    print(best_params1, file=f)
    print(best_params2, file=f)
    # teniamo il bias in uno solo dei due
    # adesso vorremmo sommare le predizioni, come?
    multi_m_model = MultipleRegressor(**best_params1)+MultipleRegressor(**best_params2)
    # qual è un modo logico di sommare i due modelli?
    print(multi_m_model.a,multi_m_model.b,file=f)
    y_pred = multi_m_model.predict(X)
    print(multi_m_model.evaluate(X, y, eval_obj.set_metric('mse')), file=f)

    # ma non sarebbe meglio stimare i pesi a1 e a2 simultaneamente?
    multi_m_model = MultipleRegressor(a=[-2,2],b=0)
    print(multi_m_model.a, multi_m_model.b, file=f)
    y_pred = multi_m_model.predict(X)
    print(multi_m_model.evaluate(X, y, eval_obj.set_metric('mse')), file=f)

    tuner = ParametersTuner(model_class=MultipleRegressor, X=X, y=y, supported_eval_types=['kfold'], output_path='output/')
    best_params = tuner.tune_parameters({'a':[[i,j] for i in np.linspace(-1,2,11) for j in np.linspace(-1,2,11)],
                                            'b':np.linspace(0,5,11)}, eval_type='kfold', eval_obj=eval_obj, **{'K':5, 'test_proportion':0.2})
    print(best_params, file=f)
    multi_m_model = MultipleRegressor(**best_params)
    print(multi_m_model.evaluate(X, y, eval_obj.set_metric('mse')), file=f)

f.close()
# END
from src.output_compare import compare_outputs
if compare_outputs('output/logs.txt','output/logs_'+data_path.split('/')[-1].split('.')[0]+'.txt'):
    print('Il tuo output e\' corretto.')
else:
    print('Il tuo output non e\' corretto.')