## DO NOT MODIFY

from abc import abstractmethod, ABC #ABC è l'equivalente della classe object per la classi astratte, una classe astratta sarà figlia di ABC, così che possa utilizzare il decoratore abstractmethod
class Regressor(ABC):
    def __init__(self, **params):
        self.params = params 

    @abstractmethod # questo decoratore fa sì che il metodo sia puramente astratto, questo implica che il tentativo di instanziare un oggetto di tipo Regressor risulterà in un TypeError
    def fit(self, X, y):
        '''
        Addestra il modello di Machine Learning sulla base dei parametri forniti (differenti modelli avranno differenti parametri ed un differente utilizzo)
        aggiornando questi ultimi al fine di minimizzare l'errore sui dati forniti.

        Parametri
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Il dato su cui addestrare il modello.
        y: {array-like, sparse matrix} of shape (n_samples, 1)
            Le label su cui addestrare il modello.

        '''
        pass

    @abstractmethod
    def predict(self, X):
        '''
        Effettua una previsione utilizzando i parametri attualmente salvati sul modello

        Parametri
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Il dato su cui addestrare il modello.

        '''
        pass

    def fit_predict(self, X, y):
        '''
        Addestra il modello di Machine Learning sulla base dei parametri forniti (differenti modelli avranno differenti parametri ed un differente utilizzo)
        aggiornando questi ultimi al fine di minimizzare l'errore sui dati forniti.
        Dopodiché effettua una previsione utilizzando i dati forniti in input con i parametri appena computati.

        Parametri
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Il dato su cui addestrare il modello.
        y: {array-like, sparse matrix} of shape (n_samples, 1)
            Le label su cui addestrare il modello.

        '''
        self.fit(X, y)
        return self.predict(X)

    def evaluate(self, X, y, eval_obj):
        '''
        Valuta la performance previsionale del modello su un dataset.

        Parametri
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Il dato da utilizzare per fini previsionali.
        y: {array-like, sparse matrix} of shape (n_samples, 1)
            Le label reali.
        metric: {object Evaluator}
            Un oggetto di tipo Evaluator dovrà calcolare la performance del modello a partire da valori predetti su quelli reali.

        Output
        -----------
        scores: {dict-like, dict}
            Dizionario (o oggetto di tipo dizionario) contenente l'output della call all'oggetto di tipo metrica. Una metrica può contenere un valore puntuale,
            la deviazione standard dell'errore, etc.
        '''
        from aim_machine_learning.metrics import Evaluator
        y_pred = self.predict(X)

        if isinstance(eval_obj, Evaluator):
            scores =  eval_obj(y, y_pred)
        else:
            raise TypeError('Expected Metric-like object.')

        return scores

        
        
        



