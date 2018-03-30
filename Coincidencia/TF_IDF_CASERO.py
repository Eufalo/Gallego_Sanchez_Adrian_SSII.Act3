import os
import numpy
import sys
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import wordpunct_tokenize
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.stem import SnowballStemmer 
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QMessageBox,QTableWidgetItem,QWidget,QHeaderView
from PyQt5.QtGui import QImage, QPalette, QBrush
from PyQt5 import uic
from PyQt5 import QtCore
# -*- coding: utf-8 -*-
#Classe preprocesado 
class Preprocedo(BaseEstimator, TransformerMixin):
    
    
    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
       
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = self.stopwor()
        self.steam = SnowballStemmer("spanish")
        
        
    def fit(self, X, y=None):
        
        return self

    def stopwor(self):
        with open('.'+os.path.sep+'utiliti'+os.path.sep+'stopWordsSpanish.txt','r') as stop_words: 
            lineas = [linea.strip() for linea in stop_words]
            
        return lineas
    def transform(self, X):
        
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        vectortokens=[ ]
        # Rompemos el documento en lineas
        for sent in sent_tokenize(document):
            # Rompemos la line en tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Aplicamos cada trasformacion a cada token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token
                #print("TOKEEEN     ",token)
                
            
                    # Eliminamos las palabras que aparezcan en nuestra lista de parada 
                if token not in self.stopwords: 
                        #Below code will remove all punctuation marks as well as non alphabetic characters
                    if  token.isalpha():
                            # Lematizamos cada token para quedarnos con la raiz 
                        stemm = self.steam.stem(token)
                            #print("LEEMMAAA     ",lemma)
                        vectortokens.append(stemm);
        return vectortokens
#cargar contenido del fichero 
def load_files (dirname0, description=None,
               load_content=True, shuffle=False, encoding='CP1252',
               decode_error='ignore', random_state=0):
        
    #Cargamos los datos de la funcion obtener_path
    filenames =  os.listdir(dirname0)
    
    #Convertimos los datos a un array Numpy
    filenames = np.array(filenames)
  
    #Carga los archivos gracias al path que obtiene de filename
    if load_content:
        data = []
        for filename in filenames:
            
            #rb = lectura y escritura
            with open(dirname0+os.path.sep+filename, 'rb') as f:
                data.append(f.read())
        if encoding is not None:
            data = np.array([d.decode('CP1252', 'ignore') for d in data])
            
        #Gracias a los datos creamos un Bunch que es un 'diccionario'
        return data
       
    return data
    
def tf(token, doc):
    return doc.count(token)/ len(doc)

def n_containing(token, list_documents):
    return sum(1 for d in list_documents if token in list_documents)

def idf(token, list_documents):
    
    return math.log(len(list_documents) / (1 + n_containing(token, list_documents)))

def tfidf(token,doc, list_documents):
    
    return tf(token,doc) * idf(token, list_documents)

def scorePalabras(lista)   :
    scores =[]
    palabras=[]
    
    for  documento in lista:
        #puede que no haya que ordenarlos ya que estamos buscando que tengan las mismas caracteristicas
        documento=sorted(documento)
        for word in documento:
            scores.append(tfidf(word, documento, lista))
            palabras.append(word)
            
    return palabras,scores    
#Distancia coseno entrada->2 arrays salida->coincidencia(distancia coseno)    
def distancia_coseno(u, v):
   return numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v)))        
def coindicencias(documento,query):
    
    #Realizamos la matriz con las veces que se ha repetido la palabra en el documento
    vectorizer=CountVectorizer(stop_words=Preprocedo().stopwor())
    transformer = TfidfTransformer()
  
    #Vectorizamos los documentos segun las palabras que aparecen en la query
    documentoVectorize = vectorizer.fit_transform(documento).toarray()
    #Vectorizamos la query preprocesada por las stopwords 
    queryVectorize = vectorizer.transform(query).toarray()
    #Una vez vectorizados los documentos realizamos el tfidf 
    transformer.fit(documentoVectorize)
    documentotdif = transformer.transform(documentoVectorize).toarray()
    #Una vez vectorizada la query realizamos tfidf
    transformer.fit(queryVectorize)
    querytfidf=transformer.transform(queryVectorize).toarray()
    
    aux=[]
    #Realizamos la distancia al cose para cada documento
    for i,d in enumerate(documento) :
        aux.append(distancia_coseno(documentotdif[i],querytfidf[0]))
    return aux

class Ventana_Principal(QMainWindow):
 #Método constructor de la clase
 def __init__(self):
  #Iniciar el objeto QMainWindow
  QMainWindow.__init__(self)
  #Cargar la configuración del archivo .ui en el objeto
  
  uic.loadUi("Ventana_Bus_Coincidencias.ui", self)
  self.Button_Buscar.clicked.connect(self.clasificar)
 def clasificar(self):
     query=self.Text_Buscador.text()
     documento=load_files('.'+os.path.sep+'Documentos')
     coinciden=coindicencias(documento,[query])
     self.con_tabla(documento,coinciden)
 def contenido_tabla(self,documento,coincidencia):
    tabla=[]
    for e,i in enumerate(documento):
        noty=[]
        noty.append(str(e))
        noty.append(str(documento[e]))
        noty.append(str(coincidencia[e]))
        
        tabla.append(noty)
    
    return tabla

 def con_tabla(self,documento,coincidencia):
  #añadirle las cabeceras
  header = ["Numero","Documento","Coincidencia"]
  data=self.contenido_tabla(documento,coincidencia)
  self.tabla_Coincidencia.setColumnCount(3)
  self.tabla_Coincidencia.setHorizontalHeaderLabels(header)
  self.tabla_Coincidencia.setRowCount(len(data))
  r=0

  for i in data:
    c=0  
    for e in i:
        item=QTableWidgetItem(e)
        item.setFlags( QtCore.Qt.ItemIsSelectable |  QtCore.Qt.ItemIsEnabled )
        self.tabla_Coincidencia.setItem(r,c, item)
        c=c+1
    r=r+1
  head = self.tabla_Coincidencia.horizontalHeader()
  head.setSectionResizeMode(QHeaderView.Stretch)
  head.setStretchLastSection(True)
  self.repaint
#Instancia para iniciar una aplicación
app = QApplication(sys.argv)
#Crear un objeto de la clase
_ventana = Ventana_Principal()
#Mostra la ventana
_ventana.show()
#Ejecutar la aplicación
app.exec_()  
