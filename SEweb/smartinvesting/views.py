from django.shortcuts import render
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas_datareader as pdr
import datetime
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,self.width_, axis=1)
# Create your views here.
def index(req):

    return render(req,'indexT.html',{})
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def cal(req):
    out=dict()
    if req.method == 'POST':
        data = req.POST
        ss = data['Sday']
        se = data['Lday']
        s =  datetime.datetime.strptime(ss, '%Y-%m-%d')
        e =  datetime.datetime.strptime(se, '%Y-%m-%d' )
        h = data['set50'] + '.BK'
        poly_model = make_pipeline(GaussianFeatures(18), LinearRegression())
        pttgc = pdr.get_data_yahoo(h.upper(), start = datetime.datetime(1999, 9, 5 ), end = datetime.date.today())
        kk = pttgc[['Adj Close']]
        kk = kk.values
        yt = kk.reshape(len(kk))
        xt = pttgc.index
        xt = xt.map(datetime.datetime.toordinal)
        my = xt.values
        xt = np.reshape(my, (len(kk), 1))
        poly_model.fit(xt,yt)
        table = []
        s = s.toordinal()
        e = e.toordinal()
        # xf = [[i] for i in range(s,e+1)]
        table = [[datetime.date.fromordinal(i).strftime('%A %d-%m-%Y'),'-' if len(pttgc[pttgc.index.date==datetime.date.fromordinal(i)]['Adj Close'].values)==0 else round(pttgc[pttgc.index.date==datetime.date.fromordinal(i)]['Adj Close'].values[0],3) ,round(poly_model.predict(np.asarray([[i]]))[0],3)]  for i in range(s,e+1)]
        out['result'] = table
        out['set'] = data['set50'].upper()
        tb = np.asarray(table)
        yp =[]
        yr =[]
        for i in tb:
            date = []
            sd = i[0].split()[1]
            date += sd.split('-')
            yp+=['{ x: new Date('+date[2]+','+str(int(date[1])-1)+','+date[0]+'), y:'+str(i[2])+'},']
            if i[1]!='-':
                yr+=['{ x: new Date('+date[2]+','+str(int(date[1])-1)+','+date[0]+'), y:'+str(i[1])+'},']
        out['yp']=yp
        out['yr']=yr

        return render(req,'cal.html',out)

    else :
        return render(req,'cal.html',out)
# def compare(req):
#     return render(req,'compare.html',{})
