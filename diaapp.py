from flask import Flask,redirect,url_for,render_template,request
import numpy as np
import os
import pickle
app=Flask(__name__)
pa1=pickle.load(open('dia.pkl','rb'))
"""
@app.route('/')
def sf():
        return"om namo venkatesya"
"""


picFolder=os.path.join('static','images')
app.config['UPLOAD_FOLDER']=picFolder


@app.route('/')

def home():
        pic1=os.path.join(app.config['UPLOAD_FOLDER'],'dia2.png')
        return render_template('diahome.html',pic1_1=pic1)
"""
@app.route('/diapredict')

def diapredict():
        return render_template('diapd.html')"""

@app.route('/prd')
def prd():
        return render_template('diapd.html')

@app.route('/submit',methods=['post'])

def submit():
        a=[x for x in request.form.values()]
        fa=[np.array(a)]
        crrprdt=pa1.predict(fa)
        if crrprdt==[0]:
                return render_template('diapd.html',optext1='you donot have diabetes')
        else:
                return render_template('diapd.html',optext1='you have diabetes')
"""
        #op1=round(crrprdt[0],2)
        return render_template('diapd.html',optext1='you have{}'.format(crrprdt))
"""




if __name__=='__main__':
        app.run()
