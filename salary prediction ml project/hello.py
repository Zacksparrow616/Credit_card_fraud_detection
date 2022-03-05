import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt 

print ('pandas version is {}'.format(pd.__version__))
print ('numpy version is {}'.format(np.__version__))
print ('pickle version is {}'.format(pk.__doc__))
print ('matplotlib version is {}'.format(plt.__doc__))


from flask import Flask,request,render_template,jsonify

app=Flask(__name__)

@app.route('/')

def hello():
	return "hello world ! this is my first flask program...."
	if __name__=='__main__':
		app.run()
		'''app.run(debug=ture)'''
