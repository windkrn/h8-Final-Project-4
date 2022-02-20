from re import split
import flask
import numpy as np
import pickle


model = pickle.load(open("model/modelfp4.pkl", "rb"))
app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return(flask.render_template('main.html'))

if __name__ == '__main__':
    app.run()

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = {0: 'cluster 0', 1: 'cluster 1', 2: 'cluster 2', 3: 'cluster 3' }

    return flask.render_template('main.html', prediction_text='Pengguna kredit termasuk {} '.format(output[prediction[0]]))

if __name__ == "__main__":
    app.run(debug=True)