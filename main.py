from flask import Flask, render_template, request

from model import check_is_covid
app = Flask(__name__)

# Load the trained model and label encoder


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_covid', methods=['POST'])
def check_covid():
    selected_symptoms = request.form.getlist('symptoms')
    d={}
    for i in selected_symptoms:
        d[i]=1
  
    check=check_is_covid(d)
    return render_template('result.html', has_covid=check)

if __name__ == '__main__':
    app.run(debug=True)
