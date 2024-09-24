


from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model and label encoders
with open('loan_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from form
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        credit_history = float(request.form['credit_history'])

        # Manual prediction conditions
        if applicant_income > 2000 and coapplicant_income > 1000 and loan_amount < 200 and credit_history == 0:
            prediction = '   Congratulations !! Your Loan has been approved..  '
        else:
            prediction = '  Oops !! Your Loan has been rejected  '

        return jsonify({'result': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
