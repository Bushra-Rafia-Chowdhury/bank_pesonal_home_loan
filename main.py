from flask import Flask, render_template, request
import pandas as pd
import joblib
import pickle

app = Flask(__name__)


# load the saved model
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        for key, value in request.form.items():
            print(f"{key}: {value}")
        age = request.form['age']
        experience = request.form['experience']
        income = request.form['income']
        zip_code = request.form['zip_code']
        family = request.form['family']
        cc_avg = request.form['cc_avg']
        education = request.form['education']
        mortgage = request.form['mortgage']
        cd_account = request.form['cd_account']
        securities_account = request.form['securities_account']
        online = request.form['online']
        credit_card = request.form['credit_card']

        # User input
        user_input = {'Age': age, 'Experience': experience, 'Income': income, 'ZIP Code': zip_code, 'Family': family,
                      'CCAvg': cc_avg, 'Education': education, 'Mortgage': mortgage, 'Securities Account': securities_account,
                      'CD Account': cd_account, 'Online': online, 'CreditCard': credit_card}

        print('user_input', user_input)
        # Loading the trained model
        model = load_pkl('model.pkl')

        # Creating a DataFrame from user input
        user_df = pd.DataFrame([user_input])
        feature_names = user_input.keys()

        # Load the trained StandardScaler instance
        scaler = load_pkl('scaler.pkl')  # Replace 'scaler.pkl' with the actual file path

        # Scaling user input data
        user_scaled = scaler.transform(user_df)
        user_scaled_df = pd.DataFrame(user_scaled, columns=feature_names)

        # Predicting loan approval
        prediction = model.predict(user_scaled_df)

        result = 'Loan is approved' if prediction[0] == 1 else 'Loan is not approved'
        # result = {'Loan Approval Prediction': prediction[0]}
        return render_template('index.html', result=result)

    return render_template('index.html', result=None)



@app.route('/home')
def home():
    # User input
    user_input = {'Age': 25, 'Experience': 1, 'Income': 49, 'ZIP Code': 91107, 'Family': 4,
                'CCAvg': 1.68, 'Education': 1, 'Mortgage': 0, 'Securities Account': 1,
                'CD Account': 0, 'Online': 0, 'CreditCard': 1}
    # Loading the trained model
    model = load_pkl('model.pkl')

    # Creating a DataFrame from user input
    user_df = pd.DataFrame([user_input])
    feature_names = user_input.keys()

    # Load the trained StandardScaler instance
    scaler = load_pkl('scaler.pkl')  # Replace 'scaler.pkl' with the actual file path

    # Scaling user input data
    user_scaled = scaler.transform(user_df)
    user_scaled_df = pd.DataFrame(user_scaled, columns=feature_names)

    # Predicting loan approval
    prediction = model.predict(user_scaled_df)

    result = 'Loan is approved' if prediction[0] == 1 else 'Loan is not approved'
    return result


if __name__ == '__main__':
    app.run()
