from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('data/random_forest_model.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Base numeric inputs
            displ = float(request.form['displ'])
            cylinders = float(request.form['cylinders'])

            # Category values (match what the model saw)
            drive = request.form['drive']
            trany = request.form['trany']
            fuelType = request.form['fuelType']
            VClass = request.form['VClass']

            # Load the model's expected feature names
            input_dict = {'displ': displ, 'cylinders': cylinders}
            dummy_input = pd.DataFrame(columns=model.feature_names_in_)
            dummy_input.loc[0] = 0  # Initialize with all zeros

            # Fill in the actual values
            dummy_input.at[0, 'displ'] = displ
            dummy_input.at[0, 'cylinders'] = cylinders

            # One-hot category flags (only if column exists)
            for cat_feature in [
                f'drive_{drive}',
                f'trany_{trany}',
                f'fuelType_{fuelType}',
                f'VClass_{VClass}'
            ]:
                if cat_feature in dummy_input.columns:
                    dummy_input.at[0, cat_feature] = 1

            # Predict
            prediction = round(model.predict(dummy_input)[0], 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
