
# Gold Price Prediction App

A Gold Price Prediction App which utilizes LSTM and ARIMA models to forecast gold prices. The preprocessing and training of models are done using Python libraries such as pandas and matplotlib. The app is deployed using Streamlit, providing an intuitive interface for users to interact with the prediction models. The app provides predictions that users can rely on for making informed decisions related to gold investments or trading


## Screenshots
![gold-price-project](https://github.com/cyrusnguyen/gold-price-pred-app/assets/52537523/93628dcf-20d1-417f-8429-e580f1b20274)



## Features

- Utilizes LSTM and ARIMA models for gold price prediction.
- Preprocessing of data to prepare it for model training.
- Visualization of historical gold prices using matplotlib.
- Intuitive and user-friendly interface deployed using Streamlit.


## Run Locally

Clone the project

```bash
  git clone https://github.com/cyrusnguyen/gold-price-pred-app
```

Go to the project directory

```bash
  cd gold-price-pred-app
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the Streamlit app

```bash
  streamlit run app.py

```

The app should now be running locally, and you can access it through your web browser.





## File Structure

`data/` folder: Folder contains sample datasets

`app.py`: Main script containing the Streamlit application.

`lstm_model.h5`: Compressed LSTM model training and prediction

`arima_model.pkl`: Pickled ARIMA model for training and prediction.

`functions.py`: Python script for data preprocessing and training model.

`requirements.txt`: Text file containing the necessary Python dependencies.


## How to use

- Upon running the app, users will be presented with an interface where they can upload historical gold price data.
- After uploading the data, the app preprocesses it and trains both LSTM and ARIMA models.
- Users can then input a date range for which they want to forecast gold prices.
- The app will generate predictions using both models and display them graphically for easy interpretation.
## Deployment

- The Gold Price Prediction App is deployed using Streamlit, allowing users to access it through a web browser interface. For production deployment, ensure to follow Streamlit's deployment guidelines.

## Issues
If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository.
