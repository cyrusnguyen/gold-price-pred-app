if True:
    if True: #import libraries
        import streamlit as st

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import csv
        import json
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score,confusion_matrix,\
        classification_report,roc_auc_score,roc_curve

        import base64
        from statsmodels.tsa.arima_model import ARIMA
        from statsmodels.tsa.arima_model import ARIMAResults 
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error
        import os.path
        from os import path

        from tensorflow.keras.models import load_model
        from sklearn.preprocessing import MinMaxScaler
        import warnings
        warnings.filterwarnings("ignore")


    
    def load_model_arima():
        arima_model = ARIMAResults.load("arima_model.pkl")
        return arima_model

    def load_model_lstm():
        # lstm_model = Sequential()
        # lstm_model.add(LSTM(units = 120, return_sequences = True))
        # lstm_model.add(Dropout(0.2))
        # lstm_model.add(LSTM(units = 120, return_sequences = True))
        # lstm_model.add(Dropout(0.2))
        # lstm_model.add(LSTM(units = 120, return_sequences = True))
        # lstm_model.add(Dropout(0.2))
        # lstm_model.add(LSTM(units = 120))
        # lstm_model.add(Dropout(0.2))

        # lstm_model.add(Dense(units = 1))

        # lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        # lstm_model.load_weights("mymodel_15.h5")
        lstm_model = load_model("lstm_model.h5")
        return lstm_model

    
    def read_data(file):
        try:
            data = pd.read_json(file)
            return data
        except ValueError:
            try:
                data = pd.read_csv(file)
                return data
            except ValueError: return 'The file is not json nor csv'


    def upload_file(file_data_path='./data/gold_price.json'):
        file_data = st.file_uploader("Upload file csv ho???c json t???i ????y (ph???i c?? c???t 'date' v?? 'price')", type=([".csv", ".json"]))
        st.markdown("""N???u kh??ng c?? m???u, b???n c?? th??? tham kh???o data <a href="https://github.com/cyrusnguyen/gold-price-pred-app/tree/main/data">t???i ????y</a>.""", unsafe_allow_html=True,)

        if st.button('Ho???c ????n gi???n nh???n v??o ????y!'):
            file_data=file_data_path
        
        return file_data
        
    def xu_ly_du_lieu(data):
        try:
            data=data[["date", "price"]]
        except KeyError:
            data.rename(columns={data.columns[0]: "date"}, inplace = True)
        try:
            data.price.astype('int')
            data = data.drop_duplicates(subset='date')
            # Set date as index
            data.set_index('date', inplace=True)
            data.sort_index(inplace=True)
            data.index = pd.to_datetime(data.index)
            data = fill_missing_dates(data)
            return data, 1

        except ValueError:
            # Split "/" from col price, first element is buy_price, next element is sell_price and replace "," by ""
            data['buy_price'] = data.price.str.split('/').apply(lambda x: [element.strip() for element in x]).str[0].str.replace(',','')
            data['sell_price'] = data.price.str.split('/').apply(lambda x: [element.strip() for element in x]).str[1].str.replace(',','')

            # Convert column price to 'int'
            data['buy_price'] = data['buy_price'].astype('int')
            data['sell_price'] = data['sell_price'].astype('int')
            data = data[['date', 'buy_price', 'sell_price']]
            data = data.drop_duplicates(subset='date')

            # Set date as index
            data.set_index('date', inplace=True)
            data.sort_index(inplace=True)
            data.index = pd.to_datetime(data.index)
            
            return data, 2
            
    def fill_missing_dates(data):
        # Fill missing dates by NaN values, then fillna by nearest neighbor
        idx = pd.date_range(data.index[0], data.index[-1])
        data.index = pd.DatetimeIndex(data.index)
        data = data.reindex(idx, fill_value=np.nan)
        data = data.apply(lambda x: x.interpolate('nearest').bfill().ffill().astype('int'))
        return data

    def xu_ly_price(data, price_type):

        if price_type == "Gi?? mua":
            data = data[['buy_price']]
        else: 
            data = data[['sell_price']]

        data = data.rename(columns={data.columns[0]: 'price'})
        data = fill_missing_dates(data)

        return data
    
    def chuan_doan_du_lieu(data, arima_model, lstm_model):
        # Arima Model
        st.header("ARIMA")
        train_data, test_data = data[0:int(len(data)*0.9)], data[int(len(data)*0.9):]
        train_ar = train_data.price.values
        test_ar = test_data.price.values
        arima_predictions = list()

        for t in range(len(test_ar)):
            # model = ARIMA(history, order=(7,1,0)) 
            # arima_model = model.fit(disp=0)
            output = arima_model.forecast()
            yhat = output[0]
            arima_predictions.append(yhat)

        arima_error = mean_squared_error(test_ar, arima_predictions)
        arima_root_error = np.sqrt(arima_error)
        st.write('Arima Testing Mean Squared Error: %.3f' % arima_error)
        st.write('Arima Testing Root Mean Squared Error: %.3f' % arima_root_error)
        st.write(pd.read_html(arima_model.summary().tables[1].as_html(), header=0, index_col=0)[0])

        # LSTM Model
        st.header("LSTM")
        training_set = train_data.values
        testing_set = test_data.values

        sc = MinMaxScaler(feature_range = (0, 1)) # scale
        training_set_scaled = sc.fit_transform(training_set)
        lstm_model = load_model("lstm_model.h5")

        look_back = 60
        dataset_total = pd.concat((train_data, test_data), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(test_data) - look_back:].values
        inputs = inputs.reshape(-1,1)
        inputs = sc.transform(inputs)

        X_test = []
        no_of_sample = len(inputs)

        for i in range(look_back, no_of_sample):
            X_test.append(inputs[i-look_back:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))\


        predicted_price = lstm_model.predict(X_test)
        lstm_predictions = sc.inverse_transform(predicted_price)


        lstm_error = mean_squared_error(testing_set, lstm_predictions)
        lstm_root_error = np.sqrt(lstm_error)

        st.write('LSTM Testing Mean Squared Error: %.3f' % lstm_error)
        st.write('LSTM Testing Root Mean Squared Error: %.3f' % lstm_root_error)

        st.write(lstm_model.summary())

        return train_data, test_data, arima_predictions, arima_error, lstm_predictions, lstm_error
    
    
    


    def ve_bieu_do(train_data, test_data, arima_predictions, arima_error, lstm_predictions, lstm_error):
        figure = plt.figure(figsize=(12,20))

        figure.add_subplot(3, 1, 1)
        plt.title('Gold Prices Entire Dataset', size = 20)
        plt.xlabel('Date')
        plt.ylabel('Prices')
        plt.plot(train_data, 'blue', label='Training Data')
        plt.plot(test_data, 'greenyellow', label='Testing Data')
        plt.legend()


        figure.add_subplot(3, 1, 2)
        plt.plot(train_data.index, train_data.values, color='blue', label='Training Data')
        plt.plot(test_data.index, test_data.values, color='greenyellow', marker='o', markersize=4, linestyle='dashed', 
                label='Actual Price')
        plt.plot(test_data.index, arima_predictions, color='red', label='Predicted Price')
        plt.title('Gold Prices Prediction using ARIMA', size = 20)
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.legend()


        figure.add_subplot(3, 1, 3)
        plt.plot(train_data.index, train_data.values, color='blue', label='Training Data')
        plt.plot(test_data.index, test_data.values, color='greenyellow', marker='o', markersize=4, linestyle='dashed', 
                label='Actual Price')
        plt.plot(test_data.index, lstm_predictions, color='red', label='Predicted Price')
        plt.title('Gold Prices Prediction using LSTM', size = 20)
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.legend()

        st.write(figure)
        st.write('Nh???n x??t:')
        st.write('* Ch??? s??? mean_squared_error ??? Arima th???p h??n LSTM')
        st.write('* Nh??ng LSTM c?? th??? d??? b??o ???????c trend ??? th???i gian d??i')
        st.write('* Model Arima s??? ph?? h???p cho d??? ??o??n ng???n h???n (1-3 ng??y), c??n LSTM s??? ph?? h???p h??n ??? d??i h???n')
        return
    ############################

    ############################
    def future_prediction(data, arima_model, lstm_model, num_predictions):
        if num_predictions < 5:
            #ARIMA
            arima_results = list()
            for t in range(num_predictions):
                output = arima_model.forecast()
                yhat = output[0]
                arima_results.append(yhat)
        else:
            #ARIMA
            arima_results = list()
            arima_results = np.round(arima_model.forecast(steps=3)[0],1)

            #LSTM
            lstm_results = list()
            train_data, test_data = data[0:int(len(data)*0.9)], data[int(len(data)*0.9):]
            
            sc = MinMaxScaler(feature_range = (0, 1))
            training_set_scaled = sc.fit_transform(train_data.values)
            dataset_test = test_data

            # Predict future values
            dataset_test = dataset_test[len(dataset_test)-60:len(dataset_test)].to_numpy()
            dataset_test = np.array(dataset_test)

            inputs = dataset_test
            inputs = inputs.reshape(-1,1)
            inputs = sc.fit_transform(inputs)

            i = 0
            while i<num_predictions:   
                X_test = []
                no_of_sample = len(dataset_test)

                # Lay du lieu cuoi cung
                
                X_test.append(inputs[no_of_sample - 60:no_of_sample, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                # Du doan gia
                predicted_gold_price = lstm_model.predict(X_test)

                # chuyen gia tu khoang (0,1) thanh gia that
                predicted_gold_price = sc.inverse_transform(predicted_gold_price)

                # Them ngay hien tai vao
                dataset_test = np.append(dataset_test, predicted_gold_price, axis=0)
                # print(dataset_test[len(dataset_test) - 2:len(dataset_test), 0])
                inputs = dataset_test
                inputs = inputs.reshape(-1, 1)
                inputs = sc.transform(inputs)
                predicted_price = predicted_gold_price[0][0]
                lstm_results.append(predicted_price)

                i = i +1

        for i in range(len(arima_results)):
            lstm_results[i] = arima_results[i]
        predictions = lstm_results
        return predictions
    
    def plot_future_results(data, results):
        title_string = str("Predicted gold prices of {0} days".format(len(results)))

        figure = plt.figure(figsize=(15,10))
        date_index = pd.date_range(data.index.values[-1], periods=len(results)).tolist()
        figure.add_subplot(1, 1, 1)
        plt.plot(date_index, results, color='blue', label='Predicted Data')
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.title(title_string, size=20)

        st.write(figure)
        return

    def plot_comparison(actual, predicted):
        figure = plt.figure(figsize=(15,10))
        if len(actual.values) < len(predicted):
            predicted = predicted[:len(actual.values)]
        else: 
            actual = actual.iloc[0:len(predicted),:]
        figure.add_subplot(1, 1, 1)         
        plt.plot(actual.index, actual.values, color='blue', label='Actual Price')
        plt.plot(actual.index, predicted, color='red', label='Predicted Price')

        plt.xticks(rotation=45)
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.title("Compare actual and predicted values", size=24)
        plt.legend()

        st.write(figure)
        return actual, predicted
    
    
    ############################
    def thong_tin_huu_ich():
        list_a=['B???n c?? bi???t: ','C?? th??? b???n ch??a bi???t: ','Th??ng tin h???u ??ch: ','S??? th???t th?? v??? r???ng: ']
        list_b=['Theo WGC, t??nh ?????n cu???i n??m 2019 lo??i ng?????i ???? khai th??c g???n 200.000 t???n v??ng',
                'Gi?? tr??? th??? tr?????ng hi???n t???i c???a t???t c??? v??ng hi???n c?? tr??n th??? gi???i l?? 8 ngh??n t??? ???? la M???.',
                'M???i d???m kh???i n?????c bi???n ch???a trung b??nh 25 t???n v??ng.',
                'G???n 50% t???ng s??? v??ng ???????c khai th??c t??? Witwatersrand ??? Nam Phi.',
                'T???ng l?????ng v??ng ???????c khai th??c trong l???ch s??? lo??i ng?????i hi???n ???????c ghi l???i l?? kho???ng 161,000 t???n.',
                'Th???i gian d??? ??o??n c??ng l??u th?? ????? ch??nh x??c c???a model c??ng gi???m!!',
                'C??c model d??? ??o??n ph???i c?? c??ng t???n su???t d??? li???u.',
                'T??? ???v??ng??? (gold) b???t ngu???n t??? t??? ???geolu??? trong ti???ng Anh c???, c?? ngh??a l?? m??u v??ng.',
                'Nh?? ?????u t?? th?????ng mua v??ng ????? ?????u t?? d??i h???n.',
                'V??ng ???? t??ng gi?? tr??? kho???ng 500% trong v??ng 15 n??m qua.',
                'Theo WGC, l?????ng v??ng ch??a khai th??c tr??n th??? gi???i ?????c t??nh ch??? c??n kho???ng 54.000 t???n.',
                'G???n nh?? kh??ng th??? ph?? h???y n??n v??ng th?????ng ???????c n???u ch???y, tinh ch??? v?? t??i s??? d???ng.',
                '46 t???n v??ng ???????c s??? d???ng h??ng n??m ????? tr??m r??ng.',
                'V??ng ???? t??ng gi?? tr??? kho???ng 500% trong v??ng 15 n??m qua.']
        return list_a[int(np.random.randint(len(list_a)))]+list_b[int(np.random.randint(len(list_b)))]

    

    def download_csv(df):
        """
        Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href =f'Ok! Nh???n<a href="data:file/csv;base64,{b64}"\
            download="labeled_gold_price.csv"   > v??o ????y </a>????? t???i file csv xu???ng'
        st.markdown(href, unsafe_allow_html=True)
        return
    ############################
