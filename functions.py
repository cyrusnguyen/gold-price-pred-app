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

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
        from tensorflow.keras import backend as K
        from tensorflow.keras.models import load_model
        from sklearn.preprocessing import MinMaxScaler
        import warnings
        warnings.filterwarnings("ignore")

        from contextlib import contextmanager
        from io import StringIO
        from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
        from threading import current_thread
        import streamlit as st
        import sys

    
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
        file_data = st.file_uploader("Upload file csv hoặc json tại đây (phải có cột 'date' và 'price')", type=([".csv", ".json"]))
        # st.markdown("""Nếu không có mẫu, bạn có thể tham khảo data <a href="https://github.com/Thienlong1312/titanic-pred-app/tree/main/data/">tại đây</a>.""", unsafe_allow_html=True,)

        if st.button('Hoặc đơn giản nhấn vào đây!'):
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

        if price_type == "Giá mua":
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
        history = [x for x in train_ar]


        arima_predictions = list()
        for t in range(len(test_ar)):
            # model = ARIMA(history, order=(7,1,0)) 
            # arima_model = model.fit(disp=0)
            output = arima_model.forecast()
            yhat = output[0]
            arima_predictions.append(yhat)
            obs = test_ar[t]
            history.append(obs)

        arima_error = mean_squared_error(test_ar, arima_predictions)
        arima_root_error = np.sqrt(arima_error)
        st.write('Arima Testing Mean Squared Error: %.3f' % arima_error)
        st.write('Arima Testing Root Mean Squared Error: %.3f' % arima_root_error)
        st.write(arima_model.summary())

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
        st.write('Nhận xét:')
        st.write('* Chỉ số mean_squared_error ở Arima thấp hơn LSTM')
        st.write('* Nhưng LSTM có thể dự báo được trend ở thời gian dài')
        st.write('* Model Arima sẽ phù hợp cho dự đoán ngắn hạn (1-3 ngày), còn LSTM sẽ phù hợp hơn ở dài hạn')
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
        list_a=['Bạn có biết: ','Có thể bạn chưa biết: ','Thông tin hữu ích: ','Sự thật thú vị rằng: ']
        list_b=['Theo WGC, tính đến cuối năm 2019 loài người đã khai thác gần 200.000 tấn vàng',
                'Giá trị thị trường hiện tại của tất cả vàng hiện có trên thế giới là 8 nghìn tỷ đô la Mỹ.',
                'Mỗi dặm khối nước biển chứa trung bình 25 tấn vàng.',
                'Gần 50% tổng số vàng được khai thác từ Witwatersrand ở Nam Phi.',
                'Tổng lượng vàng được khai thác trong lịch sử loài người hiện được ghi lại là khoảng 161,000 tấn.',
                'Thời gian dự đoán càng lâu thì độ chính xác của model càng giảm!!',
                'Các model dự đoán phải có cùng tần suất dữ liệu.',
                'Từ “vàng” (gold) bắt nguồn từ từ “geolu” trong tiếng Anh cổ, có nghĩa là màu vàng.',
                'Nhà đầu tư thường mua vàng để đầu tư dài hạn.',
                'Vàng đã tăng giá trị khoảng 500% trong vòng 15 năm qua.',
                'Theo WGC, lượng vàng chưa khai thác trên thế giới ước tính chỉ còn khoảng 54.000 tấn.',
                'Gần như không thể phá hủy nên vàng thường được nấu chảy, tinh chế và tái sử dụng.',
                '46 tấn vàng được sử dụng hàng năm để trám răng.',
                'Vàng đã tăng giá trị khoảng 500% trong vòng 15 năm qua.']
        return list_a[int(np.random.randint(len(list_a)))]+list_b[int(np.random.randint(len(list_b)))]

    

    def download_csv(df):
        """
        Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href =f'Ok! Nhấn<a href="data:file/csv;base64,{b64}"\
            download="labeled_gold_price.csv"   > vào đây </a>để tải file csv xuống'
        st.markdown(href, unsafe_allow_html=True)
        return
    ############################
