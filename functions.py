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
            arima_results = np.round(arima_model.forecast(steps=5)[0],1)

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

        # for i in range(len(arima_results)):
        #     lstm_results[i] = arima_results[i]
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
    
    def gioi_tinh():
        #Giới tính: Nam, Nữ --> sex_male
        gender=st.radio('- *Giới tính người đó:',['Nam','Nữ'])
        if gender=='Nữ':
            SEX_MALE=0
        else:
            SEX_MALE=1
        return gender,SEX_MALE

    def tuoi():
        #Tuổi --> 0-80
        age=st.text_input('- *Nhập tuổi:')
        AGE=1
        if age!='':
            if age.isdigit():
                if int(age)>=1 and int(age)<=100:
                    st.success('Ok!')
                    AGE=int(age)
                else:
                    st.warning('Tuổi nằm trong khoảng từ 1 đến 100')
            else:
                st.warning('Chỉ nhập số!')
        else:
            st.write('Chỉ nhập số')
        return AGE

    def sibsp():
        #Có vợ/chồng đi cùng không
        #Có người thân đi cùng không
        st.write('- *Đánh dấu tích nếu có')
        VO_CHONG=st.checkbox('Có vợ/chồng đi cùng')
        nguoi_than=st.checkbox('Có anh chị em ruột đi cùng')
        if nguoi_than==True:
            NGUOI_THAN=st.slider('*Mấy người đi cùng:',min_value=0,max_value=8,step=1)
        else:
            NGUOI_THAN=0
        return VO_CHONG,NGUOI_THAN

    def parch():
        #Bố mẹ có đi cùng không
        #Có dẫn trẻ em đi cùng không
        BO=st.checkbox('Có bố đi cùng')
        ME=st.checkbox('Có mẹ đi cùng')
        tre_em=st.checkbox('Có dẫn trẻ em đi cùng')
        if tre_em==True:
            TRE_EM=st.slider('*Mấy trẻ em đi cùng:',min_value=0,max_value=8,step=1)
        else:
            TRE_EM=0
        return BO,ME,TRE_EM

    def pclass():
        #Mua vé thuộc hạng nào
        ve=st.radio('- *Mua vé thuộc hạng nào:',['1st','2nd','3rd'])
        if ve=='1st':
            PCLASS=1
        elif ve=='2nd':
            PCLASS=2
        else:
            PCLASS=3
        return ve,PCLASS

    def fare():
        #Giá vé bao nhiêu (Tính theo tiền đô) --> 0-512
        FARE=st.slider('- *Giá vé bao nhiêu:',min_value=0,max_value=512,step=1)
        return FARE

    def Embarked():
        #Lên tàu ở cảng nào
            #Cảng Southampton --> [0,1]
            #Cảng Queenstown  --> [1,0]
            #Cảng Cherbourg   --> [0,0]
        embarked=st.selectbox('- *Lên tàu ở cảng nào:',
        ['Cảng Queenstown','Cảng Cherbourg','Cảng Southampton'])
        if embarked=='Cảng Queenstown':
            EMBARKED_Q=1
            EMBARKED_S=0
        elif embarked=='Cảng Cherbourg':
            EMBARKED_Q=0
            EMBARKED_S=0
        else:
            EMBARKED_Q=0
            EMBARKED_S=1
        return embarked,EMBARKED_Q,EMBARKED_S

    def Name():
        # Họ tên nếu có
        name=st.text_input('- Họ tên (Nếu có):')
        return name

    def Quan_he():
        #Mối quan hệ với người dùng
        quan_he=st.radio('- Người đó có quan hệ gì với bạn',
        ['Là chính bạn','Người quen/bạn bè...','Không có quan hệ gì'])
        return quan_he

    def nhap_thong_tin():
        gender,SEX_MALE=gioi_tinh()
        AGE=tuoi()

        VO_CHONG,NGUOI_THAN=sibsp()
        #--> Cộng tổng lại xem có bao nhiêu người --> sibsp (0-8)
        SIBSP=VO_CHONG+NGUOI_THAN

        BO,ME,TRE_EM=parch()
        #--> Cộng tổng lại xem có bao nhiêu người --> parch (0-9)
        PARCH=BO+ME+TRE_EM

        ve,PCLASS=pclass()
        FARE=fare()
        embarked,EMBARKED_Q,EMBARKED_S=Embarked()

        #Tên
        name=Name()
        #Mối quan hệ
        quan_he=Quan_he()
        return gender,SEX_MALE,AGE,VO_CHONG,NGUOI_THAN,\
            SIBSP,BO,ME,TRE_EM,PARCH,ve,PCLASS,FARE,embarked,\
            EMBARKED_Q,EMBARKED_S,name,quan_he
    ############################

    ############################
    def thong_tin_huu_ich():
        list_a=['Bạn có biết: ','Có thể bạn chưa biết: ','Thông tin hữu ích: ','Sự thật thú vị rằng: ']
        list_b=['Trên chuyến tàu Titanic, chỉ có khoảng 20% nam giới sống sót và nữ lên tới 75%!',
                'Giới tính quyết định đến 25% trong việc dự đoán của model',
                'Tuổi quyết định đến tận 32% trong việc dự đoán và chiếm vị trí Thuộc tính quan trọng nhất!!',
                'Giá vé chiếm tận 25% độ quan trọng khi dự đoán!!',
                'Chỉ có khoảng 6% hành khách mua vé với giá trên 100$ và 70% trong số họ sống sót!',
                'Nếu bạn là nam và bạn chịu mua vé với giá trên 100$, bạn có tới tận 32% cơ hội sống sót!!!',
                'Thảm họa Titanic đã cướp đi sinh mạng gần 2/3 số hành khách trên tàu chỉ 1/3 số hành khách sống sót',
                'Đa số hành khách mua vé với giá dưới 100$, và chỉ 35% trong số họ sống sót',
                'Nếu bạn là nam và bạn mua vé rẻ (dưới 100$), xin chia buồn, bạn chỉ có khoảng 18% cơ hội sống sót :(',
                'Khoảng 60% trẻ em dưới 10 tuổi và khoảng 45% người già trong độ tuổi 50-60 được cứu sống',
                'Độ tuổi từ 20 đến 45 tuổi là bấp bênh nhất, chỉ có 35% cơ hội sống sót',
                'Nếu bạn là nam và tuổi của bạn trong khoảng 20 đến 40, bạn sẽ phải 1 chọi 6 nếu muốn có cơ hội sống sót :(',
                'Nhóm nữ có cơ hội sống sót thấp nhất theo phân tích là nhóm nữ có độ tuổi 25-30, mua vé dưới 100$ với 63% cơ hội sống sót',]
        return list_a[int(np.random.randint(len(list_a)))]+list_b[int(np.random.randint(len(list_b)))]

    def speak_sorry(name,gender,quan_he):
        list_sorry_1=['Rất tiếc! ','Xin chia buồn! ','Tin buồn! ']
        list_sorry_2=[' không thể sống sót sau vụ Titanic :(',' sẽ hi sinh anh dũng :(',' sẽ ra đi :(']
        if quan_he=='Là chính bạn':
            speak=list_sorry_1[int(np.random.randint(len(list_sorry_1)))]+'Bạn'+list_sorry_2[int(np.random.randint(len(list_sorry_2)))]
        else:
            if name=='':
                if quan_he=='Người quen':
                    if gender=='Nam':
                        list_name=['Người bạn quen biết','Họ','Anh ấy']
                    else:
                        list_name=['Người bạn quen biết','Họ','Cô ấy']
                    speak=list_sorry_1[int(np.random.randint(len(list_sorry_1)))]+list_name[int(np.random.randint(len(list_name)))]+list_sorry_2[int(np.random.randint(len(list_sorry_2)))]
                else:
                    list_sorry_2=[' không thể sống sót sau vụ Titanic :(',' rất khó để sống sót :(']
                    if gender=='Nam':
                        list_name=['Người này','Hành khách này','Họ','Anh ấy']
                    else:
                        list_name=['Người này','Hành khách này','Họ','Cô ấy']
                    speak=list_sorry_1[int(np.random.randint(len(list_sorry_1)))]+list_name[int(np.random.randint(len(list_name)))]+list_sorry_2[int(np.random.randint(len(list_sorry_2)))]
            else:
                speak=list_sorry_1[int(np.random.randint(len(list_sorry_1)))]+name.capitalize()+list_sorry_2[int(np.random.randint(len(list_sorry_2)))]

        return speak

    def speak_good(name,gender,quan_he):
        list_good_1=['Rất may mắn! ','Tuyệt vời! ','Tin tốt! ']
        list_good_2=[' sẽ sống sót sau vụ Titanic!',' sẽ sống sót trở về!',' sẽ còn nguyên vẹn trở về sau chuyến đi!']
        if quan_he=='Là chính bạn':
            speak=list_good_1[int(np.random.randint(len(list_good_1)))]+'Bạn'+list_good_2[int(np.random.randint(len(list_good_2)))]
        else:
            if name=='':
                if quan_he=='Người quen':
                    if gender=='Nam':
                        list_name=['Người bạn quen biết','Họ','Anh ấy']
                    else:
                        list_name=['Người bạn quen biết','Họ','Cô ấy']
                    speak=list_good_1[int(np.random.randint(len(list_good_1)))]+list_name[int(np.random.randint(len(list_name)))]+list_good_2[int(np.random.randint(len(list_good_2)))]
                else:
                    list_good_2=[' có khả năng sống sót cao trong vụ Titanic!',' có thể sống sót trở về!']
                    if gender=='Nam':
                        list_name=['Người này','Hành khách này','Họ','Anh ấy']
                    else:
                        list_name=['Người này','Hành khách này','Họ','Cô ấy']
                    speak=list_good_1[int(np.random.randint(len(list_good_1)))]+list_name[int(np.random.randint(len(list_name)))]+list_good_2[int(np.random.randint(len(list_good_2)))]
            else:
                speak=list_good_1[int(np.random.randint(len(list_good_1)))]+name.capitalize()+list_good_2[int(np.random.randint(len(list_good_2)))]
        return speak

    def thong_bao(model,gender,SEX_MALE,AGE,VO_CHONG,NGUOI_THAN,
                SIBSP,BO,ME,TRE_EM,PARCH,ve,PCLASS,FARE,embarked,
                EMBARKED_Q,EMBARKED_S,name,quan_he):
        result=model.predict([[PCLASS,SEX_MALE,AGE,SIBSP,PARCH,FARE,EMBARKED_Q,EMBARKED_S]])
        if quan_he=='Là chính bạn':
            st.subheader('Bạn đã chọn:')
            st.write('- Giới tính của bạn:',gender)
            st.write('- Tuổi của bạn:',AGE,'tuổi')
            if VO_CHONG==0 and NGUOI_THAN==0 and BO==0 and ME==0 and TRE_EM==0:
                st.write('- Bạn đi một mình')
            else:
                if VO_CHONG==1:
                    st.write('- Có vợ/chồng đi cùng')
                else:
                    pass
                if NGUOI_THAN!=0:
                    st.write('- Có',NGUOI_THAN,'anh chị em ruột đi cùng')
                else:
                    pass
                if TRE_EM!=0:
                    st.write('- Có dẫn theo',TRE_EM,'trẻ em đi cùng')
                else:
                    pass
                if BO==1 and ME==1:
                    st.write('- Bạn đi cùng bố mẹ')
                else:
                    if BO==1:
                        st.write('- Bạn đi cùng bố')
                    elif ME==1:
                        st.write('- Bạn đi cùng mẹ')
                    else:
                        pass
            st.write('- Mua vé hạng',ve,'với giá tiền',FARE,'đô và lên tàu ở',embarked)
            st.write('...')
            if result==0:
                st.write('-->',speak_sorry(name,gender,quan_he))
            else:
                st.write('-->',speak_good(name,gender,quan_he))
        else:
            st.subheader('Người đó có những đặc điểm sau:')
            st.write('- Giới tính:',gender)
            st.write('- Tuổi:',AGE,'tuổi')
            if name=='':
                if VO_CHONG==0 and NGUOI_THAN==0 and BO==0 and ME==0 and TRE_EM==0:
                    st.write('- Người đó đi một mình')
                else:
                    if VO_CHONG==1:
                        st.write('- Có vợ/chồng đi cùng')
                    else:
                        pass
                    if NGUOI_THAN!=0:
                        st.write('- Có',NGUOI_THAN,'anh chị em ruột đi cùng')
                    else:
                        pass
                    if TRE_EM!=0:
                        st.write('- Có dẫn theo',TRE_EM,'trẻ em đi cùng')
                    else:
                        pass
                    if BO==1 and ME==1:
                        st.write('- Người đó đi cùng bố mẹ')
                    else:
                        if BO==1:
                            st.write('- Người đó đi cùng bố')
                        elif ME==1:
                            st.write('- Người đó đi cùng mẹ')
                        else:
                            pass
                st.write('- Mua vé hạng',ve,'với giá tiền',FARE,'đô và lên tàu ở',embarked)
                st.write('...')
            else:
                if VO_CHONG==0 and NGUOI_THAN==0 and BO==0 and ME==0 and TRE_EM==0:
                    st.write('-',name.capitalize(),'đi một mình')
                else:
                    if VO_CHONG==1:
                        st.write('- Có vợ/chồng đi cùng')
                    else:
                        pass
                    if NGUOI_THAN!=0:
                        st.write('- Có',NGUOI_THAN,'anh chị em ruột đi cùng')
                    else:
                        pass
                    if TRE_EM!=0:
                        st.write('- Có dẫn theo',TRE_EM,'trẻ em đi cùng')
                    else:
                        pass
                    if BO==1 and ME==1:
                        st.write('-',name.capitalize(),'đi cùng bố mẹ')
                    else:
                        if BO==1:
                            st.write('-',name.capitalize(),'đi cùng bố')
                        elif ME==1:
                            st.write('-',name.capitalize(),'đi cùng mẹ')
                        else:
                            pass
                st.write('-',name.capitalize(),'mua vé hạng',ve,'với giá tiền',FARE,'đô và lên tàu ở',embarked)
                st.write('...')
            if result==0:
                st.write('-->',speak_sorry(name,gender,quan_he))
            else:
                st.write('-->',speak_good(name,gender,quan_he))
        return
    ############################

    ############################
    def xu_ly_du_lieu_2(X):
        X_pre=X.interpolate()
        X_pre=X_pre.drop(['cabin','home.dest'],axis=1)
        X_pre=X_pre.dropna()
        X_pre=pd.get_dummies(X_pre,columns=['sex','embarked'],drop_first=True)
        X_pre=X_pre[['pclass','sex_male','age','sibsp','parch','fare','embarked_Q','embarked_S']]
        return X_pre

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