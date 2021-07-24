if True:
    if True: #import libraries
        import streamlit as st
        from datetime import timedelta
        
        import functions as ft
        import SessionState
        
        import numpy as np
        import pandas as pd
        
        import pickle as pk
        from PIL import Image
        
    ############################

    ############################
    def main():
        st.title('Time Series - Dự đoán giá vàng')

        #Load model
        arima_model = ft.load_model_arima()
        lstm_model = ft.load_model_lstm()
        
        menu=['I. Load và xử lý dữ liệu', 'II. Kiểm tra model','III. Dự đoán giá vàng','IV. So sánh kết quả dự đoán']
        choice=st.sidebar.selectbox('Menu',menu)
        st.subheader(choice)
        ss = SessionState.get(data=None)
        future_state = SessionState.get(future_prediction=0)
        
        if choice==menu[0]: # Kiểm tra model
            #Load dữ liệu
            st.subheader('1. Load dữ liệu có chứa label')

            file_data = ft.upload_file()

            if file_data:
                data = ft.read_data(file_data)
                st.write(data.head())
                if 'price' not in data.columns:
                    st.write('Dữ liệu không có nhãn, vui lòng chọn dữ liệu khác')
                else:
                    #Xử lý dữ liệu
                    st.subheader('2. Xử lý dữ liệu')
                    data, num_of_price_type = ft.xu_ly_du_lieu(data)
                    
                    if num_of_price_type == 1:
                        st.write("Dữ liệu đã xử lý:")
                        st.write(data.head())
                    elif num_of_price_type == 2:
                        # Chọn sẵn giá mua nếu file_data là str chứ không phải object
                        if isinstance(file_data, str):
                            data = ft.xu_ly_price(data, "Giá mua")
                            st.write("Dữ liệu đã xử lý:")
                            st.write(data.head())
                        # Cho phép người dùng chọn giá mua hoặc giá bán
                        else:
                            # Nếu dữ liệu có cả giá mua và giá bán, cho user chọn
                            if len(data.columns) > 1:
                                price_type = st.radio('Dự đoán giá mua hay giá bán?', ['Giá mua','Giá bán'])
                                data = ft.xu_ly_price(data, price_type)
                                st.write("Dữ liệu đã xử lý:")
                                st.write(data.head())
                            # Nếu không đủ thì lấy mặc định cột giá
                            else: 
                                data = data.rename(columns={data.columns[0]: 'price'})
                                data = ft.fill_missing_dates(data)
                                st.write("Dữ liệu đã xử lý:")
                                st.write(data.head())

                    else: print("Dữ liệu xử lý bị lỗi")
                    ss.data = data
                
                    # Download dữ liệu
                    st.subheader('3. Tải dữ liệu đã xử lý')
                    ft.download_csv(ss.data)

            else:
                image = Image.open('image/gold.jpg')
                st.image(image, caption=None,use_column_width=True,width=None)
        elif choice==menu[1]:
            try:
                if ss.data is not None:
                    

                    #Chuẩn đoán dữ liệu
                    st.subheader('1. Chuẩn đoán dữ liệu')
                    train_data, test_data, arima_predictions, arima_error, lstm_predictions, lstm_error = ft.chuan_doan_du_lieu(ss.data,arima_model, lstm_model)




                    #Vẽ biểu đồ
                    st.subheader('2. Vẽ biểu đồ')
                    ft.ve_bieu_do(train_data, test_data, arima_predictions, arima_error, lstm_predictions, lstm_error)
                else:
                    image = Image.open('image/gold.jpg')
                    st.image(image, caption=None,use_column_width=True,width=None)
                    st.header("Vui lòng LOAD DỮ LIỆU (I.) trước khi dự đoán")
            except AttributeError:
                st.header("Vui lòng LOAD DỮ LIỆU (I.) trước khi dự đoán")
                image = Image.open('image/gold.jpg')
                st.image(image, caption=None,use_column_width=True,width=None)
                
        elif choice==menu[2]: #'Dự đoán giá vàng'
            try:
                if ss.data is not None:
                    st.subheader('1. Dự đoán giá trong khoảng thời gian 90 ngày')
                    num_predictions = st.slider('Lựa chọn số ngày cần dự đoán', min_value=1, max_value=90)
                    
                    if st.button('Dự đoán'):
                        
                        results = ft.future_prediction(ss.data, arima_model, lstm_model, num_predictions)
                        future_state.future_prediction = results
                        ft.plot_future_results(ss.data, future_state.future_prediction)

                        st.subheader("2. Tải dữ liệu đã so sánh:")
                        new_data = ss.data.copy()
                        idx = pd.date_range(ss.data.index[-1]+timedelta(days=1), periods=len(future_state.future_prediction))
                        temp_data = pd.DataFrame(future_state.future_prediction, columns=['price'], index=idx)
                        new_data = new_data.append(temp_data)
                        st.write(new_data.head())
                        ft.download_csv(new_data)

                        
                    else:
                        image = Image.open('image/timeseries.jpg')
                        st.image(image, caption=None,use_column_width=True,width=None)
                        
                else:
                    st.header("Vui lòng LOAD DỮ LIỆU (I.) trước khi dự đoán")
                    image = Image.open('image/timeseries.jpg')
                    st.image(image, caption=None,use_column_width=True,width=None)
            except AttributeError:
                st.header("Vui lòng LOAD DỮ LIỆU (I.) trước khi dự đoán")
                image = Image.open('image/timeseries.jpg')
                st.image(image, caption=None,use_column_width=True,width=None)
                
        else:
            try:
                if future_state.future_prediction is not None:
                    st.subheader('1. Load dữ liệu')
                    file_data = ft.upload_file('./data/gold_price_future.json')
                    if file_data:
                        data=ft.read_data(file_data)
                        
                        if 'price' not in data.columns:
                            st.write('Dữ liệu không có nhãn, vui lòng chọn dữ liệu khác')
                        else:
                            #Xử lý dữ liệu
                            st.subheader('2. Xử lý dữ liệu')
                            data, num_of_price_type = ft.xu_ly_du_lieu(data)
                            
                            if num_of_price_type == 1:
                                st.write("Dữ liệu đã xử lý:")
                                st.write(data.head())
                            elif num_of_price_type == 2:
                                data = ft.xu_ly_price(data, "Giá mua")
                                st.write("Dữ liệu đã xử lý:")
                                st.write(data.head())
                            else: print("Dữ liệu xử lý bị lỗi")
                            st.subheader('3. So sánh giá dự đoán và giá thực')  
                            actual, predicted = ft.plot_comparison(data, future_state.future_prediction)   
                            st.subheader("4. Tải dữ liệu đã so sánh:")
                            new_data = actual.copy()
                            new_data['predicted'] = predicted
                            st.write(new_data.head())
                            ft.download_csv(new_data)
                    else:
                        image = Image.open('image/timeseries.jpg')
                        st.image(image, caption=None,use_column_width=True,width=None)

                else:
                    st.header("Vui lòng DỰ ĐOÁN (III.) trước khi so sánh kết quả")
                    image = Image.open('image/gold.jpg')
                    st.image(image, caption=None,use_column_width=True,width=None)
            except AttributeError:
                st.header("Vui lòng DỰ ĐOÁN (III.) trước khi so sánh kết quả")
                image = Image.open('image/gold.jpg')
                st.image(image, caption=None,use_column_width=True,width=None)
        
    if __name__=='__main__':
        main()





































































