import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pymongo

def Dialog(Facebook,Whatsapp,Youtube,Telegram,Tiktok,Other):

    A1 = 20.00  # Fun Blaster 217 - Whatsapp Facebook
    P1 = 217.00

    A2F = 8.00
    A2W = 7.00
    A2Y = 20.00
    P2 = 310.00

    A3F = 4.00
    A3W = 4.00
    A3Y = 1.00
    A3A = 19.00  # Anytime data
    P3 = 370.00

    # Details of Package - Dialog 4G Video Blaster 327 Unlimited Package

    A4A = 3.5
    P4 = 327

    # Dialog Unlimited YouTube 100 Package
    P5 = 100

    # Dialog Fun Blaster Prepaid 447 Unlimited Package
    P6 = 447
    A6A = 1

    # Internet Card 30days 239
    P7 = 239
    A7A = 2.62 + 0.666

    # Fun Blaster 297 - Tiktok Instagram
    P8 = 297
    A8A = 15

    # Inputs
    Instagram = 0.0


    if Other > 0:
        flag = -1
    else:
        flag = 1

    # For Fun Blaster 217 - Whatsapp + Facebook deciding
    m = P1 * (1.00 - (Whatsapp + Facebook) / A1)
    if flag == -1 or Youtube or Tiktok or Instagram or Telegram:
        m = -1

    # For Package two deciding
    n1 = P2 * ((A2W - Whatsapp) / A2W)
    n2 = P2 * ((A2F - Facebook) / A2F)
    n3 = P2 * ((A2Y - Youtube) / A2Y)

    n_values = [n1, n2, n3]
    sign = 1 if all(value >= 0 for value in n_values) and flag >= 0 else -1
    n = (sum(abs(value) for value in n_values) * sign) / 4.1

    # For Package three deciding
    q = (P3 * (1.00 - (Whatsapp + Facebook + Youtube + Tiktok + Instagram + Telegram) / (A3A + A3W + A3F + A3Y))) / 1.09

    # For Dialog 4G Video Blaster 327 Unlimited Package
    DialogVideoBlaster = P4 * (1.00 - (Whatsapp + Facebook + Youtube + Tiktok + Instagram + Telegram + Other) / A4A)

    # For Dialog Unlimited YouTube 100 Package
    YouTube100Package = -1 * (Whatsapp + Facebook + Youtube + Tiktok + Instagram + Telegram + Other)
    if YouTube100Package >= 0:
        YouTube100Package += 100

    # For Dialog Fun Blaster Prepaid 447 Unlimited Package
    FunBlaster = P6 * (1.00 - (Whatsapp + Facebook + Youtube + Tiktok + Instagram + Telegram + Other) / A6A)

    # For Internet Card 30days 239
    IC239 = P7 * (1.00 - (Whatsapp + Facebook + Youtube + Tiktok + Instagram + Telegram + Other) / A7A)

    # For Fun Blaster 297 - Tiktok Instagram
    FB297 = P8 * (1.00 - (Tiktok + Instagram) / A8A)
    if flag == -1 or Youtube or Whatsapp or Facebook or Telegram:
        FB297 = -1

    # ALL unlimited Package
    P9 = 1560
    A1560 = 100000000

    print(m)
    # print(n)
    # print(q)
    #print(DialogVideoBlaster)
    #print(YouTube100Package)
    #print(FunBlaster)
    #print(IC239)
    #print(FB297)
    #print(A1560)

    def select_package():
        packages = {
            "Fun Blaster 217 - Whatsapp Facebook": m,
            # "n": n,
            # "q": q,
            "DialogVideoBlaster": DialogVideoBlaster,
            "YouTube100Package": YouTube100Package,
            "Dialog Fun Blaster Prepaid 447": FunBlaster,
            "Internet Card for 30days for Rs 239": IC239,
            "Fun Blaster 297 - Tiktok Instagram": FB297,
            "ALL unlimited Package for 1560": A1560,
        }
        available_packages = [pkg for pkg, value in packages.items() if value >= 0]
        if available_packages:
            return min(available_packages, key=lambda x: packages[x])
        return None

    package = select_package()

    if package:
       ## print (f"Select package {package}")
        return (package)
    else:
        return ("No available package")

##############################################################################################################################################


def Mobitel (Facebook,Whatsapp,Youtube,Telegram,Tiktok,Other):
    A1 = 20.00  # Fun Blaster 217 - Whatsapp Facebook
    P1 = 217.00

    A2F = 8.00
    A2W = 7.00
    A2Y = 20.00
    P2 = 310.00

    A3F = 4.00
    A3W = 4.00
    A3Y = 1.00
    A3A = 19.00  # Anytime data
    P3 = 370.00

    # Details of Package - Dialog 4G Video Blaster 327 Unlimited Package

    A4A = 3.5
    P4 = 327

    # Mobitel Unlimited YouTube Package
    P5 = 360

    # Mobitel Tripple Buddy
    P6 = 447
    A6A = 1

    # Mobitel One Shot Unlimited
    P11 = 989
    A11A = 30

    # Internet Card 30days 239
    P7 = 239
    A7A = 2.62 + 0.666

    # Fun Blaster 297 - Tiktok Instagram
    P8 = 297
    A8A = 15

    # Mobitel Unlimited TikTok Package
    P10 = 385

    # Mobitel Unlimited Messenger Package
    P12 = 70

    # Mobitel Social Networking Unlimited
    P13 = 145

    # Mobitel Nonstop Lokka
    P14 = 520

    Instagram = 0.0


    if Other > 0:
        flag = -1
    else:
        flag = 1

    # For Fun Blaster 217 - Whatsapp + Facebook deciding
    m = P1 * (1.00 - (Whatsapp + Facebook) / A1)
    if flag == -1 or Youtube or Tiktok or Instagram or Telegram:
        m = -1

    # For Package two deciding
    n1 = P2 * ((A2W - Whatsapp) / A2W)
    n2 = P2 * ((A2F - Facebook) / A2F)
    n3 = P2 * ((A2Y - Youtube) / A2Y)

    n_values = [n1, n2, n3]
    sign = 1 if all(value >= 0 for value in n_values) and flag >= 0 else -1
    n = (sum(abs(value) for value in n_values) * sign) / 4.1

    # For Package three deciding
    q = (P3 *
         (1.00 - (Whatsapp + Facebook + Youtube + Tiktok + Instagram + Telegram) /
          (A3A + A3W + A3F + A3Y))) / 1.09

    # For Dialog 4G Video Blaster 327 Unlimited Package
    DialogVideoBlaster = P4 * (1.00 - (Whatsapp + Facebook + Youtube + Tiktok +
                                       Instagram + Telegram + Other) / A4A)

    # For Mobitel Unlimited YouTube 360 Package
    YouTube360Package = -1 * (Whatsapp + Facebook + Youtube + Tiktok + Instagram +
                              Telegram + Other)
    if YouTube360Package >= 0:
        YouTube360Package += 100

    # For Mobitel Unlimited TikTok Package
    Mobitel_Unlimited_TikTok = -1 * (Whatsapp + Facebook + Youtube + Instagram +
                                     Telegram + Other)
    if Mobitel_Unlimited_TikTok >= 0:
        Mobitel_Unlimited_TikTok += 99

    # For Mobitel Unlimited Messaging Package
    Mobitel_Unlimited_Messaging_Package = -1 * (Facebook + Youtube + Tiktok +
                                                Instagram + Telegram + Other)
    if Mobitel_Unlimited_Messaging_Package >= 0:
        Mobitel_Unlimited_Messaging_Package += 98

    # For Mobitel Social Networking Unlimited
    Mobitel_Social_Networking_Unlimited = -1 * (Whatsapp + Youtube + Tiktok +
                                                Telegram + Other)
    if Mobitel_Social_Networking_Unlimited >= 0:
        Mobitel_Social_Networking_Unlimited += 97

    # For Mobitel GINI Nonstop Lokka
    Mobitel_GINI_Nonstop_Lokka = -1 * (Tiktok + Telegram + Other)
    if Mobitel_GINI_Nonstop_Lokka >= 0:
        Mobitel_GINI_Nonstop_Lokka += 97

    # For Mobitel Tripple Buddy
    Mobitel_Tripple_Buddy = P6 * (1.00 - (Whatsapp + Facebook + Youtube + Tiktok +
                                          Instagram + Telegram + Other) / A6A)

    # For Mobitel One shot Unlimted
    One_shot_Unlimted = P11 * (1.00 - (Telegram + Other) / A11A)

    # For Internet Card 30days 239
    IC239 = P7 * (1.00 - (Whatsapp + Facebook + Youtube + Tiktok + Instagram +
                          Telegram + Other) / A7A)

    # For Fun Blaster 297 - Tiktok Instagram
    FB297 = P8 * (1.00 - (Tiktok + Instagram) / A8A)
    if flag == -1 or Youtube or Whatsapp or Facebook or Telegram:
        FB297 = -1

    # ALL unlimited Package
    P9 = 1876
    A1560 = 100000000

    print(m)
    # print(n)
    # print(q)
    # print(DialogVideoBlaster)
    # print(YouTube100Package)
    # print(FunBlaster)
    print(IC239)
    print(FB297)
    print(A1560)

    def select_package():
        packages = {
            # "Fun Blaster 217 - Whatsapp Facebook": m,
            # "n": n,
            # "q": q,
            # "DialogVideoBlaster": DialogVideoBlaster,
            "YouTube360Package": YouTube360Package,
            "Mobitel_Unlimited_TikTok": Mobitel_Unlimited_TikTok,
            "Mobitel Tripple Buddy": Mobitel_Tripple_Buddy,
            "One_shot_Unlimted": One_shot_Unlimted,
            "Mobitel_Unlimited_Messaging_Package": Mobitel_Unlimited_Messaging_Package,
            "Mobitel_Social_Networking_Unlimited": Mobitel_Social_Networking_Unlimited,
            "Mobitel_GINI_Nonstop_Lokka": Mobitel_GINI_Nonstop_Lokka,
            # "Internet Card for 30days for Rs 239":IC239,
            # "Fun Blaster 297 - Tiktok Instagram":FB297,
            "ALL unlimited Package for 1876": A1560,
        }
        available_packages = [pkg for pkg, value in packages.items() if value >= 0]
        if available_packages:
            return min(available_packages, key=lambda x: packages[x])
        return None

    package = select_package()

    if package:
        return (package)
    else:
        return ("No available package")

####################################################################################################################################################################
def Hutch (Facebook,Whatsapp,Youtube,Telegram,Tiktok,Other):
    A1 = 20.00  # Fun Blaster 217 - Whatsapp Facebook
    P1 = 217.00

    A2F = 8.00
    A2W = 7.00
    A2Y = 20.00
    P2 = 310.00

    A3F = 4.00
    A3W = 4.00
    A3Y = 1.00
    A3A = 19.00  # Anytime data
    P3 = 370.00

    # Details of Package - Hutch Nonstop Youtube 315 Unlimited Package

    A4A = 1.3
    P4 = 315

    # Mobitel Unlimited YouTube Package
    P5 = 360

    # Hutch Nonstop whatsapp Facebook with anytime
    P6 = 652
    A6A = 8.2

    # Mobitel One Shot Unlimited
    P11 = 989
    A11A = 30

    # Anytime Pack of 4.7 GB
    P7 = 389
    A7A = 4.7

    #Fun Blaster 297 - Tiktok Instagram
    P8 = 297
    A8A = 15

    # Hutch Unlimited TikTok Package
    P10 = 353

    # Mobitel Unlimited Messenger Package
    P12 = 70

    # Mobitel Social Networking Unlimited
    P13 = 145

    # Hutch All your favorite Apps Non-Stop
    P14 = 479

    # Inputs
    Whatsapp = 0.00  # change these values to tune the code
    Facebook = 0.01
    Youtube = 0.0
    Tiktok = 0.01
    Instagram = 0.0
    Telegram = 0.0
    Other = 0

    if Other > 0:
        flag = -1
    else:
        flag = 1

    # For Fun Blaster 217 - Whatsapp + Facebook deciding
    m = P1 * (1.00 - (Whatsapp + Facebook) / A1)
    if flag == -1 or Youtube or Tiktok or Instagram or Telegram:
        m = -1

    # For Package two deciding
    n1 = P2 * ((A2W - Whatsapp) / A2W)
    n2 = P2 * ((A2F - Facebook) / A2F)
    n3 = P2 * ((A2Y - Youtube) / A2Y)

    n_values = [n1, n2, n3]
    sign = 1 if all(value >= 0 for value in n_values) and flag >= 0 else -1
    n = (sum(abs(value) for value in n_values) * sign) / 4.1

    # For Package three deciding
    q = (P3 *
        (1.00 - (Whatsapp + Facebook + Youtube + Tiktok + Instagram + Telegram) /
        (A3A + A3W + A3F + A3Y))) / 1.09

    # Hutch Nonstop Youtube 315 Unlimited Package
    Hutch_Nonstop_Youtube = P4 * (1.00 - (Whatsapp + Facebook + Youtube + Tiktok + Instagram + Telegram + Other) / A4A)

    # For Hutch Unlimited Facebook Whatsapp Package
    Hutch_Unlimited_Facebook_Whatsapp_Package = -1 * ( Youtube + Tiktok + Instagram +
                            Telegram + Other)
    if Hutch_Unlimited_Facebook_Whatsapp_Package >= 0:
        Hutch_Unlimited_Facebook_Whatsapp_Package = 224

    # For Hutch Unlimited TikTok Package
    Hutch_Unlimited_TikTok = -1 * (Whatsapp + Facebook + Youtube + Instagram +
                                    Telegram + Other)
    if Hutch_Unlimited_TikTok >= 0:
        Hutch_Unlimited_TikTok = 353

    # For Hutch Nonstop Youtube,Whatsapp,Facebook Package
    Hutch_Nonstop_Youtube_Whatsapp_Facebook = -1 * (+Tiktok + Instagram +
                                    Telegram + Other)
    if Hutch_Nonstop_Youtube_Whatsapp_Facebook >= 0:
        Hutch_Nonstop_Youtube_Whatsapp_Facebook = 361

    # For Mobitel Unlimited Messaging Package
    Mobitel_Unlimited_Messaging_Package = -1 * (Facebook + Youtube + Tiktok +
                                                Instagram + Telegram + Other)
    if Mobitel_Unlimited_Messaging_Package >= 0:
        Mobitel_Unlimited_Messaging_Package += 98

    # For Mobitel Social Networking Unlimited
    Mobitel_Social_Networking_Unlimited = -1 * (Whatsapp + Youtube + Tiktok +
                                                Telegram + Other)
    if Mobitel_Social_Networking_Unlimited >= 0:
        Mobitel_Social_Networking_Unlimited += 97

    # For Hutch All your favorite Apps Non-Stop
    favorite_Apps_Non_Stop =  -1 * Other
    if favorite_Apps_Non_Stop >= 0:
        favorite_Apps_Non_Stop += 97

    # For Hutch Nonstop whatsapp Facebook with anytime
    Hutch_Nonstop_whatsapp_Facebook_with_anytime = P6 * (1.00 - (Youtube + Tiktok + Instagram + Telegram + Other) / A6A)

    # For Mobitel One shot Unlimted
    One_shot_Unlimted = P11 * (1.00 - (Telegram + Other) / A11A)

    # For Anytime Pack of 4.7 GB for 30 days
    Anytime_Pack_of_47 = P7 * (1.00 - (Whatsapp + Facebook + Youtube + Tiktok + Instagram + Telegram + Other) / A7A)

    # For Fun Blaster 297 - Tiktok Instagram
    FB297 = P8 * (1.00 - (Tiktok + Instagram) / A8A)
    if flag == -1 or Youtube or Whatsapp or Facebook or Telegram:
        FB297 = -1

    # ALL unlimited Package
    P9 = 1625
    A1560 = 100000000

    print(m)
    #print(n)
    #print(q)
    #print(DialogVideoBlaster)
    #print(YouTube100Package)
    #print(FunBlaster)
    #print(IC239)
    #print(FB297)
    #print(A1560)


    def select_package():
        packages = {
        #"Fun Blaster 217 - Whatsapp Facebook": m,
        #"n": n,
        "Hutch_Nonstop_Youtube_Whatsapp_Facebook for 361": Hutch_Nonstop_Youtube_Whatsapp_Facebook,
        "Hutch_Nonstop_Youtube for 315": Hutch_Nonstop_Youtube,
        #"YouTube360Package": YouTube360Package,
        "Hutch_Unlimited_TikTok": Hutch_Unlimited_TikTok,
        #"Mobitel Tripple Buddy": Mobitel_Tripple_Buddy,
        #"One_shot_Unlimted": One_shot_Unlimted,
        #"Mobitel_Unlimited_Messaging_Package": Mobitel_Unlimited_Messaging_Package,
        #"Mobitel_Social_Networking_Unlimited": Mobitel_Social_Networking_Unlimited,
        "Hutch_Nonstop_whatsapp_Facebook_with_anytime": Hutch_Nonstop_whatsapp_Facebook_with_anytime,
        "Hutch_Unlimited_Facebook_Whatsapp_Package":Hutch_Unlimited_Facebook_Whatsapp_Package,
        "Anytime_Pack_of_4.7_GB":Anytime_Pack_of_47,
        "Hutch_Nonstop_whatsapp_Facebook_with_anytime":Hutch_Nonstop_whatsapp_Facebook_with_anytime,
        #"Fun Blaster 297 - Tiktok Instagram":FB297,
        "Hutch Cliq Non-Stop package for 1625": A1560,
    }
        available_packages = [pkg for pkg, value in packages.items() if value >= 0]
        if available_packages:
            return min(available_packages, key=lambda x: packages[x])
        return None


    package = select_package()

    if package:
        return (package)
    else:
        return ("No available package")

##############################################################################################################################################################################

app = Flask(__name__)
@app.route('/', methods=['POST'])
def receive_data():
    try:
        # Get the JSON data from the request body
        data = request.get_json()
        # Access the number from the received data
        num = data.get('number')
        phone_number1=int(num)
        # Process the received number
        # Replace this code with your logic to handle the number

        client = pymongo.MongoClient("mongodb+srv://test:test@cluster0.nx3yizo.mongodb.net/?retryWrites=true&w=majority")
        db = client['test']
        collection = db['data_usage']

    # load the dataset
        #data1 = request.get_json()  # Get JSON data from Flutter
        #phone_number1 = int(data1['phone_number'])
        #phone_number1 = 776405654
        Number = str(phone_number1)
        query = {'phone_number': phone_number1}
        projection = {'_id': 0, 'phone_number': 0}  # Exclude the _id field from the result

        dataset = pd.DataFrame(list(collection.find(query, projection)))

        # print(dataset)

        # split the dataset into train and test sets
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset.iloc[0:train_size], dataset.iloc[train_size:len(dataset)]

        # normalize the data to a range between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        train = scaler.fit_transform(train)
        test = scaler.transform(test)


        # create sequences of data for training
        def create_sequences(dataset, seq_length):
            X = []
            y = []
            for i in range(len(dataset) - seq_length - 1):
                X.append(dataset[i:(i + seq_length), :])
                y.append(dataset[i + seq_length, 0])
            return np.array(X), np.array(y)


        seq_length = 1  # number of months to use for prediction
        X_train, y_train = create_sequences(train, seq_length)
        X_test, y_test = create_sequences(test, seq_length)

        # create and train the LSTM model
        model = Sequential()
        model.add(LSTM(units=6, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

        # predict the next month's data usage
        last_month_data = np.zeros((1, seq_length, 7))
        last_month_data[0, :, :-1] = test[-1, :-1]
        # print(last_month_data.shape)
        predicted_data = model.predict(last_month_data)
        # reshape predicted_data to match the shape of the original input data
        predicted_data = np.reshape(predicted_data, (1, -1))
        print(predicted_data.shape)
        predicted_data = np.broadcast_to(predicted_data, (1, 7))

        # determine the most suitable data package
        facebook_usage = predicted_data[0, 0]
        youtube_usage = predicted_data[0, 1]
        whatsapp_usage = predicted_data[0, 2]
        telegram_usage = predicted_data[0, 3]
        tiktok_usage = predicted_data[0, 4]
        web_usage = predicted_data[0, 5]
        total_usage = predicted_data[0, 6]

        # Load input data from CSV file
        data = dataset

        # Define input data and output labels
        X = data.iloc[:, 1:7].values
        y = data.iloc[:, 1:].values

        # Normalize input and output data
        X_norm = X / np.max(X)
        y_norm = y / np.max(y)

        # Reshape input data for LSTM layers
        X_norm_re = np.reshape(X_norm, (X_norm.shape[0], 1, X_norm.shape[1]))

        # Define LSTM-based model
        model = Sequential()
        model.add(LSTM(64, input_shape=(1, 6)))
        model.add(Dense(6))

        # Compile and fit the model
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_norm_re, y_norm, epochs=100, batch_size=1, verbose=2)

        # Predict the next month data usage
        prev_month_norm = X_norm[-1]
        prev_month_norm_re = np.reshape(prev_month_norm, (1, 1, 6))
        next_month_norm = model.predict(prev_month_norm_re)
        next_month = next_month_norm.reshape(-1, 6) * np.max(y)
        print("Next month data usage prediction:")
        print("Facebook: " + str(next_month[0][2]))
        print("Whatsapp: " + str(next_month[0][1]))
        print("Youtube: " + str(next_month[0][0]))
        print("Telegram: " + str(next_month[0][3]))
        print("Tiktok: " + str(next_month[0][4]))
        print("Web browsing: " + str(next_month[0][5]))
        print("Total: " + str(np.sum(next_month)))

        if (Number[1] == '7'):
            pakage = Dialog(float(next_month[0][0]),float(next_month[0][1]),float(next_month[0][2]),float(next_month[0][3]),float(next_month[0][4]),float(next_month[0][5]))
        elif((Number[1] == '1' ) or (Number[1] == '0')):
            pakage = Mobitel(float(next_month[0][0]),float(next_month[0][1]),float(next_month[0][2]),float(next_month[0][3]),float(next_month[0][4]),float(next_month[0][5]))
        else:
            pakage = Hutch(float(next_month[0][0]),float(next_month[0][1]),float(next_month[0][2]),float(next_month[0][3]),float(next_month[0][4]),float(next_month[0][5]))

        
        response_data = {
        "pakage": str(pakage),
        "Facebook": float(next_month[0][2]),
        "Whatsapp": float(next_month[0][1]),
        "Youtube": float(next_month[0][0]),
        "Telegram": float(next_month[0][3]),
        "Tiktok": float(next_month[0][4]),
        "Web browsing": float(next_month[0][5]),
        "Total": float(np.sum(next_month))
        }
        
        return jsonify(response_data)

        # else:
        #     print("The most suitable data package is Basic Package")
        #
        # if total_usage > 0.5:
        #     print("The most suitable data package is Social Media Package")
        # elif web_usage > 0.5:
        #     print("The most suitable data package is Web Browsing Package")
        # elif youtube_usage > 0.5:
        #     print("The most suitable data package is Youtube Package")
        # else:
        #     return ("The most suitable data package is Basic Package")


        #response = {'message': 'Number received successfully'}
        #return jsonify(response), 200
    except Exception as e:
        # Handle any errors that occurred during processing
        return jsonify({'error': str(e)}), 500
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)