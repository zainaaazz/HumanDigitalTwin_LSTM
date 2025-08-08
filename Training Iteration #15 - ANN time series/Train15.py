#cd C:\Users\S_CSIS-PostGrad\Desktop\HumanDigitalTwin_LSTM
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#. .\venv\Scripts\Activate.ps1
#python ".\Training Iteration #15 - ANN time series\Train15.py"

# %%
#ANN-LSTM model
 # %%  
import os
import pandas as pd
import numpy as np
from numpy import std
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import mean
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sklearn
print(f'NumPy: {np.__version__}')
print(f'pandas: {pd.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print('##############################################################################################################################')
print("stage 1 import neccessary packages completed successfully ")
print()
print('##############################################################################################################################')

global all_accuracy
all_accuracy=list()
#to print DataFrame on single line
pd.set_option('expand_frame_repr', True)
time_steps=1
global x_train
global x_test
global y_train
global y_test


MODEL_DIR = "Train15_Daywise_Models"
os.makedirs(MODEL_DIR, exist_ok=True)

# %%
# load the prepared OULAD dataset
data_final=pd.read_csv('C:\\Users\\S_CSIS-PostGrad\\Desktop\\HumanDigitalTwin_LSTM\\Training Iteration #15 - ANN time series\\data.csv', sep=',')
print(data_final.head())
print(data_final.shape)
print()
print()

# %%



def prepare_dataset():
    print('experiment of time_steps= ',time_steps)
    print()
    print('##############################################################################################################################')

    global data_final 
    dataset_range=time_steps*32592
    print(data_final.head())
    custom_steps=data_final[data_final['date']<time_steps]
    
    print('19- custom_steps containing dataset rang of time_steps=  ',time_steps)
    print(custom_steps.head())
    print()
    print('##############################################################################################################################')
    #########################################################################################
    
    sample_rows=custom_steps
    print('sample_rows containing ')
    print(sample_rows.head())
    print(sample_rows.shape)
    print()
    print('##############################################################################################################################')

    

    ############################################################################################################################
    print('##############################################################################################################################')
    print()
    print('stage 6 prepare X and Y numby arrays')
    print()
    print('##############################################################################################################################')
    
    global Y
    global X

    X = sample_rows.drop(columns=['final_result','id_student']).to_numpy()
    print('X numpy containing sample_rows except final_result column')
    print(X[:5])
    print(X.shape)
    print()
    print('##############################################################################################################################')
     
    Y = np.array(sample_rows['final_result'])
    print('Y numpy containing sample_rows final_result column')
    print(Y[:5])
    print(Y.shape)
    print()
    print('##############################################################################################################################')
    
    print('##############################################################################################################################')
    print()
    print('stage 6 prepare X and Y completed successfully')
    print()
    print('##############################################################################################################################')
    
    #     data_final=data_final.drop(['final_result'],1)
   
    
    # apply one hot encoder for final result feature
    print('##############################################################################################################################')
    print()
    print('stage 7 # apply one hot encoder for final result feature in Y')
    print()
    print('##############################################################################################################################')
    
    
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    print('Y after apply fit_transform(y) label encoder')
    print(Y[:5])
    print(Y.shape)
    print()
    print('##############################################################################################################################')
    
    
    global target_strings 
    target_strings = label_encoder.classes_
    
    print('##############################################################################################################################')
    print()
    print('stage 7  apply one hot encoder for  Y completed successfully')
    print()
    print('##############################################################################################################################')
    

#     Y.reshape(-1,1,1)
#num of all instances=32,590 num of smple =round(.25*32590)
    print('##############################################################################################################################')
    print()
    print('stage 8  apply minmax scaler for features in X')
    print()
    print('##############################################################################################################################')
    # define min max scaler
    scaler = MinMaxScaler()

    # capture feature names for later preprocessing in Streamlit
    feature_cols = sample_rows.drop(columns=['final_result','id_student']).columns.tolist()

     # fit & transform
    X = scaler.fit_transform(X)

    # --- save preprocessing artifacts ---
    import pickle
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)


    print('X after apply minmax transforming')
    print(X[:1])
    print(X.shape)
    print()
    print('##############################################################################################################################')
    print('##############################################################################################################################')
    print()
    print('stage 8  Scaling features in X completed successfully')
    print()
    print('##############################################################################################################################')
    
    print('##############################################################################################################################')
    print()
    print('stage 9  splitting dataset  X,Y to train test 70 30')
    print()
    print('##############################################################################################################################')
    
    #split data to train test 70 30
    
    global x_train
    global x_test
    global y_train
    global y_test
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=(10000*time_steps))
    
    print('x_train ')
    print(x_train[:1])
    print(x_train.shape)
    print()
    print('##############################################################################################################################')
    
    print('y_train ')
    print(y_train[:1])
    print(y_train.shape)
    print()
    print('##############################################################################################################################')
    
    print('x_test ')
    print(x_test[:1])
    print(x_test.shape)
    print()
    print('##############################################################################################################################')
    
    print('y_test ')
    print(y_test[:1])
    print(y_test.shape)
    print()
    print('##############################################################################################################################')
    
    print('##############################################################################################################################')
    print()
    print('stage 9  splitting dataset  X,Y to train test 70 30 completted successfully')
    print()
    print('##############################################################################################################################')
    
    print('##############################################################################################################################')
    print()
    print('stage 10  Converts Y arrays  to binary class matrix using to_categorical ')
    print()
    print('##############################################################################################################################')
    
    
    ########################################################################
    #convert labelclass to categorical values 
    #You use to_categorical to transform your training data before you pass it to your model
    #Converts a class vector (integers) to binary class matrix.
    
    y_train = to_categorical(y_train)                                        #
    y_test = to_categorical(y_test)                                          #
    #########################################################################
    print('y_train ')
    print(y_train[:1])
    print(y_train.shape)
    print()
    print('##############################################################################################################################')
    
    print('y_test ')
    print(y_test[:1])
    print(y_test.shape)
    print()
    print('##############################################################################################################################')
    
    print('##############################################################################################################################')
    print()
    print('stage 10  Converts a class vector (integers) to binary class matrix completed successfully')
    print()
    print('##############################################################################################################################')
    

    print('##############################################################################################################################')
    print()
    print('stage 11  reshaping x_train,x_test to (round(len(x_train)/time_steps),time_steps,77) and ,y_train,y_test to (round(len(y_train)/time_steps),time_steps,4)')
    print()
    print('##############################################################################################################################')
    


    x_train=x_train.reshape(-1,time_steps,71)
    x_test=x_test.reshape(-1,time_steps,71)
    y_train=y_train.reshape(-1,time_steps,4)
    y_test=y_test.reshape(-1,time_steps,4)
    
    print('xxx_train ')
    print(x_train)
    print(x_train.shape)
    print()
    print('##############################################################################################################################')
    
    print('y_train ')
    print(y_train[:1])
    print(y_train.shape)
    print()
    print('##############################################################################################################################')
    
    print('x_test ')
    print(x_test[:1])
    print(x_test.shape)
    print()
    print('##############################################################################################################################')
    
    print('y_test ')
    print(y_test[:1])
    print(y_test.shape)
    print()
    print('##############################################################################################################################')
    
    print('##############################################################################################################################')
    print()
    print('stage 11  reshaping x_train,x_test ,y_train,y_test completted successfully')
    print()
    print('##############################################################################################################################')
    

# %%


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 100, 100
   # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    n_timesteps=time_steps
    n_features= 71
    n_outputs=4
   # n_outputs=trainy.shape[1] 
    model = Sequential()
    model._name='ANN-LSTM'
    model.add(LSTM(200, input_shape=(n_timesteps,n_features),return_sequences=True,recurrent_dropout=0.2,name="LSTM_Layer"))
    model.add(Dropout(0.5,name="Dropout_layer"))
    model.add(Dense(100, activation='relu',name="ANN_Hidden_Layer"))

    model.add(Dense(n_outputs, activation='softmax',name="ANN_Output_Layer"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())
   
    from keras.callbacks import CSVLogger
    log_path = os.path.join(MODEL_DIR, f"history_day_{time_steps}.csv")
    csv_logger = CSVLogger(log_path, append=False)

    history = model.fit(trainX, trainy,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=[csv_logger])    
    
    
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
    
    pyplot.title('categorical accuracy')
    pyplot.plot(history.history['categorical_accuracy'], label='train')
    pyplot.plot(history.history['val_categorical_accuracy'], label='val')
    #print(test_history)
    #pyplot.plot(test_history.history['categorical_accuracy'], label='test')
    pyplot.ylabel('categorical accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'val','test'], loc='upper left')
    pyplot.legend()

    #Save categorical accuracy images
    plot_path = os.path.join(MODEL_DIR, f"trainval_day_{time_steps}.png")
    pyplot.savefig(plot_path)
    pyplot.close()
    print(f"[Saved] train/val curves → {plot_path}")
   
    # -------save the trained model for this day ---
    model_filename = f"model_day_{time_steps}.h5"
    model.save(os.path.join(MODEL_DIR, model_filename))
    print(f"[Saved] day-{time_steps} model → {os.path.join(MODEL_DIR, model_filename)}")


    predicted = model.predict(testX)
    predictions = [np.round(value) for value in predicted]
   
    d = np.array(predictions, dtype=np.int32)
    
    
    testy=np.vstack(testy)
    d=np.vstack(d)
    
    report = classification_report(testy, d, target_names=target_strings)
   
    print(report)

    return accuracy

# %%
###########################################################################################################################
# summarize scores
def summarize_results(scores):
   # print(scores)
    m, s = mean(scores), std(scores)
    all_accuracy.append(m)
    print('time steps=%d  Evaluation Accuracy: %.3f%% (+/-%.3f)' % (time_steps,m, s))


# %%
###########################################################################################################################    
# run an experiment
def run_experiment(repeats=1):
    
    # repeat experiment
    scores = list()
    for r in range(repeats):
       
        score = evaluate_model(x_train, y_train, x_test,y_test)
       
        scores.append(score)
    # summarize results
    summarize_results(scores)


  

# %%
# run the experiment
###########################################################################################################################
while(time_steps<=211):
   
    prepare_dataset()
    run_experiment()
    time_steps=time_steps+1
print(all_accuracy)
pyplot.title('categorical accuracy')
pyplot.plot(all_accuracy, label='test')
pyplot.ylabel('categorical accuracy')
pyplot.xlabel('days')
pyplot.legend(['test'], loc='upper left')
pyplot.legend()

summary_path = os.path.join(MODEL_DIR, "all_days_accuracy.png")
pyplot.savefig(summary_path)
pyplot.close()
print(f"[Saved] all-days accuracy plot → {summary_path}")


# %%



