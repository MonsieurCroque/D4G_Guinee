######################### Import des libraries #######################

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from pyrsgis import raster
from pyrsgis.convert import changeDimension
from tensorflow import keras
from scipy.signal import convolve2d as conv2
import os
import numpy as np

############################## Parameters ############################

stop = 1000
epochs = 7

######################### Import des données #########################

os.chdir("C:\\Users\\simon\\Documents\\Simon\\Data4Good\\GROUP_CNN\\")

#for testing
"""MSI = ['1_MSI_Seredou_2015_tiled\\S2A_20151203_MSI_1km2_1.tif']
test = ['1_Images_Seredou_2015_16bit_tiled\\S2A_20151203_seredou_1km2_1.tif']
truth = ['1_Mask_Seredou_2015_tiled\\GroundTruth_seredou_20151203_1km2_1.tif']"""

#for prod
test_raw = os.listdir("1_Images_Seredou_2015_16bit_tiled")
MSI_raw = os.listdir("1_MSI_Seredou_2015_tiled")
truth_raw = os.listdir("1_Mask_Seredou_2015_tiled")

test = {k.split("_")[-1]:('1_Images_Seredou_2015_16bit_tiled\\'+k) for k in test_raw}
MSI = {k.split("_")[-1]:('1_MSI_Seredou_2015_tiled\\'+k) for k in MSI_raw}
mask = {k.split("_")[-1]:('1_Mask_Seredou_2015_tiled\\'+k) for k in truth_raw}

########################### Prétraitement ############################

first = True
preprocessed_truth = np.array([])

for i in mask.keys():

    if i == str(stop) + ".tif":
        break
    
    #Read the rasters as array
    ds1,featuretest_MSI = raster.read(MSI[i], bands='all')
    ds2,featuretest = raster.read(test[i], bands='all')
    ds3,truth = raster.read(mask[i], bands='all')
    
    #On remplace les valeurs négatives par 0
    featuretest_MSI[featuretest_MSI < 0] = 0
    featuretest[featuretest < 0] = 0
    featuretest = np.append(featuretest, np.array([featuretest_MSI]),axis=0)
    
    preprocessed_truth = np.append(preprocessed_truth, changeDimension(truth))
    
    #On importe les données
    for i in range (11):
    
        featuretest[i] = featuretest[i]/featuretest[i].max()
        
        if i == 0:
            preprocessed_data = np.array([changeDimension(featuretest[i])])
        else:
            preprocessed_data = np.append(preprocessed_data, [changeDimension(featuretest[i])], axis = 0)
        
        #On calcule les moyennes des cases adjacentes - creation de 11 nouvelles colonnes
        featuretest_avg = conv2(featuretest[i],np.ones((5,5)),'same')/25
        
        #On passe en tabulaire
        preprocessed_data = np.append(preprocessed_data, [changeDimension(featuretest_avg)], axis = 0)

    #on reshape
    new_data = np.transpose(preprocessed_data)
    if first:
        data_predict = np.array([new_data])
        data_train = new_data
        first = False
    else:
        data_predict = np.append(data_predict,np.array([new_data]),axis=0)
        data_train = np.append(data_train,new_data,axis=0)
        

########################## Entrainement du modèle ###########################

#Decoupage en train test 
xTrain, xTest, yTrain, yTest = train_test_split(data_train, preprocessed_truth, test_size=0.20, random_state=42)

#Define the parameters of the model
model = keras.Sequential([keras.layers.Dense(100),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(4, activation='softmax')])

#Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Run the model
model.fit(xTrain, yTrain, epochs=epochs)

############################### Statistiques ################################

#Predict for test data 
yTestPredicted = model.predict(xTest)

#Calculate and display the error metrics
yTestPredicted = (yTestPredicted>0.5).astype(int)
yTestPredicted = np.argmax(yTestPredicted, axis=1)

cMatrix = confusion_matrix(yTest, yTestPredicted)
pScore = precision_score(yTest, yTestPredicted,average='micro')
rScore = recall_score(yTest, yTestPredicted,average='micro')

print("Confusion matrix: for 100 nodes\n", cMatrix)

print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))

########################## Entrainement du modèle ###########################

count = 0

for i in mask.keys():
    
    if i == str(stop) + ".tif":
        break
    
    #Export des resultats
    predicted = model.predict(data_predict[count,:,:])
    predicted_test = np.argmax(predicted, axis=1)
    
    #Export raster
    prediction = np.reshape(predicted_test, (ds1.RasterYSize, ds1.RasterXSize))
    outFile = 'data_predicted' + i
    raster.export(prediction, ds3, filename=outFile, dtype='float')
    count += 0