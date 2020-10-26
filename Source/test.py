import sagemaker
import sagemaker.amazon.common as smac
from sagemaker import get_execution_role
from sagemaker.predictor import json_deserializer
from sagemaker.tensorflow.serving import Model

import sys, json, csv, time, pprint
from time import gmtime, strftime
import numpy as np

start = time.time()

endpoint_name = sys.argv[1]
#prepdata_result_filepath = sys.argv[2]
cf_configuration_filepath = sys.argv[2]

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
json_str = {'instances': np.asarray(x_test).astype(float).tolist()}
# load parameters created by previous data prep and training steps
#
#with open(prepdata_result_filepath, 'r') as prepdata_result_file:
#    prepdata_result = json.load(prepdata_result_file)

#nbUsers = prepdata_result["Parameters"]["NbUsers"]
#nbFeatures = prepdata_result["Parameters"]["NbFeatures"]
#nbRatingsTest = prepdata_result["Parameters"]["NbRatingsTest"]
#testData = prepdata_result["Parameters"]["TestData"]



with open(cf_configuration_filepath) as cf_configuration_file:
    cf_configuration = json.load(cf_configuration_file)

commit_id = cf_configuration["Parameters"]["CommitID"]
timestamp = cf_configuration["Parameters"]["Timestamp"]
model_data = cf_configuration["Parameters"]["ModelData"]
stage = cf_configuration["Parameters"]["Environment"]

sagemaker_session = sagemaker.Session()
role = get_execution_role()

tensorflow_serving_model = Model(model_data=model_data,
                                 role=role,
                                 framework_version='1.13',
                                 sagemaker_session=sagemaker_session)

predictor = tensorflow_serving_model.deploy(initial_instance_count=1, instance_type='ml.t2.medium')

result = predictor.predict(json_str)
result = result['predictions']
pred=np.array(result)
pred = pred.argmax(axis = 1).tolist()
y_test_pred = y_test.argmax(axis = 1).tolist()

if((stage == 'qa') or (stage == 'prod')):
    predictor.delete_endpoint()

match = 0
for i,x in enumerate(pred):
    if(y_test_pred[i] == pred[i]):
        match = match + 1
acc = match/len(pred)
#endpoint_name = endpoint_name + "-" + commit_id + "-" + timestamp

#def loadDataset(filename, lines, columns):
#    # Features are one-hot encoded in a sparse matrix
#    X = lil_matrix((lines, columns)).astype('float32')
#    # Labels are stored in a vector
#    Y = []
#    line=0
#    with open(filename,'r') as f:
#        samples=csv.reader(f,delimiter='\t')
#        for userId,movieId,rating,timestamp in samples:
#            X[line,int(userId)-1] = 1
#            X[line,int(nbUsers)+int(movieId)-1] = 1
#            if int(rating) >= 4:
#                Y.append(1)
#            else:
#                Y.append(0)
#            line=line+1

#    Y=np.array(Y).astype('float32')
#    return X,Y

#X_test, Y_test = loadDataset(testData,nbRatingsTest,nbFeatures)
#print(X_test.shape)
#print(Y_test.shape)
#assert X_test.shape  == (nbRatingsTest, nbFeatures)
#assert Y_test.shape  == (nbRatingsTest, )
#zero_labels = np.count_nonzero(Y_test)
#print("Test labels: %d zeros, %d ones" % (zero_labels, nbRatingsTest-zero_labels))

#
# Create a sagemaker real-time predictor
#
#def fm_serializer(data):
#    js = {'instances': []}
#    for row in data:
#        js['instances'].append({'features': row.tolist()})
#    #print js
#    return json.dumps(js)

#fm_predictor = sagemaker.predictor.RealTimePredictor(endpoint_name,
                                                     #serializer=fm_serializer,
                                                     #deserializer=json_deserializer,
                                                     #content_type='application/json',
                                                     #sagemaker_session=sagemaker.Session())
#nb_predictions = 10
#offset = 1000

# Run some predictions
#result = fm_predictor.predict(X_test[offset:offset+nb_predictions].toarray())
#pprint.pprint(result["predictions"])
#pprint.pprint(Y_test[offset:offset+nb_predictions])

# Compare predictions to labelled test data to check for match rate
#matches = 0
#for index in list(range(nb_predictions)):
#    offset_index = offset + index
#    if int(result["predictions"][index]["predicted_label"]) == int(Y_test[offset_index]):
#        matches = matches + 1

#match_rate = matches / nb_predictions
#print("Match Rate: %s" % (match_rate))

If match rate is not 80% we throw an error that will break the codepipeline test stage
assert acc >= 0.80
