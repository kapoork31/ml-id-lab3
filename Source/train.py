import sagemaker
import sagemaker.amazon.common as smac
from sagemaker import get_execution_role
from sagemaker.predictor import json_deserializer
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter
import boto3, csv, io, json, re, os, sys, pprint, time, random
from time import gmtime, strftime
from botocore.client import Config
import numpy as np

# sagemaker containers for factorization machines
containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/factorization-machines:latest',
             'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/factorization-machines:latest',
             'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/factorization-machines:latest',
             'ap-northeast-1': '351501993468.dkr.ecr.ap-northeast-1.amazonaws.com/factorization-machines:latest',
             'ap-northeast-2': '835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/factorization-machines:latest',
             'ap-southeast-2': '712309505854.dkr.ecr.ap-southeast-2.amazonaws.com/factorization-machines:latest',
             'eu-central-1': '664544806723.dkr.ecr.eu-central-1.amazonaws.com/factorization-machines:latest',
             'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/factorization-machines:latest'}

start = time.time()
current_timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())

role = sys.argv[1]
bucket = sys.argv[2]
stack_name = sys.argv[3]
commit_id = sys.argv[4]
commit_id = commit_id[0:7]

files = os.listdir('.')
print(files)

#
# load parameters created by previous data prep step
#
#with open('prepdata_result.json', 'r') as prepdata_result_file:
#    prepdata_result = json.load(prepdata_result_file)#

#nbUsers = prepdata_result["Parameters"]["NbUsers"]
#nbFeatures = prepdata_result["Parameters"]["NbFeatures"]
#nbRatingsTrain = prepdata_result["Parameters"]["NbRatingsTrain"]
#nbRatingsTest = prepdata_result["Parameters"]["NbRatingsTest"]
#trainingData = prepdata_result["Parameters"]["TrainingData"]
#testData = prepdata_result["Parameters"]["TestData"]

# For each user, build a list of rated movies.
# We'd need this to add random negative samples.
#moviesByUser = {}
#for userId in range(nbUsers):
#    moviesByUser[str(userId)]=[]

#with open(trainingData,'r') as f:
#    samples=csv.reader(f,delimiter='\t')
#    for userId,movieId,rating,timestamp in samples:
#        moviesByUser[str(int(userId)-1)].append(int(movieId)-1)

#
# Build training set and test set
#
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
#
#    Y=np.array(Y).astype('float32')
#    return X,Y



#X_train, Y_train = loadDataset(trainingData, nbRatingsTrain, nbFeatures)
#X_test, Y_test = loadDataset(testData,nbRatingsTest,nbFeatures)


data_key_x_train = 'data/x_train.npy'
data_key_x_test = 'data/x_test.npy'
data_key_y_train = 'data/y_train.npy'
data_key_y_test = 'data/y_test.npy'

s3 = boto3.resource('s3')
s3.Bucket(bucket).download_file(data_key_x_train, 'x_train.npy')
s3.Bucket(bucket).download_file(data_key_x_test, 'x_test.npy')
s3.Bucket(bucket).download_file(data_key_y_train, 'y_train.npy')
s3.Bucket(bucket).download_file(data_key_y_test, 'y_test.npy')

#x_train = np.load('x_train.npy')
#x_test = np.load('x_test.npy')
#y_train = np.load('y_train.npy')
#y_test = np.load('y_test.npy')
#print(x_train.shape)

data_key = 'data'
data_location = 's3://{}/{}/'.format(bucket, data_key)
#print(data_location)
inputs = {'train':data_location}
print(inputs)

import sagemaker
from sagemaker.tensorflow import TensorFlow

model_dir = '/opt/ml/model'
train_instance_type='ml.m5.large'
instance_count = 2

distributions = {'parameter_server': {'enabled': True}}

hyperparameters = {'epochs': 1, 'batch_size': 8, 'learning_rate': 0.01}

estimator = TensorFlow(
                       py_version="py3",
                       source_dir='Source/training',
                       entry_point='script_train.py',
                       model_dir=model_dir,
                       train_instance_type=train_instance_type,
                       train_instance_count=instance_count,
                       hyperparameters=hyperparameters,
                       role = role,
                       base_job_name='tf-fizzyo-breaths',
                       framework_version='1.13',
                       distributions = distributions,
                       script_mode=True)

estimator.fit(inputs)
#print('sdf')
#print(x_test.shape)
#assert X_train.shape == (nbRatingsTrain, nbFeatures)
#assert Y_train.shape == (nbRatingsTrain, )
#zero_labels = np.count_nonzero(Y_train)
#print("Training labels: %d zeros, %d ones" % (zero_labels, nbRatingsTrain-zero_labels))
model = estimator.model_data
json_test_path = 's3://{}/{}'.format(bucket, 'data/test_json.json')
output_path  = 's3://{}/{}'.format(bucket, 'test_data')

from sagemaker.tensorflow.serving import Model
#from sagemaker.tensorflow import TensorFlowModel
from sagemaker import get_execution_role
sagemaker_session = sagemaker.Session()
role = get_execution_role()

#tensorflow_serving_model = Model(model_data=model,
#                                 role=role,
#                                 framework_version='1.13',
#                                 sagemaker_session=sagemaker_session)

#transformer = tensorflow_serving_model.transformer(
#    instance_count=1,
#    instance_type='ml.m5.large',
#    output_path=output_path
#)

#transformer.transform(json_test_path, content_type='application/json')
#transformer.wait()

#print(x_train.shape)
#print(y_train.shape)
#assert X_test.shape  == (nbRatingsTest, nbFeatures)
#assert Y_test.shape  == (nbRatingsTest, )
#zero_labels = np.count_nonzero(Y_test)
#print("Test labels: %d zeros, %d ones" % (zero_labels, nbRatingsTest-zero_labels))

#
# Convert to protobuf and save to S3
#
#prefix = 'sagemaker/fm-movielens-pipeline'

#train_key      = 'train.protobuf'
#train_prefix   = '{}/{}'.format(prefix, 'train')

#test_key       = 'test.protobuf'
#test_prefix    = '{}/{}'.format(prefix, 'test')

#output_prefix  = 's3://{}/{}/output'.format(bucket, prefix)

#def writeDatasetToProtobuf(X, Y, bucket, prefix, key):
#    buf = io.BytesIO()
#    smac.write_spmatrix_to_sparse_tensor(buf, X, Y)
#    buf.seek(0)
#    obj = '{}/{}'.format(prefix, key)
#    boto3.resource('s3').Bucket(bucket).Object(obj).upload_fileobj(buf)
#    return 's3://{}/{}'.format(bucket,obj)

#train_data = writeDatasetToProtobuf(X_train, Y_train, bucket, train_prefix, train_key)
#test_data  = writeDatasetToProtobuf(X_test, Y_test, bucket, test_prefix, test_key)

#print(train_data)
#print(test_data)
#print('Output: {}'.format(output_prefix))

#best_model = ""

#job_name = stack_name + "-" + commit_id

#fm = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                   #role,
                                   #train_instance_count=1,
                                   #train_instance_type='ml.c4.xlarge',
                                   #output_path=output_prefix,
                                   #base_job_name=job_name,
                                   #sagemaker_session=sagemaker.Session())

#no_hyper_parameter_tuning = False

#if (no_hyper_parameter_tuning):
    #
    # Run the training job
    #
#    fm.set_hyperparameters(feature_dim=nbFeatures,
#                          predictor_type='binary_classifier',
#                          mini_batch_size=1000,
#                          num_factors=64,
#                          epochs=100)

#    fm.fit({'train': train_data, 'test': test_data})

#    best_model = fm.model_data
#else:
#    fm.set_hyperparameters(feature_dim=nbFeatures,
#                          predictor_type='binary_classifier',
#                          mini_batch_size=1000,
#                          num_factors=64,
#                          epochs=100)

#    my_tuner = HyperparameterTuner(
#        estimator=fm,
#        objective_metric_name='test:binary_classification_accuracy',
#        hyperparameter_ranges={
#            'epochs': IntegerParameter(1, 200),
#            'mini_batch_size': IntegerParameter(10, 10000),
#            'factors_wd': ContinuousParameter(1e-8, 512)},
#        max_jobs=4,
#        max_parallel_jobs=4)

#    my_tuner.fit({'train': train_data, 'test': test_data}, include_cls_metadata = False)

#    my_tuner.wait()

    #sm_session = sagemaker.Session()
    #best_log = sm_session.logs_for_job(my_tuner.best_training_job())
    #print(best_log)
#    best_model = '{}/{}/output/model.tar.gz'.format(output_prefix, my_tuner.best_training_job())

#print('Best model: {}'.format(best_model))
#
# Save config files to be used later for qa and prod sagemaker endpoint configurations
# and for prediction tests
#
config_data_qa = {
  "Parameters":
    {
        "BucketName": bucket,
        "CommitID": commit_id,
        "Environment": "qa",
        "ParentStackName": stack_name,
        "ModelData": model,
        "ContainerImage": containers[boto3.Session().region_name],
        "Timestamp": current_timestamp
    }
}

config_data_prod = {
  "Parameters":
    {
        "BucketName": bucket,
        "CommitID": commit_id,
        "Environment": "prod",
        "ParentStackName": stack_name,
        "ModelData": model,
        "ContainerImage": containers[boto3.Session().region_name],
        "Timestamp": current_timestamp
    }
}

#pprint.pprint(config_data_qa)
#pprint.pprint(config_data_prod)

json_config_data_qa = json.dumps(config_data_qa)
json_config_data_prod = json.dumps(config_data_prod)

f = open( './CloudFormation/configuration_qa.json', 'w' )
f.write(json_config_data_qa)
f.close()

f = open( './CloudFormation/configuration_prod.json', 'w' )
f.write(json_config_data_prod)
f.close()

end = time.time()
print(end - start)
