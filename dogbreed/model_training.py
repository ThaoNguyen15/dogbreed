from sklearn.datasets import load_files
from keras.utils import np_utils
from glob import glob
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

# Not sure if this is necessary
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# import dog  datasets
def load_dataset(path):
    data = load_files(path)
    filenames = np.array(data['filenames'])
    # TODO: how to dynamically get the number of targets
    num_targets = 133
    targets = np_utils.to_categorical(np.array(data['target']), num_targets)
    return filenames, targets

# import train, validation, test sets
train_files, train_targets = load_dataset('dogImages/train')
test_files, test_targets = load_dataset('dogImages/test')
valid_files, valid_targets = load_dataset('dogImages/valid')

# Get list of dog names here
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# detect human using opencv
import cv2
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# create detector function
def face_detector(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# Detect dogs, using existing architecture (RestNet50)
ResNet50_model = ResNet50(weights='imagenet')

# Pre-process data for ResNet50
def path_to_sensor(img_path):
    img = image.load_img(img_path, target_size=())
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# create a dog detector function
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    # Dog categories are from 151 to 268
    return ((prediction <= 268) & (prediction >= 151))

# create a dog breed detector using transfer learning

# TODO: this is only for training the network
# pre-process the data for Keras
# scaling pixel by 255
# train_tensors = paths_to_tensor(train_files).astype('float32')/255
# valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
# test_tensors = paths_to_tensor(test_files).astype('float32')/255

# TODO: Create bottneck features
# Right now, assume we load this from a pre-computed file
# We'll use Xception architecture 

bottleneck_features = np.load('bottleneck_features-npz-file')
train_data = bottleneck_features['train']
valid_data = bottleneck_features['valid']
test_data = bottleneck_features['test']

# Define Keras Network Architecture
tl_model = Sequential()
tl_model.add(AveragePooling2D(pool_size=4, strides=4, padding='same',
                              input_shape=train_data.shape[1:]))
tl_model.add(Flatten())
tl_model.add(Dense(133, activation='relu'))
tl_model.add(Dropout(.5))
tl_model.add(Dense(133, activation='softmax'))

saved_model_file = 'saved_models/weights.best.transferlearning.hdf5'
checkpointer = ModelCheckpoint(
    filepath=saved_model_file,
    verbose=1, save_best_only=True)

tl_model.fit(train_data, train_targets,
             validation_data=(valid_data, valid_targets),
             epochs=10, batch_size=20, callbacks=[checkpointer], verbose=1)

# load weights with best validation loss
tn_model.load_weights(saved_model_file)

# Print out test error
tn_predictions = [np.argmax(tn_model.predict(np.expand_dims(feature, axis=0)))
                  for feature in test_data]

correct = np.sum(np.array(tn_predictions)==np.argmax(test_targets, axis=1))
test_accuracy_score = 100 * correct / len(tn_predictions)
print('Test accuracy: %.4f%%' % test_accuracy_score)

from keras.applications.xception import Xception, preprocess_input

def extract_bottleneck(tensor):
    """Extract bottle neck features from 1 tensor
    We'll use Xception architecture
    """
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))    

def predict_breed(img_path):
    bottleneck_feature = extract_bottleneck(path_to_tensor(img_path))
    predicted_vector = tn_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

# write function to predict dog breed from dog and human picture
def human_dog_breed(filepath):
    # TODO: adjust this in an application sense
    # Determine if a dog
    is_dog = dog_detector(filepath)
    # Determine if a human
    is_human = face_detector(filepath)
    if not is_dog and not is_human:
        print("We can't detect a human face nor a dog")
        return None
    type_ = 'dog' if is_dog else 'human'
    print('Hello, {0}'.format(type_))
    img = cv2.imread(filepath)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    pred = tn_predict_breed(filepath)
    print('You look like a {0}'.format(pred))
