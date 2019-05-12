from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

def inception_v4_base(input):

    net = _conv2d(input, 32, 3, 3, subsample=(2,2), padding='valid')
    net = _conv2d(net, 32, 3, 3, padding='valid')
    net = _conv2d(net, 64, 3, 3)

    branch_0 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
    branch_1 = _conv2d(net, 96, 3, 3, subsample=(2,2), padding='valid')

    net = concatenate([branch_0,branch_1], axis=-1)

    branch_0 = _conv2d(net, 64, 1, 1)
    branch_0 = _conv2d(branch_0, 96, 3, 3, padding='valid')
    branch_1 = _conv2d(net, 64, 1, 1)
    branch_1 = _conv2d(branch_1, 64, 1, 7)
    branch_1 = _conv2d(branch_1, 64, 7, 1)
    branch_1 = _conv2d(branch_1, 96, 3, 3, padding='valid')
    net = concatenate([branch_0,branch_1], axis=-1)
    branch_0 = _conv2d(net, 192, 3, 3, subsample=(2,2), padding='valid')
    branch_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)

    net = concatenate([branch_0,branch_1], axis=-1)

    for _ in range(4):
      net = block_inception_a(net)

    net = block_reduction_a(net)

    for _ in range(7):
      net = block_inception_b(net)

    net = block_reduction_b(net)

    for _ in range(3):
      net = block_inception_c(net)

    return net


def inception_v4_model(img_rows, img_cols, color_type=1, num_classeses=None, dropout_keep_prob=0.2):
    inputs = Input((299, 299, 3))
    net = inception_v4_base(inputs)

    net_old = AveragePooling2D((8,8), padding='valid')(net)

    net_old = Dropout(dropout_keep_prob)(net_old)
    net_old = Flatten()(net_old)

    predictions = Dense(units=1001, activation='softmax')(net_old)

    model = Model(inputs, predictions, name='inception_v4')


    _weights= './inception-v4_weights_tf_dim_ordering_tf_kernels.h5'

    model.load_weights(_weights, by_name=True)

    net_ft = AveragePooling2D((8,8), padding='valid')(net)
    net_ft = Dropout(dropout_keep_prob)(net_ft)
    net_ft = Flatten()(net_ft)
    predictions_ft = Dense(units=num_classes, activation='softmax')(net_ft)

    model = Model(inputs, predictions_ft, name='inception_v4')

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def block_inception_c(input):

    branch_0 = _conv2d(input, 256, 1, 1)

    branch_1 = _conv2d(input, 384, 1, 1)
    branch_1a = _conv2d(branch_1, 256, 1, 3)
    branch_1b = _conv2d(branch_1, 256, 3, 1)
    branch_1 = concatenate([branch_1a,branch_1b], axis=-1)


    branch_2 = _conv2d(input, 384, 1, 1)
    branch_2 = _conv2d(branch_2, 448, 3, 1)
    branch_2 = _conv2d(branch_2, 512, 1, 3)
    branch_2a = _conv2d(branch_2, 256, 1, 3)
    branch_2b = _conv2d(branch_2, 256, 3, 1)
    branch_2 = concatenate([branch_2a,branch_2b], axis=-1)


    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = _conv2d(branch_3, 256, 1, 1)
    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)

    return x

def block_inception_a(input):
    branch_0 = _conv2d(input, 96, 1, 1)

    branch_1 = _conv2d(input, 64, 1, 1)
    branch_1 = _conv2d(branch_1, 96, 3, 3)

    branch_2 = _conv2d(input, 64, 1, 1)
    branch_2 = _conv2d(branch_2, 96, 3, 3)
    branch_2 = _conv2d(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = _conv2d(branch_3, 96, 1, 1)

    x = concatenate([branch_0,branch_1, branch_2], axis=-1)

    return x

def block_inception_b(input):
    branch_0 = _conv2d(input, 384, 1, 1)

    branch_1 = _conv2d(input, 192, 1, 1)
    branch_1 = _conv2d(branch_1, 224, 1, 7)
    branch_1 = _conv2d(branch_1, 256, 7, 1)

    branch_2 = _conv2d(input, 192, 1, 1)
    branch_2 = _conv2d(branch_2, 192, 7, 1)
    branch_2 = _conv2d(branch_2, 224, 1, 7)
    branch_2 = _conv2d(branch_2, 224, 7, 1)
    branch_2 = _conv2d(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = _conv2d(branch_3, 128, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)

    return x

def _conv2d(x, num_filters, h, w, padding='same', subsample=(1, 1), bias=False):

    x = Conv2D(num_filters, kernel_size=(h, w),
                      strides=subsample,
                      padding=padding,
                      use_bias=bias)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x

def block_reduction_b(input):
    branch_0 = _conv2d(input, 192, 1, 1)
    branch_0 = _conv2d(branch_0, 192, 3, 3, subsample=(2, 2), padding='valid')

    branch_1 = _conv2d(input, 256, 1, 1)
    branch_1 = _conv2d(branch_1, 256, 1, 7)
    branch_1 = _conv2d(branch_1, 320, 7, 1)
    branch_1 = _conv2d(branch_1, 320, 3, 3, subsample=(2,2), padding='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)
    x = concatenate([branch_0,branch_1, branch_2], axis=-1)

    return x

def block_reduction_a(input):
    branch_0 = _conv2d(input, 384, 3, 3, subsample=(2,2), padding='valid')

    branch_1 = _conv2d(input, 192, 1, 1)
    branch_1 = _conv2d(branch_1, 224, 3, 3)
    branch_1 = _conv2d(branch_1, 256, 3, 3, subsample=(2,2), padding='valid')

    branch_2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)
    x = concatenate([branch_0,branch_1, branch_2], axis=-1)

    return x


if __name__ == '__main__':

    train_data_dir = '../dataset/train'
    validation_data_dir = '../dataset/val'

    num_train_samples = 33363
    num_val_samples = 8343

    img_rows, img_cols = 299, 299 
    channel = 3
    num_classes = 2 
    batch_size = 1 
    epochs = 1


    model = inception_v4_model(img_rows, img_cols, channel, num_classes, dropout_keep_prob=0.2)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_rows),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_rows),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='categorical')

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch= num_train_samples // batch_size,
        epochs= epochs,
        validation_data=validation_generator,
        validation_steps= num_val_samples // batch_size)