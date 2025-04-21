from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Concatenate
)


def Wide_CNN(weeks, days, channel, wide_len, lr=0.005, decay=1e-5, momentum=0.9):  
    # Deep branch input (reshaped sequence as image-like tensor)
    inputs_deep = Input(shape=(weeks * 3, days * 3, channel))
    # Wide branch input (raw features)
    inputs_wide = Input(shape=(wide_len,))

    # Deep convolutional path
    x_deep = Conv2D(32, (3, 3), strides=(3, 3), padding='same', kernel_initializer='he_normal')(inputs_deep)
    x_deep = MaxPooling2D(pool_size=(3, 3))(x_deep)
    x_deep = Flatten()(x_deep)
    x_deep = Dense(128, activation='relu')(x_deep)

    # Wide dense path
    x_wide = Dense(128, activation='relu')(inputs_wide)
        
    # Merge wide and deep paths
    x = Concatenate()([x_wide, x_deep])
    x = Dense(64, activation='relu')(x)

    # Final prediction layer
    pred = Dense(1, activation='sigmoid')(x)
    
    # Build and compile model
    model = Model(inputs=[inputs_wide, inputs_deep], outputs=pred)
    # Updated to use `learning_rate` instead of deprecated `lr` and remove decay
    sgd = SGD(learning_rate=lr, momentum=momentum, nesterov=True)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
