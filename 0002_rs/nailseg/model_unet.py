import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
from keras.models import Model
from keras.optimizers import Adam


class FingerNailUNet(object):
    def __init__(self, input_size):
        self.input_size = input_size
        self.pos_weight = 50
        self.__create_unet()

    # 此函数用于在预存的训练集上训练模型
    def fit(self, x, y, validation_split=0.0, verbose=1, batch_size=4, epochs=50):
        self.model.fit(x, y, validation_split=validation_split, verbose=verbose,
                       batch_size=batch_size, epochs=epochs, callbacks=self.__build_callbacks())

    def load_model(self, model_loc='./model/unet.h5'):
        self.model.load_weights(model_loc)

    def predict(self, x, verbose=1):
        return self.model.predict(x, verbose=verbose)

    def __create_unet(self):
        s = Input(self.input_size)
        c1 = Conv2D(8, 3, activation='relu', padding='same')(s)
        c1 = Conv2D(8, 3, activation='relu', padding='same')(c1)
        p1 = MaxPooling2D(padding='same')(c1)
        c2 = Conv2D(16, 3, activation='relu', padding='same')(p1)
        c2 = Conv2D(16, 3, activation='relu', padding='same')(c2)
        p2 = MaxPooling2D(padding='same')(c2)
        c3 = Conv2D(32, 3, activation='relu', padding='same')(p2)
        c3 = Conv2D(32, 3, activation='relu', padding='same')(c3)
        p3 = MaxPooling2D(padding='same')(c3)
        c4 = Conv2D(64, 3, activation='relu', padding='same')(p3)
        c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)
        p4 = MaxPooling2D(padding='same')(c4)
        c5 = Conv2D(128, 3, activation='relu', padding='same')(p4)
        c5 = Conv2D(128, 3, activation='relu', padding='same')(c5)
        u6 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c5)
        u6 = Concatenate(axis=3)([u6, c4])
        c6 = Conv2D(64, 3, activation='relu', padding='same')(u6)
        c6 = Conv2D(64, 3, activation='relu', padding='same')(c6)
        u7 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c6)
        u7 = Concatenate(axis=3)([u7, c3])
        c7 = Conv2D(32, 3, activation='relu', padding='same')(u7)
        c7 = Conv2D(32, 3, activation='relu', padding='same')(c7)
        u8 = Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(c7)
        u8 = Concatenate(axis=3)([u8, c2])
        c8 = Conv2D(16, 3, activation='relu', padding='same')(u8)
        c8 = Conv2D(16, 3, activation='relu', padding='same')(c8)
        u9 = Conv2DTranspose(8, 2, strides=(2, 2), padding='same')(c8)
        u9 = Concatenate(axis=3)([u9, c1])
        c9 = Conv2D(8, 3, activation='relu', padding='same')(u9)
        c9 = Conv2D(8, 3, activation='relu', padding='same')(c9)
        outputs = Conv2D(1, 1, activation='sigmoid')(c9)

        self.model = Model(inputs=[s], outputs=[outputs])
        self.model.compile(optimizer=Adam(), loss=self.weighted_binary_crossentropy(), metrics=[self.__mean_iou])

    def __mean_iou(self, y_true, y_pred):
        yt0 = y_true[:, :, :, 0]
        yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
        inter = tf.math.count_nonzero(tf.math.logical_and(tf.math.equal(yt0, 1), tf.math.equal(yp0, 1)))
        union = tf.math.count_nonzero(tf.math.add(yt0, yp0))
        iou = tf.where(tf.math.equal(union, 0), 1., tf.cast(inter / union, 'float32'))
        return iou

    def __build_callbacks(self):
        # TODO: Here we only have 2 data points, which doesn't have validate set,
        #  should change monitor to "val_loss" later after we have validate set in train set
        checkpointer = ModelCheckpoint(monitor='loss', filepath='./model/unet.h5', verbose=1, save_best_only=True,
                                       save_weights_only=True)
        # stop_train = EarlyStopping(monitor='loss', patience=10, verbose=1)
        # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8,
        #                               patience=3, min_lr=0.00001)
        callbacks = [checkpointer]  # origin is [checkpointer, reduce_lr, stop_train]
        return callbacks

    def weighted_binary_crossentropy(self):
        def _to_tensor(x, dtype):
            return tf.convert_to_tensor(x, dtype=dtype)

        def _calculate_weighted_binary_crossentropy(target, output, from_logits=False):
            if not from_logits:
                _epsilon = _to_tensor(K.common.epsilon(), output.dtype.base_dtype)
                output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
                output = tf.math.log(output / (1 - output))
            return tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=self.pos_weight)

        def _weighted_binary_crossentropy(y_true, y_pred):
            return K.mean(_calculate_weighted_binary_crossentropy(y_true, y_pred), axis=-1)

        return _weighted_binary_crossentropy
