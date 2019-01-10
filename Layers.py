def DenseLayer(inputs, k, bn=None, drop=None):
    x = BatchNormalization()(inputs)
    if bn: # Bottleneck
        x = Activation('relu')(x)
        x = Conv3D(k*bn, (1,1,1))(x)
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(k, (1,3,3), padding='same')(x)
    if drop:
        x = Dropout(drop)(x)
    return concatenate([inputs, x], axis=4)

def DenseBlock(x, k, L, bn=None, drop=None):
    for l in range(L):
        x = DenseLayer(x, k, bn, drop)
    return x

def TransitionLayer(x, k):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(k, (1,1,1))(x)
    x = AvgPool3D((1,2,2), strides=(1,2,2), padding='same')(x)
    return x

def DenseNet(shape, k, bn, theta, drop, B, L, outs):
    img_input = Input(shape=(4,shape,shape,3))
    x = Conv3D(k*2, (1,7,7), strides=(1,2,2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool3D((1,3,3), strides=(1,2,2), padding='same')(x)
    for i in range(B):
        x = DenseBlock(x, k, L, bn, drop)
        if i != B-1:
            k = int(k*theta) # compression
            x = TransitionLayer(x, k)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(37, (1,7,7))(x)
    x = GlobalAveragePooling3D()(x)
    x = Activation('sigmoid')(x)
    return Model(inputs=img_input, outputs=x)
