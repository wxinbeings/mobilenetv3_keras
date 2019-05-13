from functools import partial

import keras


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Identity(keras.layers.Layer):
    def call(self, input):
        return input


class HardSigmoid(keras.layers.Layer):
    def __init__(self, name='HardSigmoid'):
        super().__init__(name=name)
        self.relu6 = keras.layers.ReLU(max_value=6, name='{}/relu6'.format(name))

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0


class HardSwish(keras.layers.Layer):
    def __init__(self, name='HardSwish'):
        super().__init__(name=name)
        self.hard_sigmoid = HardSigmoid(name='{}/HardSigmoid'.format(name))

    def call(self, input):
        return input * self.hard_sigmoid(input)


class GlobalAveragePooling2D(keras.layers.Layer):
    """Return output shape (batch_size, rows, cols, channels).
   `keras.layer.GlobalAveragePooling2D` is (batch_size, channels),
    """

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, input_shape[-1])

    def call(self, input):
        pool_size = tuple(map(int, input.shape[1:3]))
        return keras.layers.AveragePooling2D(pool_size=pool_size, name='{}/AP'.format(self.name))(input)
        

def SqEx(n_features, reduction=4, name='SE'):
    if n_features % reduction != 0:
        raise ValueError('n_features must be divisible by reduction (default = 4)')

    def forward(x):
        y = GlobalAveragePooling2D(name='{}/GlobalAP'.format(name))(x)
        y = keras.layers.Dense(n_features // reduction, use_bias=True, name='{}/Dense1'.format(name))(y)
        y = keras.layers.ReLU(name='{}/relu'.format(name))(y)
        y = keras.layers.Dense(n_features, use_bias=True, name='{}/Dense2'.format(name))(y)
        y = HardSigmoid(name='{}/HardSigmoid'.format(name))(y)
        y = keras.layers.Multiply(name='{}/multi'.format(name))([x, y])
        return y

    return forward


def Bottleneck(outChannels, expChannels, kernel_size=3, stride=1, 
               use_se=False, nl='re', name='Bneck'
):
    if nl.lower() == 're':
        nl_layer = partial(keras.layers.ReLU, max_value=6)
    elif nl.lower() == 'hs':
        nl_layer = HardSwish
    else:
        raise ValueError

    if use_se:
        SELayer = partial(SqEx, n_features=expChannels)
    else:
        SELayer = Identity

    dw_padding = (kernel_size - 1) // 2

    def forward(x):
        y = keras.layers.Conv2D(filters=expChannels, kernel_size=1, strides=1, name='{}/pw_conv'.format(name))(x)
        y = keras.layers.BatchNormalization(momentum=0.99, name='{}/pw_BN'.format(name))(y)
        y = nl_layer(name='{}/pw_act'.format(name))(y)

        y = keras.layers.ZeroPadding2D(padding=dw_padding, name='{}/dw_padding'.format(name))(y)
        y = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, name='{}/dw_conv'.format(name))(y)
        y = keras.layers.BatchNormalization(momentum=0.99, name='{}/dw_BN'.format(name))(y)
        y = SELayer(name='{}/dw_SE'.format(name))(y)
        y = nl_layer(name='{}/dw_act'.format(name))(y)

        y = keras.layers.Conv2D(filters=outChannels, kernel_size=1, strides=1, name='{}/pw2_conv'.format(name))(y)
        y = keras.layers.BatchNormalization(momentum=0.99, name='{}/pw2_BN'.format(name))(y)
        return y

    return forward


def ConvBnAct(filters, kernel_size=3, stride=1, padding=0, name='CBA', act_layer=keras.layers.ReLU):
    if padding <= 0:
        pad_layer = Identity
    else:
        pad_layer = partial(keras.layers.ZeroPadding2D, padding=padding)

    def forward(x):
        y = pad_layer(name='{}/padding'.format(name))(x)
        y = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, name='{}/conv'.format(name))(y)
        y = keras.layers.BatchNormalization(momentum=0.99, name='{}/BN'.format(name))(y)
        y = act_layer(name='{}/act'.format(name))(y)
        return y

    return forward


def mobilenetv3(bneck_settings, input_tensor=None, num_classes=1000, width_multiplier=1.0, divisible_by=8, model_type='small', include_top=True):
    if input_tensor is None:
        input_tensor = keras.layers.Input((224, 224, 3))

    x = ConvBnAct(filters=16, kernel_size=3, stride=2, padding=1, act_layer=HardSwish, name='CBA0')(input_tensor)
    for i, (k, exp, out, se, nl, s) in enumerate(bneck_settings):
        out_channels = _make_divisible(out * width_multiplier, divisible_by)
        exp_channels = _make_divisible(exp * width_multiplier, divisible_by)
        x = Bottleneck(outChannels=out, expChannels=exp, kernel_size=k, stride=s,
                       use_se=se, nl=nl, name='Bneck{}'.format(i))(x)

    # last stage
    sizeChannels = 576 if model_type == 'small' else 960
    penultimate_channels = _make_divisible(sizeChannels * width_multiplier, divisible_by)
    last_channels = _make_divisible(1280 * width_multiplier, divisible_by)
    x = ConvBnAct(filters=penultimate_channels, kernel_size=1, stride=1, act_layer=HardSwish, name='CBA1')(x)
    x = SqEx(n_features=penultimate_channels, name='last_SE')(x)
    if include_top:
        x = GlobalAveragePooling2D(name='last_GAP')(x)
        x = HardSwish(name='last_HSwish1')(x)
        if model_type == 'small':
            x = ConvBnAct(filters=last_channels, kernel_size=1, stride=1, act_layer=HardSwish, name='CBA2')(x)
            x = ConvBnAct(filters=num_classes, kernel_size=1, stride=1, act_layer=HardSwish, name='CBA3')(x)
        elif model_type == 'large':
            x = keras.layers.Conv2D(filters=last_channels, kernel_size=1, strides=1, name='Conv2')(x)
            x = HardSwish(name='last_HSwish2')(x)
            x = keras.layers.Conv2D(filters=num_classes, kernel_size=1, strides=1, name='Conv3')(x)
    # print(x.shape)

    return keras.models.Model(inputs=input_tensor, outputs=x)


def MobileNetV3Small(input_tensor=None, num_classes=1000, width_multiplier=1.0, divisible_by=8, include_top=True):
    bneck_settings = [
        # k   exp   out  SE      nl     s
        [ 3,  16,   16,  True,   "RE",  2 ], # -- 56
        [ 3,  72,   24,  False,  "RE",  2 ], # -- 28
        [ 3,  88,   24,  False,  "RE",  1 ],
        [ 5,  96,   40,  True,   "HS",  2 ], # -- 14
        [ 5,  240,  40,  True,   "HS",  1 ],
        [ 5,  240,  40,  True,   "HS",  1 ],
        [ 5,  120,  48,  True,   "HS",  1 ],
        [ 5,  144,  48,  True,   "HS",  1 ],
        [ 5,  288,  96,  True,   "HS",  2 ], # -- 7
        [ 5,  576,  96,  True,   "HS",  1 ],
        [ 5,  576,  96,  True,   "HS",  1 ],
    ]
    return mobilenetv3(bneck_settings, input_tensor=input_tensor, num_classes=num_classes, width_multiplier=width_multiplier, 
                       divisible_by=divisible_by, include_top=include_top, model_type='small')


def MobileNetV3Large(input_tensor=None, num_classes=1000, width_multiplier=1.0, divisible_by=8, include_top=True):
    bneck_settings = [
        # k   exp   out   SE      NL     s
        [ 3,  16,   16,   False,  "RE",  1 ],
        [ 3,  64,   24,   False,  "RE",  2 ],
        [ 3,  72,   24,   False,  "RE",  1 ],
        [ 5,  72,   40,   True,   "RE",  2 ],
        [ 5,  120,  40,   True,   "RE",  1 ],
        [ 5,  120,  40,   True,   "RE",  1 ],
        [ 3,  240,  80,   False,  "HS",  2 ],
        [ 3,  200,  80,   False,  "HS",  1 ],
        [ 3,  184,  80,   False,  "HS",  1 ],
        [ 3,  184,  80,   False,  "HS",  1 ],
        [ 3,  480,  112,  True,   "HS",  1 ],
        [ 3,  672,  112,  True,   "HS",  1 ],
        [ 5,  672,  160,  True,   "HS",  1 ],
        [ 5,  672,  160,  True,   "HS",  2 ],
        [ 5,  960,  160,  True,   "HS",  1 ],
    ]
    return mobilenetv3(bneck_settings, input_tensor=input_tensor, num_classes=num_classes, width_multiplier=width_multiplier,
                       divisible_by=divisible_by, include_top=include_top, model_type='large')


if __name__ == '__main__':
    from keras.utils import plot_model
    model = MobileNetV3Small()
    print(model.summary())
    plot_model(model, to_file='./mobilenetv3_small.jpg')

    model = MobileNetV3Large()
    print(model.summary())
    plot_model(model, to_file='./mobilenetv3_large.jpg')

