import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Model, Sequential

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class GlobalAvgPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2])

class ConvBNReLU(tf.keras.layers.Layer):
    # Removed 'in_planes' because Keras infers input shapes automatically
    def __init__(self, out_planes, kernel_size=3, stride=1, groups=1, **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)
        # Added use_bias=False because BatchNormalization makes bias redundant
        if groups == 1:
            self.conv = Conv2D(out_planes, kernel_size, stride, padding='same', use_bias=False)
        else:
            self.conv = DepthwiseConv2D(kernel_size, stride, padding='same', use_bias=False)
        
        self.bn = BatchNormalization()
        self.relu = ReLU(max_value=6)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training) # CRITICAL: pass training flag
        x = self.relu(x)
        return x

class InvertedResidual(tf.keras.layers.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, stride=stride, groups=hidden_dim),
            Conv2D(oup, kernel_size=1, use_bias=False), # No bias before BN
            BatchNormalization(),
        ])
        self.conv = Sequential(layers)

    def call(self, inputs, training=None):
        x = self.conv(inputs, training=training) # CRITICAL: pass training flag
        if self.use_res_connect:
            return inputs + x
        return x

class MobileNetV2Backbone(tf.keras.Model):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, input_channel=32, last_channel=1280, **kwargs):
        super(MobileNetV2Backbone, self).__init__(**kwargs)
        block = InvertedResidual
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.out_channels = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.features = [ConvBNReLU(input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features.append(ConvBNReLU(self.out_channels, kernel_size=1))
        self.features = Sequential(self.features)

    def call(self, inputs, training=None):
        return self.features(inputs, training=training)

class MobileNetV2Head(tf.keras.Model):
    def __init__(self, input_channel=1280, num_classes=1000, has_dropout=False, activation="None", **kwargs):
        super(MobileNetV2Head, self).__init__(**kwargs)
        self.head = Sequential([GlobalAvgPooling()])
        if has_dropout:
            self.head.add(Dropout(0.2))
        self.dense = Dense(num_classes)
        
        # Renamed variable to avoid conflicts with Keras internal properties
        self.activation_fn = None
        if activation == "Sigmoid":
            self.activation_fn = tf.keras.activations.sigmoid
        elif activation == "Softmax":
            self.activation_fn = tf.keras.activations.softmax

    def call(self, inputs, training=None):
        x = self.head(inputs, training=training)
        x = self.dense(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class MobileNetV2Combine(tf.keras.Model):
    def __init__(self, backbone, head, **kwargs):
        super(MobileNetV2Combine, self).__init__(**kwargs)
        self.backbone = backbone
        self.head = head

    def call(self, inputs, training=None):
        x = self.backbone(inputs, training=training)
        x = self.head(x, training=training)
        return x

def mobilenet_v2(num_classes):
    backbone_net = MobileNetV2Backbone()
    head_net = MobileNetV2Head(backbone_net.out_channels, num_classes)
    return MobileNetV2Combine(backbone_net, head_net)

# 1. Create the Model
model = mobilenet_v2(num_classes=1000)

# 2. Build properly using a dummy forward pass to initialize all nested shapes
dummy_input = tf.zeros((1, 224, 224, 3))
_ = model(dummy_input)

# 3. Use expand_nested=True to see the inner sequential layers instead of just "backbone" and "head"
model.summary(expand_nested=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

