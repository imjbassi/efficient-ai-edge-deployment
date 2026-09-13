import tensorflow as tf

def get_mobilenet_v2(input_shape=(224, 224, 3)):
    """
    Returns a baseline MobileNetV2 model pre-trained on ImageNet.
    As per the paper, this is the primary case study for edge inference.
    """
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=True,
        weights='imagenet'
    )
    return model

if __name__ == "__main__":
    model = get_mobilenet_v2()
    model.summary()
