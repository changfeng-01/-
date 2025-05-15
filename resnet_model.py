import tensorflow as tf
from tensorflow.keras import layers, models

class ResidualBlock(layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", kernel_regularizer=None):
        super(ResidualBlock, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation(activation)
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same", 
                                  kernel_regularizer=kernel_regularizer)
        self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.Activation(activation)
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same",
                                  kernel_regularizer=kernel_regularizer)
        
        # 注意力机制
        self.attention = models.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(filters // 8, activation='relu'),
            layers.Dense(filters, activation='sigmoid'),
            layers.Reshape((1, 1, filters))
        ])
        
        # 捷径连接
        self.shortcut = models.Sequential()
        if strides != 1:
            self.shortcut.add(layers.Conv2D(filters, kernel_size=1, strides=strides,
                                          kernel_regularizer=kernel_regularizer))
            self.shortcut.add(layers.BatchNormalization())
        
    def call(self, inputs, training=None):
        residual = self.shortcut(inputs)
        x = self.bn1(inputs, training=training)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = self.activation2(x)
        x = self.conv2(x)
        
        # 通道注意力
        channel_att = self.attention(x)
        # 空间注意力
        spatial_att = tf.reduce_mean(x, axis=-1, keepdims=True)
        spatial_att = tf.sigmoid(spatial_att)
        
        # 双重注意力融合
        x = x * channel_att * spatial_att
        x = layers.add([x, residual])
        return x


def build_resnet(input_shape=(100, 100, 3), num_classes=111):
    inputs = layers.Input(shape=input_shape)
    
    # 初始卷积层
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    
    # 残差块
    x = ResidualBlock(64)(x)
    x = ResidualBlock(64)(x)
    x = ResidualBlock(128, strides=2)(x)
    x = ResidualBlock(128)(x)
    x = ResidualBlock(256, strides=2)(x)
    x = ResidualBlock(256)(x)
    
    # 特征金字塔结构
    x1 = layers.GlobalAveragePooling2D()(x)
    x2 = layers.GlobalMaxPooling2D()(x)
    x3 = layers.Dense(256, activation='relu')(layers.Flatten()(x))
    
    # BiFPN特征融合
    def bifpn_block(features):
        # 特征归一化
        features = [layers.BatchNormalization()(f) for f in features]
        # 特征交互
        interacted = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                interacted.append(layers.multiply([features[i], features[j]]))
        all_features = features + interacted
        # 联合计算权重
        concat_features = layers.Concatenate()(all_features)
        weights = layers.Softmax()(layers.Dense(len(all_features))(concat_features))
        weights = tf.split(weights, num_or_size_splits=len(all_features), axis=1)
        weighted_features = [layers.multiply([w, f]) for w, f in zip(weights, all_features)]
        return layers.add(weighted_features)
    
    x = bifpn_block([x1, x2, x3])
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # 添加多层感知器(MLP)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    # 创建模型
    model = build_resnet()
    model.summary()
    
    # 动态学习率调度器
    class DynamicLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate=0.001, warmup_steps=1000, min_lr=1e-6):
            super().__init__()
            self.initial_learning_rate = initial_learning_rate
            self.warmup_steps = warmup_steps
            self.min_lr = min_lr
            
        def __call__(self, step):
            # 周期性重启(每10000步)
            cycle = step % 10000
            # 预热阶段
            warmup_lr = self.initial_learning_rate * (cycle / self.warmup_steps)
            # 衰减阶段
            decay_lr = self.initial_learning_rate * tf.math.exp(-0.1 * (cycle - self.warmup_steps))
            # 组合策略
            lr = tf.where(cycle < self.warmup_steps, warmup_lr, decay_lr)
            return tf.maximum(lr, self.min_lr)
            
    lr_schedule = DynamicLearningRate(initial_learning_rate=0.001)
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                 loss="categorical_crossentropy",
                 metrics=["accuracy", "top_k_categorical_accuracy"])
    
    # 训练数据加载
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        directory='./1training',
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        directory='2021_fruits_dataset/1training',
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        subset='validation'
    )
    
    # 训练模型
    print("\n开始训练模型:")
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=[
            # 修改前
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
            # 修改后
            tf.keras.callbacks.ModelCheckpoint('./best_model.h5', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # 测试数据加载和评估
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        directory='./3test',
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # 评估模型
    print("\n评估测试集性能:")
    results = model.evaluate(test_generator)
    print(f"测试集损失: {results[0]:.4f}")
    print(f"测试集准确率: {results[1]*100:.2f}%")