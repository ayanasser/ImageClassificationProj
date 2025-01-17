import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

class TrafficLightClassifier:
    def __init__(self, img_size=(32, 32), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self, X_train, X_test, y_train, y_test, validation_split=0.2):
        """Load and preprocess the numpy array data"""
        # Normalize the data
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        print(X_train.shape, X_test.shape)
        
        # Split training data into train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, 
            test_size=validation_split, 
            random_state=42,
            stratify=y_train
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        # Convert labels to categorical
        self.y_train = tf.keras.utils.to_categorical(self.y_train)
        self.y_val = tf.keras.utils.to_categorical(self.y_val)
        self.y_test = tf.keras.utils.to_categorical(self.y_test)
        
        # Create tf.data.Dataset for efficient training
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_train, self.y_train)
        ).shuffle(1000).batch(self.batch_size)
        
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_val, self.y_val)
        ).batch(self.batch_size)
        
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_test, self.y_test)
        ).batch(self.batch_size)
        
        # Data augmentation layer
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.2, 0.2),
            tf.keras.layers.RandomFlip("horizontal")
        ])
        
        return self.y_train.shape[1]  # Return number of classes
        
    def build_model(self, num_classes):
        """Create the model architecture using MobileNetV2 as base"""
        # Base model
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Create the model
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        # Add augmentation layer for training
        x = self.data_augmentation(inputs)
        x = base_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, epochs=50, fine_tune_epochs=20):
        """Train the model"""
        # Callbacks
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        # Initial training
        print("Initial training...")
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Fine-tuning
        print("Fine-tuning...")
        # Unfreeze the top layers of the base model
        self.model.layers[2].trainable = True  # Index changed due to augmentation layer
        for layer in self.model.layers[2].layers[-20:]:
            layer.trainable = True
            
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train again with unfrozen layers
        fine_tune_history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=fine_tune_epochs,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Combine histories
        for key in self.history.history:
            self.history.history[key].extend(fine_tune_history.history[key])
            
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'])
        ax1.plot(self.history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'])
        
        # Plot loss
        ax2.plot(self.history.history['loss'])
        ax2.plot(self.history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'])
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, X):
        """Predict the class of input data"""
        # Ensure the input is normalized
        X = X.astype('float32') / 255.0
        
        # Make prediction
        predictions = self.model.predict(X)
        return predictions

    def evaluate(self):
        """Evaluate the model on test data"""
        return self.model.evaluate(self.test_dataset)
