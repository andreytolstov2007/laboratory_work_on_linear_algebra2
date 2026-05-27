import random
import math

class Perceptron:
    def __init__(self, input_dim, init_method='small_random'):
        self.input_dim = input_dim
        self.init_method = init_method
        self.W = None
        self.b = None
        self.Initialize_weights()
   
    def Initialize_weights(self):
        if self.init_method == 'zero':
            self.W = []
            for _ in range(self.input_dim):
                self.W.append(0.0)
            self.b = 0.0
       
        elif self.init_method == 'small_random':
            self.W = []
            for _ in range(self.input_dim):
                self.W.append(random.gauss(0, 0.01))
            self.b = 0.0
       
        elif self.init_method == 'large_random':
            self.W = []
            for _ in range(self.input_dim):
                self.W.append(random.gauss(0, 10))
            self.b = 0.0
       
        else:
            raise ValueError("Unknown init_method. Use 'zero', 'small_random', or 'large_random'")
   
    def Sigmoid(self, z):
        if z >= 0:
            result = 1.0 / (1.0 + math.exp(-z))
            return result
        else:
            exp_z = math.exp(z)
            result = exp_z / (1.0 + exp_z)
            return result
   
    def Forward_single(self, x):
        z = self.b
        for i in range(len(self.W)):
            z = z + self.W[i] * x[i]
        prediction = self.Sigmoid(z)
        return prediction
   
    def Forward(self, X):
        predictions = []
        for i in range(len(X)):
            pred = self.Forward_single(X[i])
            predictions.append(pred)
        return predictions
   
    def Compute_loss(self, y_true, y_pred):
        eps = 1e-8
        n = len(y_true)
        loss_sum = 0.0
       
        for i in range(n):
            yp = y_pred[i]
            if yp < eps:
                yp = eps
            if yp > 1 - eps:
                yp = 1 - eps
           
            if y_true[i] == 1:
                loss_sum = loss_sum - math.log(yp)
            else:
                loss_sum = loss_sum - math.log(1 - yp)
       
        loss = loss_sum / n
        return loss
   
    def Predict_single(self, x):
        prob = self.Forward_single(x)
        if prob >= 0.5:
            return 1
        else:
            return 0
   
    def Predict(self, X):
        predictions = []
        for i in range(len(X)):
            pred = self.Predict_single(X[i])
            predictions.append(pred)
        return predictions
   
    def Compute_gradients(self, X_batch, y_batch, y_pred_batch):
        batch_size = len(X_batch)
        n_features = len(self.W)
        dW = []
        for _ in range(n_features):
            dW.append(0.0)
        db = 0.0
       
        for i in range(batch_size):
            error = y_pred_batch[i] - y_batch[i]
            for j in range(n_features):
                dW[j] = dW[j] + X_batch[i][j] * error
            db = db + error
       
        for j in range(n_features):
            dW[j] = dW[j] / batch_size
        db = db / batch_size
        return dW, db
   
    def Update_with_momentum(self, v_W, v_b, dW, db, lr, momentum):
        new_v_W = []
        new_W = []
        for j in range(len(self.W)):
            v_new = momentum * v_W[j] - lr * dW[j]
            w_new = self.W[j] + v_new
            new_v_W.append(v_new)
            new_W.append(w_new)
       
        v_b_new = momentum * v_b - lr * db
        b_new = self.b + v_b_new
        return new_W, b_new, new_v_W, v_b_new
   
    def Fit(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size, momentum=0.0):
        n_samples = len(X_train)
        v_W = []
        for _ in range(len(self.W)):
            v_W.append(0.0)
        v_b = 0.0
       
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            indices = list(range(n_samples))
            random.shuffle(indices)
           
            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                if end_idx > n_samples:
                    end_idx = n_samples
               
                batch_indices = []
                for i in range(start_idx, end_idx):
                    batch_indices.append(indices[i])
               
                X_batch = []
                y_batch = []
                for i in range(len(batch_indices)):
                    idx = batch_indices[i]
                    X_batch.append(X_train[idx])
                    y_batch.append(y_train[idx])
               
                y_pred_batch = self.Forward(X_batch)
               
                dW, db = self.Compute_gradients(X_batch, y_batch, y_pred_batch)
               
                new_W, new_b, v_W, v_b = self.Update_with_momentum(
                    v_W, v_b, dW, db, lr, momentum
                )
                self.W = new_W
                self.b = new_b
           
            y_train_pred = self.Forward(X_train)
            y_val_pred = self.Forward(X_val)
            train_loss = self.Compute_loss(y_train, y_train_pred)
            val_loss = self.Compute_loss(y_val, y_val_pred)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        return train_losses, val_losses