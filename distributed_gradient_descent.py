import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp 

# Create synthetic data
np.random.seed(42)
X = 2 * np.random.rand(1000, 1) # 1000 samples, 1 feature
y = 4 + 3 * X + np.random.randn(1000, 1) * 0.1 # y = 2*x + noise

# Visualize the data
plt.scatter(X, y, c='blue', label='Data points') 
plt.xlabel('X') 
plt.ylabel('y') 
plt.title('Sample Dataset for Linear Regression') 
plt.show() 

def compute_mse(X, y, w, b):
  m = len(y)
  predictions = X.dot(w)+b
  mse = (1/m)*np.sum((predictions-y)**2)
  return mse

def compute_gradients(X, y, w, b):
  m = len(y)
  predictions = X.dot(w)+b
  dw = (2/m)*np.dot(X.T, (predictions-y))
  db = (2/m)*np.sum(predictions-y)
  return dw, db

# Function to be run by each process 
def gradient_descent_process(rank, X_split, y_split, w, b, result_queue): 
  dw, db = compute_gradients(X_split, y_split, w, b)  # Compute gradients on local data 
  result_queue.put((dw, db))  # Place gradients in result queue 
 
# Function to run distributed gradient descent 
def distributed_gradient_descent(X, y, learning_rate=0.01, epochs=100, num_processes=4): 
  # Split the data across different processes 
  data_split = np.array_split(X, num_processes) 
  target_split = np.array_split(y, num_processes) 
     
  # Initialize parameters (weights and bias) 
  w = np.zeros((X.shape[1], 1))  # Initialize weight to zero 
  b = np.zeros((1, 1))  # Initialize bias to zero 
     
  # Queue for collecting results from processes 
  result_queue = mp.Queue() 
 
  # Run gradient descent 
  for epoch in range(epochs): 
    processes = [] 
         
    # Start multiple processes for gradient computation 
    for rank in range(num_processes): 
      p = mp.Process(target=gradient_descent_process, args=(rank, data_split[rank], 
                                                            target_split[rank], w, b, result_queue)) 
      processes.append(p) 
      p.start() 
 
      # Collect results from all processes 
      dw_total = np.zeros_like(w) 
      db_total = np.zeros_like(b) 
      for p in processes: 
        p.join()  # Wait for all processes to finish 
      while not result_queue.empty(): 
        dw, db = result_queue.get() 
        dw_total += dw 
        db_total += db 

      # Update weights and bias 
      w -= learning_rate * dw_total 
      b -= learning_rate * db_total 
      # Print the progress 
      if epoch % 10 == 0: 
        mse = compute_mse(X, y, w, b) 
        print(f"Epoch {epoch}: MSE = {mse}") 

  return w, b 

# Run distributed gradient descent 
w_final, b_final = distributed_gradient_descent(X, y) 

# Print the final parameters 
print("Final weights:", w_final) 
print("Final bias:", b_final) 