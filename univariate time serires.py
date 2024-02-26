#___________________1. preparation of a univariate series for modeling_____________________________________

# Import necessary libraries
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

# Define a function to split a univariate sequence into samples
def split_sequence(sequence, n_steps): #create a function containg two value 'sequence' and 'n_steps'
    X, y = list(), list()
    for i in range(len(sequence)):  # index of loop = length of sequence
        # find the end of this pattern
        end_ix = i + n_steps

        '''
        for i = 0, end_ix = 0 + 3 = 3
        for i = 1, end_ix = 1 + 3 = 4
        for i = 2, end_ix = 2 + 3 = 5
        for i = 3, end_ix = 2 + 3 = 6
        '''
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        '''
        when end_ix > 9 then
        that is for i = 7, end_ix = 7+3 = 10
        break the loop
        '''
        
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        seq_y = sequence[end_ix]
        
        '''
        for i = 0, seq_x = sequence[0:3] where '3' is excluded
                   seq_y = sequence[3]
                   
        for i = 1, seq_x = sequence[1:4] where '4' is excluded
                   seq_y = sequence[4]
                   
        for i = 2, seq_x = sequence[2:5] where '5' is excluded
                   seq_y = sequence[5]
                   
        for i = 3, seq_x = sequence[3:6] where '6' is excluded
                   seq_y = sequence[6]

        '''
        
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
#print('input before reshaping: \n', raw_seq)

# choose a number of time steps
n_steps = 3

# split into samples
X, y = split_sequence(raw_seq, n_steps)

print('input after reshaping: \n')
# summarize the data
for i in range(len(X)):
    
    print(X[i], y[i])



#__________________2. MLP(Multilayer Perceptrons) Model_______________________________

# Define the neural network model

model = Sequential()    # Create a sequential model
# Add a dense layer with 100 units, ReLU activation, and input dimension of 'n_steps'
model.add(Dense(100, activation='relu', input_dim=n_steps))  
model.add(Dense(1))     # Add a dense layer with 1 unit (output layer)
model.compile(optimizer='adam', loss='mse')    # Compile the model using Adam optimizer and mean squared error loss


# fit model
# Train the model for 2000 epochs using input sequences 'X' and target values 'y'
model.fit(X, y, epochs=2000, verbose=0)


# demonstrate prediction
x_input = array([70, 80, 90])
print('\n input before reshaping: ', x_input)
x_input = x_input.reshape((1, n_steps))   # Reshape the input sequence for prediction
print('\n input after reshaping: ', x_input)
yhat = model.predict(x_input, verbose=0)  # Use the trained model to predict the next element in the sequence
print('\n predicted output: ', yhat)
