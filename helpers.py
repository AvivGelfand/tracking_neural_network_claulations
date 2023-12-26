# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Flatten,  Dropout,  Activation, concatenate
from tensorflow.keras.models import Sequential,Model
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec
from matplotlib.gridspec import GridSpec
import itertools
import pickle
import math

def models_generator(layer_sizes_list=None):
    from tensorflow.keras.models import Sequential
    # # Define the sizes of the layers for the neural network models we will train
    # # Each list in the list represents a different model architecture
    if layer_sizes_list is None:
        layer_sizes_list = [[10, 6],[20,10,6],[20,10,10,6],[20,20,10,10,6],[20,20,10,10,10,6],
                        [20,20,10,10,10,6],[20,20,20,10,10,10,6],[20,20,20,10,10,10,6,6]]
    """
    The code block below is designed to systematically construct and train a series of neural network models.
    Each model's architecture is determined by a predefined list of layer sizes, `layer_sizes_list`. This list
    contains tuples or lists, where each element represents the number of neurons in a respective layer of the model.

    For each architecture specified in `layer_sizes_list`, the following steps are executed:
    - A new Sequential model is initialized.
    - The first layer is a Flatten layer to transform input data into a 1D array, with an input shape suitable for 28x28 images.
    - A series of Dense (fully connected) layers are added, with the number of neurons as specified in the layer_sizes tuple.
      Each of these layers uses ReLU (Rectified Linear Unit) as the activation function.
    - The final layer is a Dense layer with 2 neurons, representing a binary classification output, with softmax activation.
    - The model is compiled with the Adam optimizer and the sparse categorical crossentropy loss function, which is appropriate
      for integer-labeled classification tasks.
    - The model is trained on the training data for 5 epochs and validated on the test data.
    - After training, the model is saved to a file named after its architecture configuration for later use or analysis.
     This automated process facilitates the examination and comparison of different model architectures' performance.
    """
    # Assuming layer_sizes_list is predefined and contains the architecture specifications
    # Example: layer_sizes_list = [[64, 32], [128, 64, 32]]

    for layer_sizes in layer_sizes_list:
        # Initialize a Sequential model
        model = Sequential()
        # Flatten the input data from 28x28 to a 1D array of 784 elements
        model.add(Flatten(input_shape=(28, 28)))
        
        # Add Dense layers as specified in the layer_sizes tuple
        for i, layer_size in enumerate(layer_sizes):
            # Each Dense layer has 'layer_size' neurons and ReLU activation
            model.add(Dense(layer_size, activation='relu', name=f'dense_layer{i+1}'))
        
        # Add the output layer with 2 neurons for binary classification and softmax activation
        model.add(Dense(2, activation='softmax', name='output_layer'))
        
        # Compile the model with Adam optimizer and sparse categorical crossentropy loss
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train the model on the training dataset
        model.fit(x_train_rand_noisy, y_train, epochs=5, validation_data=(x_test, y_test))
        
        # Save the trained model to a file
        model.save(f'model_n_{layer_sizes}.keras')
        print(f"Model with layer sizes {layer_sizes} saved to disk.")

#random_noise
def add_random_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images

    # Add noise to both the training and testing datasets
    # noise_factor = 0.5
    # x_train_rand_noisy = add_random_noise(x_train, noise_factor=noise_factor)
    # x_test_rand_noisy = add_random_noise(x_test, noise_factor=noise_factor)

# Function to add gridlines to an image
def add_gridlines(image, grid_size):
    """
    Add gridlines to an image at each grid_size interval.

    :param image: A 2D numpy array representing the image.
    :param grid_size: The interval at which to add gridlines.
    :return: A 2D numpy array representing the image with gridlines.

    # # Example usage: Add gridlines to the first image in the MNIST training set
    # from tensorflow.keras.datasets import mnist
    # # Load MNIST dataset
    # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # # Preprocess the data:
    # # We will only use two classes to simplify the problem
    # train_images = train_images[train_labels < 2]
    # train_labels = train_labels[train_labels < 2]
    # test_images = test_images[test_labels < 2]
    # test_labels = test_labels[test_labels < 2]

    # # Normalize the data to be between 0 and 1
    # x_train, x_test = x_train / 255.0, x_test / 255.0  

    # grid_size = 5  # Define the size of the grid
    # image_index = 1  # Index of the image to which we want to add gridlines

    # # Original image
    # original_image = train_images[image_index]

    # # Image with gridlines
    # image_with_gridlines = add_gridlines(original_image, grid_size)

    # # Plot the original image and the image with gridlines
    # plt.figure(figsize=(10, 5))

    # # Plot original image
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_image, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')

    # # Plot image with gridlines
    # plt.subplot(1, 2, 2)
    # plt.imshow(image_with_gridlines, cmap='gray')
    # plt.title(f'Image with {grid_size}x{grid_size} Gridlines')
    # plt.axis('off')

    # plt.show()


    # def add_gridlines_to_dataset(images, grid_size, noramlized=True):
    #     \"""
    #     Add gridlines to all images in an MNIST dataset.

    #     Parameters:
    #     images (numpy array): An array of images to which gridlines will be added.
    #     grid_size (int): The size of the grid.

    #     Returns:
    #     numpy array: A new array containing the images with added gridlines.
    #     \"""
    #     # Initialize a new array with the same shape as the input images
    #     images_with_grid = np.zeros_like(images)
    #     for i, image in enumerate(images):
    #         # Copy the image to avoid changing the original
    #         image_with_grid = np.copy(image)
            
    #         # Set every grid_size-th pixel to white (value 255)
    #         image_with_grid[::grid_size, :] = 255
    #         image_with_grid[:, ::grid_size] = 255
    #         images_with_grid[i] = image_with_grid
    #     return images_with_grid

    # # Usage:
    # # Assuming train_images and test_images are loaded MNIST datasets
    # grid_size = 2
    # gl_train_images = add_gridlines_to_dataset(train_images, grid_size)
    # gl__test_images = add_gridlines_to_dataset(test_images, grid_size)

    # # Now new_train_images and new_test_images contain the images with gridlines

    # compare_model_calcs(model1,model2, x_test, y_test, gl__test_images, n=10, ind=ind)

    """
    image_with_grid = np.copy(image)
    # Add vertical gridlines
    image_with_grid[:, ::grid_size] = 255
    # Add horizontal gridlines
    image_with_grid[::grid_size, :] = 255
    
    return image_with_grid

# Function to add noisy corners to images
def add_noisy_corners(images, noise_level=2):
    
    corner_size = images.shape[1] // 4  # Size of the corner
    images_with_noisy_corners = np.copy(images)
    
    for i in range(images.shape[0]):
        # Generate noise
        noise = np.random.rand(corner_size, corner_size)
        # Apply noise to the corners
        images_with_noisy_corners[i, :corner_size, :corner_size] = noise
        images_with_noisy_corners[i, -corner_size:, :corner_size] = noise
        images_with_noisy_corners[i, :corner_size, -corner_size:] = noise
        images_with_noisy_corners[i, -corner_size:, -corner_size:] = noise

    return images_with_noisy_corners

def add_noisy_pixel_to_mnist(mnist_images, index, noise_intensity=1):
    """
    Adds a noisy pixel to all images in the MNIST dataset.

    Parameters:
    mnist_images (numpy.ndarray): Array of MNIST images.
    index (tuple): Index of the pixel to be modified (row, column).
    noise_intensity (int): Intensity of the noise to be added (default 255 for white).

    Returns:
    numpy.ndarray: Modified MNIST images with added noisy pixel.
    """
    # Check if the index is within the bounds of the image
    if index[0] < 0 or index[0] >= mnist_images.shape[1] or index[1] < 0 or index[1] >= mnist_images.shape[2]:
        raise ValueError("Index is out of bounds for the MNIST images.")

    # Add a noisy pixel to each image
    modified_images = mnist_images.copy()
    for img in modified_images:
        img[index] = noise_intensity

    return modified_images

def save_object(obj, filename):
    """
    Save a Python object to a file using pickle.
    
    :param obj: The Python object to save.
    :param filename: The name of the file where the object will be saved.
    """
    import pickle

    with open(filename, 'wb') as output_file:
        pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)
    # Example usage
    # obj = ... # This is your Python object, e.g., the output of the get_layer_outputs function
    # save_object(obj, 'my_object.pkl')

def load_object(filename):
    import pickle
    
    """
    Load a Python object from a file using pickle.
    
    :param filename: The name of the file from which to load the object.
    :return: The loaded Python object.
    """
    with open(filename, 'rb') as input_file:
        return pickle.load(input_file)

    # Example usage
    # obj = ... # This is your Python object, e.g., the output of the get_layer_outputs function
    # save_object(obj, 'my_object.pkl')
    # To load the object back
    # loaded_obj = load_object('my_object.pkl')

def display_digit(x, y, digit=None ,sample=0):
    """
    This function displays a digit from the dataset.

    Parameters:
    x: numpy array of image data
    y: numpy array of labels
    digit: specific digit to display. If None, it will display the image at the index specified by 'sample'
    sample: index of the sample to display if 'digit' is None

    Returns:
    plt: matplotlib object with the image displayed
    """
    
    # Example usage of the function:
    # plt.show(display_digit(x_train, y_train, digit=0, sample=0))
    # plt.show(display_digit(x_train, y_train, digit=1, sample=0))
    # plt.show(display_digit(x_train_rand_noisy, y_train, sample=2))

    import matplotlib.pyplot as plt
    
    # If a specific digit is specified, display an image of that digit
    if digit!=None:
        plt.imshow(x[y == digit][sample], cmap='gray')
    # If no specific digit is specified, display the image at the index specified by 'sample'
    else:
        plt.imshow(x[sample], cmap='gray')
    
    # Set the title of the plot to the label of the image
    plt.title(f'Digit: {digit}')

    # Remove the axes of the plot
    plt.axis('off')

    # Return the matplotlib object
    return plt

def get_subset_sizes(model, ind=1):
    """
    This function takes a trained neural network model as input and returns a list 
    containing the number of nodes in each layer of the network.

    Parameters:
    model: A trained neural network model.
    ind: The index from which to start considering the layers. Default is 1, 
         which means the first (input) layer is ignored.

    Returns:
    layer_sizes: A list of integers where each integer represents the number of nodes in a layer.
    # Example usage:
    # Load a pre-trained model
    # model = tf.keras.models.load_model('models/model_[10, 6].keras')
    # Get the sizes of the layers in the model, ignoring the input layer
    # subset_sizes = get_subset_sizes(model, ind=1)
    # Print the sizes of the layers
    # print(subset_sizes)  # Output might look like: [10, 6, 2]
    """


    # Initialize an empty list to store the sizes of each layer
    layer_sizes = []

    # Iterate through each layer of the model starting from the index 'ind'
    for layer in model.layers[ind:]:
        # Get the output shape of the layer which contains the size of the layer
        # The output shape is typically in the form of (None, size) where None is the batch dimension.
        # We are interested in the size (number of nodes) which is the second element of the tuple.
        layer_size = layer.output_shape[1]

        # Append the size of the current layer to the list
        layer_sizes.append(layer_size)

    # Return the list of layer sizes
    return layer_sizes

def get_all_layer_outputs(model, data_sample, load=False):
    """
    Retrieves the outputs for each layer of a given TensorFlow Keras model for a single input data sample.
    
    Parameters:
    - model: A `tf.keras.Model` object or a string representing the path to the model file if `load` is True.
    - data_sample: A numpy array representing the input data sample. Should match the model's input shape.
    - load: A boolean flag indicating whether to load the model from the given path (True) or use the provided model directly (False).
    
    Returns:
    - A list of numpy arrays where each array corresponds to the output of a layer in the model.
    
    This function can be used for model debugging, visualization, or for advanced model analysis purposes.

    # Example usage:
    # model_outputs = get_all_layer_outputs(trained_model, test_data_sample)
    # loaded_model_outputs = get_all_layer_outputs("model_path.keras", test_data_sample, load=True)
    """

    # Load the model from disk if required.
    if load:
        model = tf.keras.models.load_model(model)
    
    # If not loading the model, create a new multi-output model to get the outputs from all layers.
    if not load:
        # Extract the outputs for all layers except the input layer.
        layer_outputs = [layer.output for layer in model.layers[1:]]  # Skip the input layer
        
        # Construct a new model that will return the outputs of all layers.
        multi_output_model = Model(inputs=model.input, outputs=layer_outputs)
        # Consider whether the model should be saved here, if needed for later analysis or debugging.

    # Prepare the input data sample if it's not already in the right shape.
    if len(data_sample.shape) == len(model.input_shape) - 1:
        data_sample = np.expand_dims(data_sample, axis=0)
    
    # Predict the outputs using the model for the given data sample.
    all_outputs = multi_output_model.predict(data_sample, verbose=0)

    return all_outputs

def multilayered_graph(*subset_sizes):
    """
    This function creates a multilayered graph where each layer corresponds to a subset of nodes.
    The size of each subset is defined by the 'subset_sizes' argument.

    Parameters:
    *subset_sizes: Variable length argument list of integers where each integer represents the size of a subset (layer).

    Returns:
    G: A NetworkX graph object representing the multilayered graph.
    """
    # Calculate the extents of each subset_size to define layers
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    
    # Create a list of ranges representing each layer
    layers = [sorted(range(start, end)) for start, end in extents]  # Sort each layer range for safety
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes to the graph with a layer attribute in a sorted manner
    for i, layer in enumerate(layers):
        G.add_nodes_from(sorted(layer), layer=i)  # Sort again for safety
    
    # Add edges between successive layers
    for layer1, layer2 in nx.utils.pairwise(layers):
        # Use sorted to ensure consistent order
        G.add_edges_from(sorted(itertools.product(layer1, layer2)))
    
    # Return the created graph
    return G

def visualize_multilayer_graph_outputs(model, test_sample, subset_color=None, ind=1):
    """
    Visualizes the outputs of a multilayered graph neural network model.

    Parameters:
    - model_filename: the filename of the model to load.
    - test_sample: the input sample to test (should be reshaped accordingly).
    - subset_sizes: a list indicating the size of each layer's subset in the graph.
    - subset_color: a dictionary mapping the layer index to its color.
    - ind: the index of the layers to start visualizing from. Default is 0.
    example usage:
    # plt.figure(figsize=(max(subset_sizes)*0.5, max(subset_sizes)*0.5))
    # plt.show(visualize_multilayer_graph_outputs(model, x_test[0], subset_color, ind=ind))
    """
    subset_sizes = get_subset_sizes(model)
    G = multilayered_graph(*subset_sizes)
    # print(subset_sizes) # Output might look like: [10, 6, 2]
    import matplotlib.pyplot as plt
    import networkx as nx
    outputs = get_all_layer_outputs(model, test_sample.reshape(1, 28, 28),load=False)
    # print("model outputs",outputs)
    
    if subset_color is None:
        subset_color = ["gold", "violet", "limegreen", "darkorange", "gold", "violet", "limegreen", "darkorange", "gold", "violet", "limegreen", "darkorange", "gold", "violet", "limegreen", "darkorange", "gold", "violet", "limegreen", "darkorange", ]

    # Assign colors to nodes based on their layer
    color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
    
    # Position nodes using the multipartite layout
    pos = nx.multipartite_layout(G, subset_key="layer")
    
    # Draw the graph
    nx.draw(G, pos, node_color=color, edge_color="lightgray",
            with_labels=False )
    # Annotate the graph with the output values
    for layer in range(len(outputs)):
        for node in range(subset_sizes[layer]):
            node_pos = sum(subset_sizes[:layer]) + node
            if node_pos in pos:
                if outputs[layer][0][node] < 0.01:
                    plt.text(pos[node_pos][0], pos[node_pos][1], 
                        str(round(outputs[layer][0][node], 5)), fontsize=12, ha='center', va='center', color='red', fontweight='bold')
                else:
                    plt.text(pos[node_pos][0], pos[node_pos][1], 
                        str(round(outputs[layer][0][node], 3)), fontsize=10, ha='center', va='center')
            else:
                print(f"Key error avoided: {node_pos} is not in pos")
    return plt

def visualize_average_multilayer_graph_outputs(model, test_samples, subset_color=None ,outputs=None,red=True):
    """
    TODO: documantaion
    example usage:
    # model0 = tf.keras.models.load_model('models/model_n_[10, 6].keras')
    # visualize_average_multilayer_graph_outputs(model0, x_test, subset_color, ind=1)
    """
    subset_sizes = get_subset_sizes(model)
    # print(subset_sizes)
    G = multilayered_graph(*subset_sizes)
    import matplotlib.pyplot as plt
    import networkx as nx
    
    if outputs is None:
        outputs = get_avg_outputs(model, test_samples)

    # Assign colors to nodes based on their layer
    # color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
    if subset_color is None:
        subset_color = ["gold", "violet", "limegreen", "darkorange", "gold", "violet", "limegreen", "darkorange", "gold", "violet", "limegreen", "darkorange", "gold", "violet", "limegreen", "darkorange", "gold", "violet", "limegreen", "darkorange", ]

    # Assign colors to nodes based on their layer
    color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
    
    # Position nodes using the multipartite layout
    pos = nx.multipartite_layout(G, subset_key="layer")
    
    # Draw the graph
    nx.draw(G, pos, node_color=color, edge_color="lightgray",
            with_labels=False )
    # Annotate the graph with the output values
    for layer in range(len(outputs)):
        for node in range(subset_sizes[layer]):
            node_pos = sum(subset_sizes[:layer]) + node
            if node_pos in pos:
                if outputs[layer][node] < 0.1 and red:
                    plt.text(pos[node_pos][0], pos[node_pos][1], 
                        str(round(outputs[layer][node], 4)), fontsize=12, ha='center', va='center', color='red', fontweight='bold')
                else:
                    plt.text(pos[node_pos][0], pos[node_pos][1], 
                        str(round(outputs[layer][node], 4)), fontsize=10, ha='center', va='center')
            else:
                print(f"Key error avoided: {node_pos} is not in pos")
    return plt

def get_avg_outputs(model, data):
    # Creating a model that will return the outputs for each layer
    layer_outputs = [layer.output for layer in model.layers[1:]]
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    # Storing the sum of outputs of each layer
    sum_of_outputs = [np.zeros((layer.output_shape[1],)) for layer in model.layers[1:]]

    # Number of data points
    num_data = len(data)

    # Iterating over each data point
    for i in range(num_data):
        # Predicting the intermediate outputs
        intermediate_output = intermediate_model.predict(data[i:i+1],verbose=0)

        # Adding the outputs to the sum_of_outputs
        for j, output in enumerate(intermediate_output):
            sum_of_outputs[j] += np.mean(output, axis=0) # Assuming the outputs are 2D (batch_size, features)

    # Averaging the outputs
    average_outputs = [sum_layer / num_data for sum_layer in sum_of_outputs]

    return average_outputs

def calculate_difference(obj1, obj2):
    """
    Calculate the difference between corresponding elements of two objects.
    Both objects should be of the same shape and contain arrays.

    :param obj1: The first object containing numpy arrays.
    :param obj2: The second object containing numpy arrays.
    :return: An object of the same shape with calculated differences.
    """
    if len(obj1) != len(obj2):
        raise ValueError("Objects must have the same length")

    result = []
    for arr1, arr2 in zip(obj1, obj2):
        if arr1.shape != arr2.shape:
            raise ValueError("All corresponding arrays must have the same shape")

        # Calculating the difference
        difference = arr1 - arr2
        result.append(difference)

    return result

def compare_model_calcs(model1,model2, X, y, X_noisy, n=1, ind=1):
    for i in range(n):
        subset_sizes1 = get_subset_sizes(model1)
        subset_sizes2 = get_subset_sizes(model2)
        plt.figure(figsize=(max(max(subset_sizes1),max(subset_sizes2), 15), max(max(subset_sizes1)*2,max(subset_sizes2)*2, 6)))
        
        # Create a GridSpec object with 1 row and 3 columns,
        # and set the width ratio for the columns
        import matplotlib.gridspec
        gs = GridSpec(4, 4, width_ratios=[1, 3, 1, 0])  # The last column will not be used

        plt.subplot(gs[0, 0])
        display_digit(X, y, digit=y[i], sample=i)
        plt.subplot(gs[0, 1:4])
        visualize_multilayer_graph_outputs(model1, X[i],  ind=ind)
        plt.subplot(gs[1, 0])
        display_digit(X_noisy, y, digit=y[i], sample=i)
        plt.subplot(gs[1, 1:4])
        visualize_multilayer_graph_outputs(model1, X_noisy[i],  ind=ind)
        plt.subplot(gs[2, 0])
        display_digit(X, y, digit=y[i], sample=i)
        plt.subplot(gs[2, 1:4])
        visualize_multilayer_graph_outputs(model2, X[i], ind=ind)
        plt.subplot(gs[3, 0])
        display_digit(X_noisy, y, digit=y[i], sample=i)
        plt.subplot(gs[3, 1:4])
        visualize_multilayer_graph_outputs(model2, X_noisy[i], ind=ind)
     
        plt.show()

def compare_model_avgs(model1, X, y, X_noisy, outputs1=None, outputs2=None, outputs3=None,digit=1, indexes=None):
    """
    Compare model averages with visualizations.
    :param model1: First model for comparison.
    :param model2: Second model for comparison.
    :param X: Input data.
    :param y: Labels for input data.
    :param X_noisy: Noisy input data.
    :param outputs1: Outputs from model1 (optional).
    :param outputs2: Outputs from model2 (optional).
    :param outputs3: Outputs for comparison (optional).
    """
    subset_sizes1 = get_subset_sizes(model1)
    # subset_sizes2 = get_subset_sizes(model2)
    plt.figure(figsize=(max(max(subset_sizes1), 15), max(max(subset_sizes1)*1.8, 6)))

    # Create a GridSpec object with 1 row and 3 columns, and set the width ratio for the columns
    import matplotlib.gridspec

    gs = GridSpec(3, 4, width_ratios=[1, 3, 1, 0])  # The last column will not be used

    # First subplots
    plt.subplot(gs[0, 0])
    display_digit(X, y, digit=digit, sample=1)
    plt.title("Example of a Digit")

    # Second subplot
    plt.subplot(gs[0, 1:4])
    visualize_average_multilayer_graph_outputs(model1, X[y == digit], outputs=outputs1)
    plt.title(f"Average Outputs for Model of Shape:{subset_sizes1}\n No Noise")

    # Third subplot
    plt.subplot(gs[1, 0])
    display_digit(X_noisy, y, digit=digit, sample=1)
    plt.title("Example of a Noisy Digit")

    # Fourth subplot
    plt.subplot(gs[1, 1:4])
    visualize_average_multilayer_graph_outputs(model1, X[y == digit], outputs=outputs2)
    plt.title(f"Average Outputs for Model of Shape:{subset_sizes1} \n With Noisy Pixel {indexes}")

    # Fifth subplot
    plt.subplot(gs[2, 0])
    display_digit(X_noisy, y, digit=digit, sample=1)
    plt.title("Example of a Noisy Digit")

    # Sixth subplot
    plt.subplot(gs[2, 1:4])
    visualize_average_multilayer_graph_outputs(model1, X[y == digit], outputs=outputs3, red=False)
    plt.title(f"Residual Network = Clean - Noisy Averages\n With Noisy Pixel {indexes}")
    
    # plt.show()
    # Directory where you want to save the images
    # os.makedirs(save_dir, exist_ok=True)
    import os
    # save_dir = '1-poc-watch-nn-wheights\images'
    save_dir = 'G:\My Drive\Work\Ariel Jaffe\images'
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f"compare_model_avgs_{subset_sizes1}_{digit}_{indexes}.png")
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free up memory


def plot_dense_layer_heatmaps(model, input_shape=(28, 28)):
    
    # print("The following image represents the first layer of a model with the following layer sizes (number of nodes)",get_subset_sizes(model))
        # Example usage with a model
    # model = ... # Your Keras model
    # model1 = tf.keras.models.load_model('models/model_n_[10, 6].keras')
    # model2 = tf.keras.models.load_model('models/model_n_[20, 20, 20, 10, 10, 10, 6].keras')

    # plot_dense_layer_heatmaps(model1)
    # plot_dense_layer_heatmaps(model2)


    # Find the first dense layer
    for layer in model.layers:
        if 'dense' in layer.name:
            first_dense_layer = layer
            break

    # Extract weights and biases
    weights, biases = first_dense_layer.get_weights()

    # Number of nodes in the layer
    num_nodes = weights.shape[1]

    # Determine grid size for subplots
    grid_size = math.ceil(math.sqrt(num_nodes)) # That is kinda cool

    # Create a figure for subplots
    plt.figure(figsize=(grid_size * 5, grid_size * 5))
    # main title
    plt.suptitle(f"Heatmaps for the first layer of the model, layer sizes: {get_subset_sizes(model)[:-1]}", fontsize=20, y=0.95)
    # Reshape weights and plot heatmaps in subplots
    for i in range(num_nodes):
        # Reshape weights to match the input shape
        node_weights = weights[:, i].reshape(input_shape)

        # Add subplot for each node
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(node_weights, cmap='viridis')
        plt.title(f"Node {i+1}")
        plt.colorbar()

    # Display the figure
    plt.tight_layout( rect=[0, 0.03, 1, 0.95])
    plt.show()


def add_noisy_pixel_to_all_mnist(mnist_images, noise_intensity=1):
    """
    Adds a noisy pixel to all images in the MNIST dataset for each pixel position.

    Parameters:
    mnist_images (numpy.ndarray): Array of MNIST images.
    noise_intensity (int): Intensity of the noise to be added (default 255 for white).

    Returns:
    dict: Dictionary of modified MNIST images with added noisy pixel for each pixel position.
    # Example usage
    # Load your MNIST dataset here (should be a numpy array)
    # mnist_images = ... 
    # noisy_datasets = add_noisy_pixel_to_all_mnist(mnist_images)
    # Access a specific dataset with noise at a particular pixel, e.g., (0, 0)
    # x_test_pixel00 = noisy_datasets[(0, 0)]
    """
    # Dimensions of the MNIST images
    rows, cols = mnist_images.shape[1], mnist_images.shape[2]

    # Dictionary to store modified images for each pixel position
    modified_images_dict = {}

    for row in range(rows):
        for col in range(cols):
            # Create a copy of the images for modification
            modified_images = mnist_images.copy()

            # Add a noisy pixel at the current position for each image
            for img in modified_images:
                img[row, col] = noise_intensity

            # Add the modified images to the dictionary
            modified_images_dict[(row, col)] = modified_images
    return modified_images_dict













