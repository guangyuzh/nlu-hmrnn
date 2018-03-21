from hmlstm import HMLSTMNetwork, convert_to_batches, plot_indicators

network = HMLSTMNetwork(input_size=1, task='regression', hidden_state_sizes=30,
                       embed_size=50, out_hidden_size=30, num_layers=2)
                       
# generate signals
num_signals = 300
signal_length = 400
x = np.linspace(0, 50 * np.pi, signal_length)
signals = [np.random.normal(0, .5, size=signal_length) +
           (2 * np.sin(.6 * x + np.random.random() * 10)) +
           (5 * np.sin(.1* x + np.random.random() * 10))
    for _ in range(num_signals)] 
    
batches_in, batches_out = convert_to_batches(signals, batch_size=10, steps_ahead=3)


network.train(batches_in[:-1], batches_out[:-1], save_vars_to_disk=True, 
              load_vars_from_disk=False, variable_path='./sinusoidal')

predictions = network.predict(batches_in[-1], variable_path='./sinusoidal')
boundaries = network.predict_boundaries(batches_in[-1], variable_path='./sinusoidal')

# visualize boundaries
plot_indicators(batches_out[-1][0], predictions[0], indicators=boundaries[0])
