namespace NeuralNet;

public class SimpleNeuralNetwork
{
    private NeuralLayerFactory _layerFactory;

    public List<NeuralLayer> _layers;
    public double _learningRate;
    public double[][] _expectedResult = Array.Empty<double[]>();

    /// <summary>
    /// Constructor of the Neural Network.
    /// Note:
    /// Initialy input layer with defined number of inputs will be created.
    /// </summary>
    /// <param name="numberOfInputNeurons">
    /// Number of neurons in input layer.
    /// </param>
    public SimpleNeuralNetwork(int numberOfInputNeurons, double learning_rate = 0.5)
    {
        _layers = new List<NeuralLayer>();
        _layerFactory = new NeuralLayerFactory();

        // Create input layer that will collect inputs.
        CreateInputLayer(numberOfInputNeurons);

        _learningRate = learning_rate;
    }

    /// <summary>
    /// Add layer to the neural network.
    /// Layer will automatically be added as the output layer to the last layer in the neural network.
    /// </summary>
    public void AddLayer(NeuralLayer newLayer)
    {
        if (_layers.Any())
        {
            var lastLayer = _layers.Last();
            newLayer.ConnectLayers(lastLayer);
        }

        _layers.Add(newLayer);
    }

    /// <summary>
    /// Push input values to the neural network.
    /// </summary>
    public void PushInputValues(double[] inputs)
    {
        Console.WriteLine($"[SimpleNeuralNetwork.PushInputValues] (inputs length: {inputs.Length})");
        _layers.First().Neurons.ForEach(x => x.PushValueOnInput(inputs[_layers.First().Neurons.IndexOf(x)]));
        Console.WriteLine($"[SimpleNeuralNetwork.PushInputValues] ...execution completed...");
    }

    /// <summary>
    /// Set expected values for the outputs.
    /// </summary>
    public void PushExpectedValues(double[][] expectedOutputs)
    {
        _expectedResult = expectedOutputs;
    }

    /// <summary>
    /// Calculate output of the neural network.
    /// </summary>
    /// <returns></returns>
    public List<double> GetOutput()
    {
        Console.WriteLine($"[SimpleNeuralNetwork.GetOutput] ()");
        var returnValue = new List<double>();

        _layers.Last().Neurons.ForEach(neuron =>
        {
             returnValue.Add(neuron.CalculateOutput());
        });

        Console.WriteLine($"[SimpleNeuralNetwork.GetOutput] ...execution completed...");
        return returnValue;
    }

    /// <summary>
    /// Train neural network.
    /// </summary>
    /// <param name="inputs">Input values.</param>
    /// <param name="numberOfEpochs">Number of epochs.</param>
    public void Train(double[][] inputs, double learning_rate, int numberOfEpochs)
    {
        _learningRate = learning_rate;

        double totalError = 0;

        for(int i = 0; i < numberOfEpochs; i++)
        {
            for(int j = 0; j < inputs.GetLength(0); j ++)
            {
                PushInputValues(inputs[j]);

                var outputs = new List<double>();

                // Get outputs.
                _layers.Last().Neurons.ForEach(x =>
                {
                    outputs.Add(x.CalculateOutput());
                });

                // Calculate error by summing errors on all output neurons.
                totalError = CalculateTotalError(outputs, j);
                HandleOutputLayer(j);
                HandleHiddenLayers();
            }
        }
    }

    /// <summary>
    /// Train neural network with MNIST data set.
    /// </summary>
    /// <param name="train_set">Input values.</param>
    /// <param name="expected_results">Dictionary of labels and expected result for label</param>
    /// <param name="learning_rate">the learning rate</param>
    /// <param name="numberOfEpochs">Number of epochs.</param>
    public void Train_MNIST((string label, double[] standardized_pixel_values)[] train_set, 
        Dictionary<string, List<double>> expected_results,
        double learning_rate, 
        int numberOfEpochs)
    {
        Console.WriteLine($"[SimpleNeuralNetwork.Train_MNIST] (train_set size:{train_set.Length}, learning_rate: {learning_rate} epochs: {numberOfEpochs})");

        _learningRate = learning_rate;

        for(int i = 0; i < numberOfEpochs; i++)
        {
            //Console.WriteLine($"[SimpleNeuralNetwork.Train_MNIST] processing 'epoch' {i} / {numberOfEpochs}");
            for(int j = 0; j < train_set.Length; j ++)
            {
                PushInputValues(train_set[j].standardized_pixel_values);  
                // Get outputs.
                var outputs = new List<double>();

                _layers.Last().Neurons.ForEach(x =>
                {
                    outputs.Add(x.CalculateOutput());
                });

                string output_values_str = string.Join(", ", outputs.Select(p => $"{p:f3}"));

                Console.WriteLine($"[SimpleNeuralNetwork.Train_MNIST] label: '{train_set[j].label}', training results: {output_values_str}");

                HandleOutputLayer_MNIST(train_set[j].label, expected_results);
                HandleHiddenLayers();

            }
        }

        Console.WriteLine($"[SimpleNeuralNetwork.Train_MNIST] ...execution completed...");
    }


    /// <summary>
    /// Hellper function that creates input layer of the neural network.
    /// </summary>
    private void CreateInputLayer(int numberOfInputNeurons)
    {
        var inputLayer = _layerFactory.CreateNeuralLayer(numberOfInputNeurons, new RectifiedActivationFuncion(), new WeightedSumFunction());
        inputLayer.Neurons.ForEach(x => x.AddInputSynapse(0));
        this.AddLayer(inputLayer);
    }

    /// <summary>
    /// Hellper function that calculates the error at each output node.
    /// </summary>
    private List<double> CalculateOutputErrors(List<double> outputs, string label, Dictionary<string, List<double>> expected_results)
    {
        List<double> errors = [];

        for (int i = 0; i < outputs.Count; i++)
        {
            errors.Add(outputs[i]-expected_results[label][i]);
        }

        return errors;
    }

    /// <summary>
    /// Hellper function that calculates total error of the neural network.
    /// </summary>
    private double CalculateTotalError(List<double> outputs, int row)
    {
        double totalError = 0;

        outputs.ForEach(output =>
        {
            var error = Math.Pow(output - _expectedResult[row][outputs.IndexOf(output)], 2);
            totalError += error;
        });

        return totalError;
    }

    /// <summary>
    /// Hellper function that runs backpropagation algorithm on the output layer of the network.
    /// </summary>
    /// <param name="row">
    /// Input/Expected output row.
    /// </param>
    private void HandleOutputLayer_MNIST(string label, Dictionary<string, List<double>> expected_results)
    {
        //Console.WriteLine($"[SimpleNeuralNetwork.HandleOutputLayer_MNIST] (label: {label})");
        //Console.WriteLine($"[SimpleNeuralNetwork.HandleOutputLayer_MNIST] Output nodes count: {_layers.Last().Neurons.Count}");

        INeuron[] output_neurons = _layers.Last().Neurons.ToArray();
        for (int i = 0; i < _layers.Last().Neurons.Count; i++)
        {
            var neuron = output_neurons[i];
            double output = neuron.CalculateOutput();
            double expectedOutput = expected_results[label][i];
            foreach (var connection in neuron.Inputs)
            {
                double netInput = connection.GetOutput();
                //double nodeDelta = (expectedOutput - output) * output * (1.0 - output);
                double nodeDelta = (expectedOutput - output) / output;
                double delta = -1.0 * netInput * nodeDelta;

                connection.UpdateWeight(_learningRate, delta);

                neuron.PreviousPartialDerivate = nodeDelta;
                //Console.WriteLine($"[SimpleNeuralNetwork.HandleOutputLayer_MNIST] label: {label}, Output Node: number: {i},  net input: {netInput},  output value: {output}, expected output: {expectedOutput}, nodeDelta: {nodeDelta}, delta: {delta}, learning rate: {_learningRate}");
            }
        }

        //Console.WriteLine($"[SimpleNeuralNetwork.HandleOutputLayer_MNIST] ...execution completed...");
    }

    /// <summary>
    /// Hellper function that runs backpropagation algorithm on the output layer of the network.
    /// </summary>
    /// <param name="row">
    /// Input/Expected output row.
    /// </param>
    private void HandleOutputLayer(int row)
    {
        _layers.Last().Neurons.ForEach(neuron =>
        {
            neuron.Inputs.ForEach(connection =>
            {
                var output = neuron.CalculateOutput();
                var netInput = connection.GetOutput();

                var expectedOutput = _expectedResult[row][_layers.Last().Neurons.IndexOf(neuron)];

                var nodeDelta = (expectedOutput - output) * output * (1 - output);
                var delta = -1 * netInput * nodeDelta;

                connection.UpdateWeight(_learningRate, delta);

                neuron.PreviousPartialDerivate = nodeDelta;
            });
        });
    }

    /// <summary>
    /// Hellper function that runs backpropagation algorithm on the hidden layer of the network.
    /// </summary>
    /// <param name="row">
    /// Input/Expected output row.
    /// </param>
    private void HandleHiddenLayers()
    {
        Console.WriteLine($"[SimpleNeuralNetwork.HandleHiddenLayers] ()");
        for (int k = _layers.Count - 2; k > 0; k--)
        {
            _layers[k].Neurons.ForEach(neuron =>
            {
                neuron.Inputs.ForEach(connection =>
                {
                    var output = neuron.CalculateOutput();
                    var netInput = connection.GetOutput();
                    double sumPartial = 0;

                    _layers[k + 1].Neurons
                    .ForEach(outputNeuron =>
                    {
                        outputNeuron.Inputs.Where(i => i.IsFromNeuron(neuron.Id))
                        .ToList()
                        .ForEach(outConnection =>
                        {
                            sumPartial += outConnection.PreviousWeight * outputNeuron.PreviousPartialDerivate;
                        });
                    });

                    var delta = -1 * netInput * sumPartial * output * (1.018 - output);
                    connection.UpdateWeight(_learningRate, delta);
                });
            });
        }

        Console.WriteLine($"[SimpleNeuralNetwork.HandleHiddenLayers] ...execution completed...");
    }
}