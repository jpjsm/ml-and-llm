using NeuralNetwork.activation;
using NeuralNetwork.aggregation;

namespace NeuralNetwork;

public class NeuralNetwork
{
    public List<NeuralLayer> layers;
    public double learningRate;
    public double[][] expectedOutputs = Array.Empty<double[]>();

    public NeuralNetwork(int numberOfInputNeurons, double learingRate)
    {
        layers = new List<NeuralLayer>();
        this.learningRate = learingRate;

        CreateInputLayer(numberOfInputNeurons);
    }

    public void AddLayer(NeuralLayer newlayer)
    {
        if (layers.Any())
        {
            var last_layer = layers.Last();
            newlayer.ConnectLayers(last_layer);
        }

        layers.Add(newlayer);
    }

    public void PushInputValues(double[] inputValues)
    {
        if (inputValues == null || inputValues.Length != layers.First().Neurons.Count)
            throw new ArgumentException($"'inputValues' is NULL or length different than input neurons count:{layers.First().Neurons.Count}.");

        for (int i = 0; i < layers.First().Neurons.Count; i++)
        {
            layers.First().Neurons[i].PushValueOnInput(inputValues[i]);
        }
    }

    public void PushExpectedOutputs(double[][] expectedOutputs)
    {
        this.expectedOutputs = expectedOutputs;
    }

    public List<double> GetOutput()
    {
        var returnValues = new List<double>();

        foreach (var outputNeuron in layers.Last().Neurons)
        {
            returnValues.Add(outputNeuron.CalculateOutput());
        }

        return returnValues;
    }

    public void Train(double[][] inputs, double learning_rate, int epochs)
    {
        this.learningRate = learning_rate;

        for (int i = 0; i < epochs; i++)
        {
            for (int j = 0; j < inputs.GetLength(0); j++)
            {
                PushInputValues(inputs[j]);

                List<double> outputs = GetOutput();

                List<double> errors = CalculateErrors(outputs, j);

                HandleOutputLayer(outputs, errors);
                HandleHiddenLayers();
            }
        }
    }

    private void HandleOutputLayer(List<double> outputs, List<double> errors)
    {
        var output_neurons = layers.Last().Neurons;

        for (int i = 0; i < output_neurons.Count; i++)
        {
            double nodeDelta = errors[i] * outputs[i] * (1 - outputs[i]);

            foreach (var synapse in output_neurons[i].Inputs)
            {
                var netInput = synapse.GetOutput();
                var delta = -1.0 * netInput * nodeDelta;
                synapse.UpdateWeight(learningRate, delta);
            }

            output_neurons[i].PreviousPartialDerivate = nodeDelta;
        }
    }

    private void HandleHiddenLayers()
    {
        for (int i = layers.Count - 2; i >0; i--)
        {
            foreach (var neuron in layers[i].Neurons)
            {
                var output = neuron.CalculateOutput();

                foreach (var input_synapse in neuron.Inputs)
                {
                    var netInput = input_synapse.GetOutput();

                    double sumPartial = 0.0;

                    foreach (var output_neuron in layers[i + 1].Neurons)
                    {
                        foreach (var output_synapse in output_neuron.Inputs.Where(s => s.IsFromNeuron(neuron.Id)))
                        {
                            sumPartial += output_synapse.PreviousWeight * output_neuron.PreviousPartialDerivate;
                        }
                    }

                    var delta = -1.0 * netInput * sumPartial * output * (1.0 - output);
                    input_synapse.UpdateWeight(learningRate, delta) ;
                }
            }
        }
    }

    private List<double> CalculateErrors(List<double> actual_outputs, int expected_output_row)
    {
        if(actual_outputs == null ||  actual_outputs.Count != expectedOutputs[expected_output_row].Length)
            throw new ArgumentException($"'actual_outputs' is NULL or length different than expected output neurons count:{expectedOutputs[expected_output_row].Length}.");

        List<double> errors = [];

        for (int i = 0; i < actual_outputs.Count; i++)
        {
            errors.Add(expectedOutputs[expected_output_row][i] - actual_outputs[i]);
        }

        return errors;
    }

    private void CreateInputLayer(int numberOfInputNeurons)
    {
        var inputLayer = NeuralLayerFactory.CreateNeuralLayer(numberOfInputNeurons, new RectifiedFuncion(), new WeightedSumFunction(), layer_id: 0);

        foreach (INeuron neuron in inputLayer.Neurons)
        {
            neuron.AddInputSynapse(0.0);
        }

        this.AddLayer(inputLayer);
    }
}
