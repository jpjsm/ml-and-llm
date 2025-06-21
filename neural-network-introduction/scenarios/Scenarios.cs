// See https://aka.ms/new-console-template for more information
using NeuralNet;

namespace scenarios
{
    internal class Application
    {

        public static void Main()
        {

        Console.WriteLine("Hello, World!");

        var network = new SimpleNeuralNetwork(3);

        var layerFactory = new NeuralLayerFactory();
        network.AddLayer(layerFactory.CreateNeuralLayer(3, new RectifiedActivationFuncion(), new WeightedSumFunction()));
        network.AddLayer(layerFactory.CreateNeuralLayer(1, new SigmoidActivationFunction(0.7), new WeightedSumFunction()));

        network.PushExpectedValues(
            new double[][] {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 0 },
                new double[] { 0 },
            });

        network.Train(
            new double[][] {
                new double[] { 150, 2, 0 },
                new double[] { 1002, 56, 1 },
                new double[] { 1060, 59, 1 },
                new double[] { 200, 3, 0 },
                new double[] { 300, 3, 1 },
                new double[] { 120, 1, 0 },
                new double[] { 80, 1, 0 },
            },
            learning_rate: 0.25,
            numberOfEpochs: 10000);

        network.PushInputValues(new double[] { 1054, 54, 1 });
        var outputs = network.GetOutput();

        foreach (double output in outputs)
        {
            Console.WriteLine($@"{output.ToString()}");
        }
        
        Console.WriteLine("Hey World... we're done here!");
        }
    }
}