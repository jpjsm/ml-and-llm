using System.Dynamic;
using System.Linq;
using System.Reflection.Emit;
using System.Text.Json;
using System.Text.Json.Nodes;
using NN = NeuralNetwork;
using NeuralNetwork.activation;
using NeuralNetwork.aggregation;

namespace mini_pictures
{
    internal class MiniPictures
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, Mini Pictures!");
            string train_set_str = File.ReadAllText(@"C:\mini-pictures\minipictures_train.json");
            string test_set_str =  File.ReadAllText(@"C:\mini-pictures\minipictures_test.json");

            JsonArray train_set_jsonarray = JsonSerializer.Deserialize<JsonArray>(train_set_str);
            JsonArray test_set_jsonarray = JsonSerializer.Deserialize<JsonArray>(test_set_str);

            var foo = train_set_jsonarray.Select(e => new List<double>(e["Inputs"].AsArray().Select(i => i.GetValue<double>())).ToArray()).ToArray(); // e["Inputs"].AsArray()).Select(a => a.ToArray<double>());//.AsArray()).ToArray();//.GetValue<double[]>()).ToArray();

            (char Label, double[] Inputs, double[] Expected_output)[] train_set = train_set_jsonarray.Select(e => (e["Label"].GetValue<char>(), new List<double>(e["Inputs"].AsArray().Select(i => i.GetValue<double>())).ToArray(), new List<double>(e["Expected_output"].AsArray().Select(i => i.GetValue<double>())).ToArray())).ToArray();
            (char Label, double[] Inputs, double[] Expected_output)[] test_set = test_set_jsonarray.Select(e => (e["Label"].GetValue<char>(), new List<double>(e["Inputs"].AsArray().Select(i => i.GetValue<double>())).ToArray(), new List<double>(e["Expected_output"].AsArray().Select(i => i.GetValue<double>())).ToArray())).ToArray();

            NN.NeuralNetwork network = new NN.NeuralNetwork(9, 0.25);


            network.AddLayer(NN.NeuralLayerFactory.CreateNeuralLayer(7, new SigmoidFunction(slope: 1, bias: 0.5), new WeightedSumFunction(), network.layers.Count));
            network.AddLayer(NN.NeuralLayerFactory.CreateNeuralLayer(4, new SigmoidFunction(slope: 0, bias: 0.5), new WeightedSumFunction(), network.layers.Count));

            // Prepare Neural Network for training
            double[][] training_inputs = train_set.Select(t => t.Inputs).ToArray();
            double[][] training_expected_outputs = train_set.Select(t => t.Expected_output).ToArray();
            network.PushExpectedOutputs(training_expected_outputs);


            Console.WriteLine($"...Training Neural Network...");
            // Training Neural Network
            int epochs = 15;
            DateTime training_start = DateTime.Now;
            network.Train(training_inputs, 0.1, epochs);
            TimeSpan training_duration = DateTime.Now - training_start;
            Console.WriteLine($"Neural Network training time: {training_duration}");
            Console.WriteLine($"Neural Network avg test train time: {training_duration.TotalMicroseconds/(training_inputs.Length * epochs)} µSecs");

            // Evaluate Neural Network performance
            Console.WriteLine($"...Evaluate Neural Network performance...");
            int[] scorecard = new int[test_set.Length];
            for (int i = 0; i < test_set.Length; i++)
                scorecard[i] = int.MinValue;

            double[][] testing_inputs = test_set.Select(t => t.Inputs).ToArray();
            DateTime test_start = DateTime.Now;
            for (int i = 0; i < testing_inputs.Length; i++)
            {
                // Run query
                network.PushInputValues(testing_inputs[i]);
                double[] actual_output = network.GetOutput().ToArray();

                // Check if max output matches the expected max
                int actual_indexAtMax = Array.IndexOf(actual_output, actual_output.Max());
                int expected_IndexAtMax = Array.IndexOf(test_set[i].Expected_output, test_set[i].Expected_output.Max());


                scorecard[i] = actual_indexAtMax == expected_IndexAtMax ? 1 : 0;
                Console.WriteLine($"Test row results: expected label: '{expected_IndexAtMax}' <--> '{actual_indexAtMax}' actual label; actual_output: {string.Join(", ", actual_output.Select(o => o.ToString("e3")))}");
            }

            double accuracy = Convert.ToDouble(scorecard.Sum()) / Convert.ToDouble(scorecard.Length);
            Console.WriteLine($"Accuracy: {accuracy}");


            Console.WriteLine("Goodbye, Mini Pictures!");
        }
    }
}
