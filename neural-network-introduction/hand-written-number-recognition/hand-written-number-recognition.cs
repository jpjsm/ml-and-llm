// See https://aka.ms/new-console-template for more information
using NeuralNet;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System;
using System.Linq;

namespace HandWrittenNumberRecognition
{
    public class Application
    {
        public static int[] Slopes= [3];// [0, 1, 2, 3, 4, 5, 6, 7];
        public static double[] Thresholds = [0.5];// [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        public static int[] Hidden_Nodes_Values = [200];// [50, 75, 100, 150, 200, 300, 400];
        public static double[] Learning_Rate_Values = [0.3];// [0.05, 0.075, 0.09, 0.1, 0.2, 0.3, 0.5, 0.75];
        public static int[] Epochs_Values = [5];// [1, 2, 3, 5, 8, 13];
        public static int Train_Set_Size = 200;//5000;
        public static int Test_Set_Size = 10;//500;

        public static void Main()
        {
            (int[] scorecard, TimeSpan training_duration, TimeSpan test_duration) evaluated_scenario;
            Console.WriteLine("Start hand-written digits recognition");
            Mnist_dataset.Load_Mnist_dataset("./mnist_dataset/mnist_train.csv", "./mnist_dataset/mnist_test.csv");

            using (StreamWriter outputFile = new StreamWriter("hand-written-number-recognition.csv"))
            {
                outputFile.WriteLine($"\"hidden_nodes\",\"learning_rate\",\"slope\",\"threshold\",\"epochs\",\"accuracy\",\"train_set_length\",\"test_set_length\",\"training_duration\",\"test_duration\"");
                outputFile.Flush();
                foreach (int hidden_nodes in Hidden_Nodes_Values)
                {
                    foreach (double learning_rate in Learning_Rate_Values)
                    {
                        foreach(int slope in Slopes)
                        {
                            foreach (double threshold in Thresholds)
                            {
                                foreach (int epochs in Epochs_Values)
                                {
                                    Console.WriteLine($"Scenario: 'hidden_nodes': {hidden_nodes}, 'learning_rate': {learning_rate}, 'slope': {slope}, 'threshold': {threshold}, 'epochs': {epochs}");

                                    Mnist_data_row[] train_set = Mnist_dataset.Dataset_sample(Dataset_types.Train, Train_Set_Size);
                                    Mnist_data_row[] test_set = Mnist_dataset.Dataset_sample(Dataset_types.Test, Test_Set_Size);
                                    evaluated_scenario = Scenario_Evaluation(train_set, test_set, hidden_nodes, learning_rate, slope, threshold, epochs);
                                    double accuracy = Convert.ToDouble(evaluated_scenario.scorecard.Sum())/Convert.ToDouble(evaluated_scenario.scorecard.Length);
                                    Console.WriteLine($"{accuracy}, {evaluated_scenario.training_duration}, {evaluated_scenario.test_duration}");
                                    outputFile.WriteLine($"{hidden_nodes},{learning_rate},{slope},{threshold},{epochs},{accuracy},{train_set.Length},{test_set.Length},{evaluated_scenario.training_duration},{evaluated_scenario.test_duration}");
                                    outputFile.Flush();
                                }                            
                            }
                        }
                    }
                }
            }

            Console.WriteLine("Finished hand-written digits recognition");
        }

        private static (int[] scorecard, TimeSpan training_duration, TimeSpan test_duration) Scenario_Evaluation(
            Mnist_data_row[] train_set,
            Mnist_data_row[] test_set,
            int hidden_nodes,
            double learning_rate,
            int slope,
            double threshold,
            int epochs)
        {
            Console.WriteLine($"[Scenario_Evaluation] (train_set size:{train_set.Length}, test_set size:{test_set.Length}, hidden_nodes: {hidden_nodes}, learning_rate: {learning_rate}, slope: {slope}, threshold: {threshold}, epochs: {epochs})");
            Dictionary<string, List<double>> expected_results = new(Mnist_dataset.Expected_Results);
            // Neural Network definition
            SimpleNeuralNetwork network = new SimpleNeuralNetwork(Mnist_dataset.Image_Pixels);

            var layerFactory = new NeuralLayerFactory();
            network.AddLayer(layerFactory.CreateNeuralLayer(hidden_nodes, new SigmoidActivationFunction(slope,threshold), new WeightedSumFunction()));
            network.AddLayer(layerFactory.CreateNeuralLayer(10, new SigmoidActivationFunction(slope,threshold), new WeightedSumFunction()));

            // Training Neural Network
            Console.WriteLine($"[Scenario_Evaluation] ...Training Neural Network...");
            DateTime training_start = DateTime.Now;
            network.Train_MNIST(train_set.Select(ts => (ts.Expected_label, ts.Standardized_Pixel_Values)).ToArray(), expected_results, learning_rate, epochs);
            TimeSpan training_duration = DateTime.Now - training_start;

            // Evaluate Neural Network performance
            Console.WriteLine($"[Scenario_Evaluation] ...Evaluate Neural Network performance...");
            int[] scorecard = new int[test_set.Length];
            for (int i = 0; i < test_set.Length; i++) scorecard[i] = int.MinValue; 
            DateTime test_start = DateTime.Now;
            for (int i = 0; i < test_set.Length; i++)
            {
                // Run query
                network.PushInputValues(test_set[i].Standardized_Pixel_Values.ToArray());
                List<double> actual_output = network.GetOutput();

                // Check if max output matches the expected max
                var indexAtMax = actual_output.IndexOf(actual_output.Max());

                scorecard[i] = indexAtMax == int.Parse(test_set[i].Expected_label) ? 1 : 0;
                Console.WriteLine($"[Scenario_Evaluation] Test row results: expected label: '{test_set[i].Expected_label}' <--> '{indexAtMax}' actual label; actual_output: {string.Join(", ", actual_output.Select(o => o.ToString()))}");
            }

            TimeSpan test_duration = DateTime.Now - test_start;

            Console.WriteLine($"[Scenario_Evaluation] ...execution completed...");
            return (scorecard, training_duration, test_duration);
        }

    }

}

