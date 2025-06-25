// See https://aka.ms/new-console-template for more information
using NeuralNet;


namespace sigmoid_analysis
{
    internal class Application
    {

        public static void Main()
        {
            Console.WriteLine("Hello, Sigmoid Analysis!");
            int[] slopes= [0, 1, 2, 3, 4, 5, 6, 7];
            double[] thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

            int resolution = 100; 
            using (StreamWriter outputFile = new StreamWriter("sigmoid-analysis.csv"))
            {
                outputFile.WriteLine($"\"slope\",\"threshold\",\"x\",\"sigmoid\"");
                foreach (int slope in slopes)
                {
                    foreach (double threshold in thresholds)
                    {
                        SigmoidActivationFunction sigmoid = new SigmoidActivationFunction(slope, threshold);

                        for (int i = 0; i <= resolution; i++)
                        {
                            double x = (i*1.0)/resolution;
                            outputFile.WriteLine($"{slope},{threshold:F1},{x:F4},{sigmoid.CalculateOutput(x):F6}");
                        }
                    }
                }
            }

            Console.WriteLine("Goodbye, Sigmoid Analysis!");
        }
    }
}
