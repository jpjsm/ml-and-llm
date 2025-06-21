namespace NeuralNet;

public class SigmoidActivationFunction : IActivationFunction
{
    private double coeficient;
    private double threshold;

    public SigmoidActivationFunction(int slope=3, double threshold=0.5)
    {
        int[] valid_slopes = [0, 1, 2, 3, 4, 5, 6, 7];
        slope = valid_slopes.Contains(slope) ? slope : 3;
        coeficient = Math.Pow(2.0, slope);
        this.threshold = threshold;
    }

    public double CalculateOutput(double input)
    {
        //Console.WriteLine($"[SigmoidActivationFunction.CalculateOutput] (input: {input})");
        double output = 1.0 / (1.0 + Math.Exp(-coeficient*(input - threshold)));

        //Console.WriteLine($"[SigmoidActivationFunction.CalculateOutput] Output: {output})");
        return output;
    }
}