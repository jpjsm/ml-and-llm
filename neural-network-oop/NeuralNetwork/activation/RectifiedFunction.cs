namespace NeuralNetwork.activation;

public class RectifiedFuncion : IActivationFunction
{
    public double CalculateOutput(double input)
    {
        if (double.IsNaN(input))
            throw new ArgumentException("'input' cannot be undefined or NaN");
        return Math.Max(0, input);
    }
}