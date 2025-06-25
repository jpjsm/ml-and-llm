namespace NeuralNet;

public class BipolarSigmoidActivationFunction : IActivationFunction
{
    private double _coeficient;

    public BipolarSigmoidActivationFunction(double coeficient)
    {
        _coeficient = coeficient;
    }

    public double CalculateOutput(double input)
    {
        return ((1 - Math.Exp(-input * _coeficient)) / (1 + Math.Exp(-input * _coeficient)));
    }
}