namespace NeuralNet;

public interface IInputFunction
{
    double CalculateInput(List<ISynapse> inputs);
}