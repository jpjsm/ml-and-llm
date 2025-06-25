namespace NeuralNet;

public class WeightedSumFunction : IInputFunction
{
    public double CalculateInput(List<ISynapse> inputs)
    {
        //Console.WriteLine($"[WeightedSumFunction.CalculateInput] (inputs count: {inputs.Count})");
        double calculated_input = inputs.Select(x => x.Weight * x.GetOutput()).Sum();

        //Console.WriteLine($"[WeightedSumFunction.CalculateInput] Calculated input: {calculated_input}");
        return calculated_input;
    }
}
