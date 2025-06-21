namespace NeuralNetwork.aggregation;

public class WeightedSumFunction : IAggregatedInputFunction
{
    public double CalculateInput(List<ISynapse> inputs)
    {
        if (inputs == null || inputs.Count == 0)
            throw new ArgumentException("'inputs' cannot be null or have zero elements.");

        //double calculated_input = inputs.Select(x => x.Weight * x.GetOutput()).Sum();
        double calculated_input = 0.0;
        foreach (var input in inputs) 
        {
            calculated_input += input.Weight * input.GetOutput();
        }

        return calculated_input;
    }
}
