using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public interface IAggregatedInputFunction
    {
        double CalculateInput(List<ISynapse> inputs);
    }
}
