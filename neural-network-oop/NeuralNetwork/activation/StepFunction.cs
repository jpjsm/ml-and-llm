using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.activation
{
    public class StepFunction : IActivationFunction
    {
        private double bias;

        public StepFunction(double bias)
        {
            if (double.IsNaN(bias))
                throw new ArgumentException("'bias' cannot be undefined or NaN");
            this.bias = bias;
        }

        public double CalculateOutput(double input)
        {
            if (double.IsNaN(input))
                throw new ArgumentException("'input' cannot be undefined or NaN");
            return Convert.ToDouble(input > bias);
        }
    }
}
