using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.activation
{

    public class SigmoidFunction : IActivationFunction
    {
        private double coeficient;
        private double bias;

        public SigmoidFunction(int slope = 3, double bias = 0.5)
        {
            if (double.IsNaN(bias))
                throw new ArgumentException("'bias' cannot be undefined or NaN");

            int[] valid_slopes = [0, 1, 2, 3, 4, 5, 6, 7];
            slope = valid_slopes.Contains(slope) ? slope : 3;
            coeficient = Math.Pow(2.0, slope);
            this.bias = bias;
        }

        public double CalculateOutput(double input)
        {
            double output = 1.0 / (1.0 + Math.Exp(-coeficient * (input - bias)));

            return output;
        }
    }
}
