using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.activation
{

    public class BipolarSigmoidFunction : IActivationFunction
    {
        private double coeficient;
        private double bias;

        public BipolarSigmoidFunction(int slope = 3, double bias = 0.0)
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
            if (double.IsNaN(input))
                throw new ArgumentException("'input' cannot be undefined or NaN");

            if (input == double.MinValue)
                return -1.0;

            if (input == double.MaxValue)
                return 1.0;

            double numerator = 1.0 - Math.Exp(-coeficient * (input - bias));
            double denominator = 1.0 + Math.Exp(-coeficient * (input - bias));

            if (double.IsNegativeInfinity(numerator) && double.IsInfinity(denominator))
                return -1.0;

            double output = numerator / denominator;

            if(double.IsNaN(output))
                throw new ArgumentException($"Function error for input = {input}: ({numerator})/({denominator}) => NaN");


            return output;
        }
    }
}
