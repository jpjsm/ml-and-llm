using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Synapse : ISynapse
    {
        private static Random rnd = new Random();
        private INeuron fromNeuron;
        private INeuron toNeuron;

        public double Weight { get ; set; }
        public double PreviousWeight { get ; set ; }

        public INeuron FromNeuron { get { return fromNeuron; } }
        public INeuron ToNeuron { get { return toNeuron; } }

        public Synapse(INeuron fromNeuron, INeuron toNeuron, double weight)
        {
            ArgumentNullException.ThrowIfNull(fromNeuron);
            ArgumentNullException.ThrowIfNull(toNeuron);
            if (double.IsNaN(weight))
                throw new ArgumentException("'weight' cannot be undefined or NaN");

            this.fromNeuron = fromNeuron;
            this.toNeuron = toNeuron;

            Weight = weight;
            PreviousWeight = double.NaN;
        }

        public Synapse(INeuron fromNeuron, INeuron toNeuron)
        {
            ArgumentNullException.ThrowIfNull(fromNeuron);
            ArgumentNullException.ThrowIfNull(toNeuron);

            this.fromNeuron = fromNeuron;
            this.toNeuron = toNeuron;

            Weight = rnd.NextDouble();//Math.Abs(rnd.NextDouble() - 0.5) + 0.1;  //0.5;// rndgauss.NextDouble(); //

            PreviousWeight = double.NaN;
        }

        public bool IsFromNeuron(string fromNeuronId)
        {
            return fromNeuron.Id == fromNeuronId;
        }

        public void UpdateWeight(double learningRate, double delta)
        {
            PreviousWeight = Weight;
            Weight += learningRate * delta;
        }

        public double GetOutput()
        {
            return fromNeuron.CalculateOutput();
        }
    }
}
