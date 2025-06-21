using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class InputSynapse : ISynapse
    {
        private readonly INeuron toNeuron;
        public double Weight { get; set; }
        public double PreviousWeight { get; set; }
        public double Output { get; set; }


        public InputSynapse(INeuron toNeuron)
        {
            this.toNeuron = toNeuron;
            Weight = 1.0;
            PreviousWeight = 1.0;
        }
        public InputSynapse(INeuron toNeuron, double output)
        {
            this.toNeuron = toNeuron;
            Weight = 1.0;
            PreviousWeight = 1.0;
            Output = output;
        }

        public bool IsFromNeuron(string fromNeuronId)
        {
            return false;
        }

        public void UpdateWeight(double learningRate, double delta)
        {
            throw new InvalidOperationException("It is not allowed to call this method on Input Connection");
        }

        public double GetOutput()
        {
            return Output;
        }
    }
}
