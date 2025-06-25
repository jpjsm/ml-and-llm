using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Neuron : INeuron
    {
        private static Random rnd = new Random();

        private IActivationFunction activationFunction;
        private IAggregatedInputFunction aggregatedInputFunction;
        private NeuralLayer parentLayer;
        public List<ISynapse> Inputs { get; set; }
        public List<ISynapse> Outputs { get; set; }

        public string Id { get; set; }

        public double PreviousPartialDerivate { get; set; }

        public Neuron(IActivationFunction activationFunction, IAggregatedInputFunction aggregatedInputFunction, int row, int layer, NeuralLayer parentLayer)
        {
            this.activationFunction = activationFunction;
            this.aggregatedInputFunction = aggregatedInputFunction;
            Inputs = new List<ISynapse>();
            Outputs = new List<ISynapse>();
            Id = $"L{layer}, R{row,4}";
            PreviousPartialDerivate = 0.0;
            this.parentLayer = parentLayer;
        }


        public void AddInputSynapse(double inputValue)
        {
            InputSynapse inputSynapse = new InputSynapse(this, inputValue);
            Inputs.Add(inputSynapse);
        }

        public void AddInputNeuron(INeuron inputNeuron)
        {
            double initialWeight = rnd.NextDouble()/(parentLayer.Neurons.Count + 1.0);
            Synapse synapse = new Synapse(inputNeuron, this, initialWeight);
            Inputs.Add(synapse);
            inputNeuron.Outputs.Add(synapse);
        }
        public void AddOutputNeuron(INeuron outputNeuron)
        {
            double initialWeight = rnd.NextDouble() / (parentLayer.Neurons.Count + 1.0);
            Synapse synapse = new Synapse(this, outputNeuron, initialWeight);
            Outputs.Add(synapse);
            outputNeuron.Inputs.Add(synapse);
        }

        public double CalculateOutput()
        {
            double aggregatedInput = aggregatedInputFunction.CalculateInput(this.Inputs);
            double calculatedOutput = activationFunction.CalculateOutput(aggregatedInput);
            return calculatedOutput;
        }

        public void PushValueOnInput(double inputValue)
        {
            ((InputSynapse)Inputs.First()).Output = inputValue;
        }
    }
}
