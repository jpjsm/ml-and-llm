using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralLayer
    {
        public List<INeuron> Neurons;

        public NeuralLayer()
        {
            Neurons = new List<INeuron>();
        }

        /// <summary>
        /// Connecting two layers.
        /// </summary>
        public void ConnectLayers(NeuralLayer inputLayer)
        {
            Neurons.SelectMany(neuron => inputLayer.Neurons, (neuron, input) => new { neuron, input })
                   .ToList()
                   .ForEach(x => x.neuron.AddInputNeuron(x.input));
        }
    }
}
