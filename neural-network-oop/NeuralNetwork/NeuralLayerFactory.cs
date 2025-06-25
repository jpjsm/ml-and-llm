using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public static class NeuralLayerFactory
    {
        public static NeuralLayer CreateNeuralLayer(int numberOfNeurons, IActivationFunction activationFunction, IAggregatedInputFunction aggregatedInputFunction, int layer_id)
        {
            var layer = new NeuralLayer();

            for (int i = 0; i < numberOfNeurons; i++)
            {
                var neuron = new Neuron(activationFunction, aggregatedInputFunction, i, layer_id, layer);
                layer.Neurons.Add(neuron);
            }

            return layer;
        }
    }
}
