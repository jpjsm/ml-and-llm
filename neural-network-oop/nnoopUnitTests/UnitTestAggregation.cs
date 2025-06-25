using NeuralNetwork;
using NeuralNetwork.activation;
using NeuralNetwork.aggregation;

namespace nnoopUnitTests
{
    public class UnitTestAggregation
    {
        private static readonly List<double> biases = [];

        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void TestWeightedSumFunction()
        {
            List<ISynapse> null_list = null;
            var ex = Assert.Throws<ArgumentException>(() => (new WeightedSumFunction()).CalculateInput(null_list)); ;
            Assert.That(ex.Message, Is.EqualTo("'inputs' cannot be null or have zero elements."));

            List<ISynapse> empty_list = new List<ISynapse>();
            ex = Assert.Throws<ArgumentException>(() => (new WeightedSumFunction()).CalculateInput(empty_list)); ;
            Assert.That(ex.Message, Is.EqualTo("'inputs' cannot be null or have zero elements."));

            var layer = new NeuralLayer();

            List<InputSynapse> input_synapses = new List<InputSynapse>() { 
                new InputSynapse(new Neuron(new RectifiedFuncion(), new WeightedSumFunction(), 0, 0, layer), 0.2),
                new InputSynapse(new Neuron(new RectifiedFuncion(), new WeightedSumFunction(), 0, 1, layer), 0.4)
            };

            WeightedSumFunction weightedSumFunction = new WeightedSumFunction();
            var weightedSum = weightedSumFunction.CalculateInput(input_synapses.Select(s => (ISynapse)s).ToList());
            Assert.That(weightedSum, Is.EqualTo(0.6).Within(1.0E-15));
        }
    }
}