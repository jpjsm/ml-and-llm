using NeuralNetwork.activation;

namespace nnoopUnitTests
{
    public class UnitTestActivation
    {
        private static readonly List<double> biases = [];

        static UnitTestActivation()
        {
            // Generate negative bias values, in descending order
            for (int i = 32; i >= -20; i--)
            {
                biases.Add(-Math.Pow(2, i));
            }

            biases.Add(0);

            // Generate positive bias values, in ascending order
            for (int i = -20; i <= 32; i++)
            {
                biases.Add(Math.Pow(2, i));
            }
        }

        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void TestBipolarSigmoidFunction()
        {
            var expected = (double input, int slope, double bias) =>
            {
                if (input == double.MinValue)
                    return -1.0;

                if (input == double.MaxValue)
                    return 1.0;
                double coeficient = Math.Pow(2.0, slope);
                double numerator = 1.0 - Math.Exp(-coeficient * (input - bias));
                double denominator = 1.0 + Math.Exp(-coeficient * (input - bias));

                if (double.IsNegativeInfinity(numerator) && double.IsInfinity(denominator))
                    return -1.0;

                double output = numerator / denominator;

                if (double.IsNaN(output))
                    throw new ArgumentException($"Function error for input = {input}: ({numerator})/({denominator}) => NaN");


                return output;
            };

            // bipolarsigmoid function throws exception for NaN bias
            var ex = Assert.Throws<ArgumentException>(() => new BipolarSigmoidFunction(0, double.NaN)); ;
            Assert.That(ex.Message, Is.EqualTo("'bias' cannot be undefined or NaN"));

            int[] slopes = [0, 1, 2, 3, 4, 5, 6, 7];

            foreach (int slope in slopes)
            {
                foreach (double bias in biases)
                {
                    BipolarSigmoidFunction bipolarsigmoid = new BipolarSigmoidFunction(slope, bias);

                    // bipolarsigmoid function at negative infinite
                    double _expected = expected(double.MinValue, slope, bias);
                    Assert.That(bipolarsigmoid.CalculateOutput(double.MinValue), Is.EqualTo(-1.0).Within(1E-15));

                    // bipolarsigmoid function for values to the left of bias
                    for (int i = 100; i >= -15; i--)
                    {
                        double input = Convert.ToDouble(bias) - Math.Pow(10, i);
                        Assert.That(bipolarsigmoid.CalculateOutput(input), Is.EqualTo(expected(input, slope, bias)).Within(1E-15));
                    }

                    // bipolarsigmoid function at bias
                    Assert.That(bipolarsigmoid.CalculateOutput(bias), Is.EqualTo(0.0).Within(1E-15));

                    // bipolarsigmoid function for values to the right of bias
                    int lower_i = bias == 0 ? 21 : 15 - (int)Math.Log10(Math.Abs((double)bias));
                    for (int i = 100; i >= lower_i; i--)
                    {
                        double input = Convert.ToDouble(bias) - Math.Pow(10, i);
                        Assert.That(bipolarsigmoid.CalculateOutput(input), Is.EqualTo(expected(input, slope, bias)).Within(1E-15));
                    }

                    // sigmoid function at positive infinite
                    Assert.That(bipolarsigmoid.CalculateOutput(double.MaxValue), Is.EqualTo(1.0).Within(1E-15));
                }

            }
        }

        [Test]
        public void TestSigmoidFunction()
        {
            static double expected(double input, int slope, double bias) => 1.0 / (1.0 + Math.Exp(-Math.Pow(2.0, slope) * (input - bias)));
            // sigmoid function throws exception for NaN bias
            var ex = Assert.Throws<ArgumentException>(() => new SigmoidFunction(0, double.NaN)); ;
            Assert.That(ex.Message, Is.EqualTo("'bias' cannot be undefined or NaN"));

            int[] slopes = [0, 1, 2, 3, 4, 5, 6, 7];

            foreach (int slope in slopes)
            {
                foreach (double bias in biases)
                {
                    SigmoidFunction sigmoid = new SigmoidFunction(slope, bias);

                    // sigmoid function at negative infinite
                    Assert.That(sigmoid.CalculateOutput(double.MinValue), Is.EqualTo(0.0).Within(1E-15));

                    // sigmoid function for values to the left of bias
                    for (int i = 100; i >= -15; i--)
                    {
                        double input = Convert.ToDouble(bias) - Math.Pow(10, i);
                        Assert.That(sigmoid.CalculateOutput(input), Is.EqualTo(expected(input, slope, bias)).Within(1E-15));
                    }

                    // sigmoid function at bias
                    Assert.That(sigmoid.CalculateOutput(bias), Is.EqualTo(0.5).Within(1E-15));

                    // sigmoid function for values to the right of bias
                    int lower_i = bias == 0 ? 21 : 15 - (int)Math.Log10(Math.Abs((double)bias));
                    for (int i = 100; i >= lower_i; i--)
                    {
                        double input = Convert.ToDouble(bias) - Math.Pow(10, i);
                        Assert.That(sigmoid.CalculateOutput(input), Is.EqualTo(expected(input, slope, bias)).Within(1E-15));
                    }

                    // sigmoid function at positive infinite
                    Assert.That(sigmoid.CalculateOutput(double.MaxValue), Is.EqualTo(1.0).Within(1E-15));
                }

            }
        }

        [Test]
        public void TestStepFunction()
        {
            // sigmoid function throws exception for NaN bias
            var ex = Assert.Throws<ArgumentException>(() =>  new StepFunction(double.NaN)); ;
            Assert.That(ex.Message, Is.EqualTo("'bias' cannot be undefined or NaN"));

            foreach (double bias in biases)
            {
                StepFunction step = new StepFunction(bias);

                // sigmoid function at negative infinite
                Assert.That(step.CalculateOutput(double.MinValue), Is.EqualTo(0.0).Within(1E-15));

                // sigmoid function for values to the left of bias
                for (int i = 100; i >= -15; i--)
                    Assert.That(step.CalculateOutput(Convert.ToDouble(bias) - Math.Pow(10, i)), Is.EqualTo(0.0).Within(1E-15));

                // sigmoid function at bias
                Assert.That(step.CalculateOutput(bias), Is.EqualTo(0.0).Within(1E-15));

                // sigmoid function for values to the right of bias
                int lower_i = bias == 0 ? 21: 15 - (int)Math.Log10(Math.Abs((double)bias));
                for (int i = 100; i >= lower_i; i--)
                    Assert.That(step.CalculateOutput(Convert.ToDouble(bias) + Math.Pow(10, i)), Is.EqualTo(1.0).Within(1E-15));

                // sigmoid function at positive infinite
                Assert.That(step.CalculateOutput(double.MaxValue), Is.EqualTo(1.0).Within(1E-15));
            }

        }

        [Test]
        public void TestRectifiedFuncion()
        {
            RectifiedFuncion activation = new RectifiedFuncion();
            Assert.That(activation.CalculateOutput(double.MinValue), Is.EqualTo(0.0).Within(1E-15));
            for (int i = 100; i >= -100; i--)
                Assert.That(activation.CalculateOutput(-Math.Pow(10, i)), Is.EqualTo(0.0).Within(1E-15));
            Assert.That(activation.CalculateOutput(0.0), Is.EqualTo(0.0).Within(1E-15));

            for (int i = -100; i <= 100; i++)
            {
                double expected = Math.Pow(10, i);
                Assert.That(activation.CalculateOutput(Math.Pow(10, i)), Is.EqualTo(expected).Within(1E-15));
            }
            Assert.That(activation.CalculateOutput(double.MaxValue), Is.EqualTo(double.MaxValue).Within(1E-15));
        }
    }
}